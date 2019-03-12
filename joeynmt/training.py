# coding: utf-8

"""
Training module
"""

import argparse
import logging
import time
import os
import shutil
import random

import numpy as np

import torch
import torch.nn as nn

from joeynmt.model import build_model
from joeynmt.batch import Batch
from joeynmt.helpers import log_data_info, \
    load_config, log_cfg, store_attention_plots, load_model_from_checkpoint
from joeynmt.prediction import validate_on_data
from joeynmt.data import load_data, make_data_iter


# pylint: disable=too-many-instance-attributes
class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""

    def __init__(self, model, config):
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model:
        :param config:
        """
        train_config = config["training"]
        self.model = model
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction='sum')
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)
        if train_config["loss"].lower() not in ["crossentropy", "xent",
                                                "mle", "cross-entropy"]:
            raise NotImplementedError("Loss is not implemented. Only xent.")
        learning_rate = train_config.get("learning_rate", 3.0e-4)
        weight_decay = train_config.get("weight_decay", 0)
        if train_config["optimizer"].lower() == "adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(), weight_decay=weight_decay, lr=learning_rate)
        else:
            # default
            self.optimizer = torch.optim.SGD(
                model.parameters(), weight_decay=weight_decay, lr=learning_rate)
        self.schedule_metric = train_config.get("schedule_metric",
                                                "eval_metric")
        self.ckpt_metric = train_config.get("ckpt_metric", "eval_metric")
        self.best_ckpt_iteration = 0
        # if we schedule after BLEU/chrf, we want to maximize it, else minimize
        scheduler_mode = "max" if self.schedule_metric == "eval_metric" \
            else "min"
        # the ckpt metric decides on how to find a good early stopping point:
        # ckpts are written when there's a new high/low score for this metric
        if self.ckpt_metric == "eval_metric":
            self.best_ckpt_score = -np.inf
            self.is_best = lambda x: x > self.best_ckpt_score
        else:
            self.best_ckpt_score = np.inf
            self.is_best = lambda x: x < self.best_ckpt_score
        self.scheduler = None
        if "scheduling" in train_config.keys() and \
                train_config["scheduling"]:
            if train_config["scheduling"].lower() == "plateau":
                # learning rate scheduler
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=self.optimizer,
                    mode=scheduler_mode,
                    verbose=False,
                    threshold_mode='abs',
                    factor=train_config.get("decrease_factor", 0.1),
                    patience=train_config.get("patience", 10))
            elif train_config["scheduling"].lower() == "decaying":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer=self.optimizer,
                    step_size=train_config.get("decaying_step_size", 10))
            elif train_config["scheduling"].lower() == "exponential":
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer=self.optimizer,
                    gamma=train_config.get("decrease_factor", 0.99)
                )
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.batch_multiplier = train_config.get("batch_multiplier", 1)
        self.criterion = criterion
        self.normalization = train_config.get("normalization", "batch")
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_tokens = 0
        self.max_output_length = train_config.get("max_output_length", None)
        self.overwrite = train_config.get("overwrite", False)
        self.model_dir = self._make_model_dir(train_config["model_dir"])
        self.logger = self._make_logger()
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
        self.logging_freq = train_config.get("logging_freq", 100)
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.eval_metric = train_config.get("eval_metric", "bleu")
        self.print_valid_sents = train_config["print_valid_sents"]
        self.level = config["data"]["level"]
        self.clip_grad_fun = None
        if "clip_grad_val" in train_config.keys():
            clip_value = train_config["clip_grad_val"]
            self.clip_grad_fun = lambda params:\
                nn.utils.clip_grad_value_(parameters=params,
                                          clip_value=clip_value)
        elif "clip_grad_norm" in train_config.keys():
            max_norm = train_config["clip_grad_norm"]
            self.clip_grad_fun = lambda params:\
                nn.utils.clip_grad_norm_(parameters=params, max_norm=max_norm)

        assert not ("clip_grad_val" in train_config.keys() and
                    "clip_grad_norm" in train_config.keys()), \
            "you can only specify either clip_grad_val or clip_grad_norm"

        if "load_model" in train_config.keys():
            model_load_path = train_config["load_model"]
            self.logger.info("Loading model from %s", model_load_path)
            self.load_checkpoint(model_load_path)

        trainable_params = [n for (n, p) in self.model.named_parameters()
                            if p.requires_grad]
        self.logger.info("Trainable parameters: %s", trainable_params)
        assert trainable_params

    def save_checkpoint(self):
        """
        Save the model's current parameters and state to a checkpoint.

        :return:
        """
        model_path = "{}/{}.ckpt".format(self.model_dir, self.steps)
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if \
            self.scheduler is not None else None,
        }
        torch.save(state, model_path)

    def load_checkpoint(self, path):
        """
        Load a model from a given checkpoint file.

        :param path:
        :return:
        """
        model_checkpoint = load_model_from_checkpoint(
            path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])
        self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])

        if model_checkpoint["scheduler_state"] is not None:
            self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])

        # restore counts
        self.steps = model_checkpoint["steps"]
        self.total_tokens = model_checkpoint["total_tokens"]
        self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
        self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda()

    def _make_model_dir(self, model_dir):
        """
        Create a new directory for the model.

        :param model_dir:
        :return:
        """
        if os.path.isdir(model_dir):
            if not self.overwrite:
                raise FileExistsError(
                    "Model directory exists and overwriting is disabled.")
        else:
            os.makedirs(model_dir)
        return model_dir

    def _make_logger(self):
        """
        Create a logger for logging the training process.
        :return:
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.DEBUG)
        fh = logging.FileHandler(
            "{}/train.log".format(self.model_dir))
        fh.setLevel(level=logging.DEBUG)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(message)s')
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        logging.getLogger("").addHandler(sh)
        logger.info("Hello! This is Joey-NMT.")
        return logger

    def train_and_validate(self, train_data, valid_data):
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data:
        :param valid_data:
        :return:
        """
        train_iter = make_data_iter(train_data, batch_size=self.batch_size,
                                    train=True, shuffle=self.shuffle)
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)
            self.model.train()

            start = time.time()
            total_valid_duration = 0
            processed_tokens = self.total_tokens
            count = 0

            for batch in iter(train_iter):
                # reactivate training
                self.model.train()
                batch = Batch(batch, self.pad_index, use_cuda=self.use_cuda)

                # only update every batch_multiplier batches
                # see https://medium.com/@davidlmorton/
                # increasing-mini-batch-size-without-increasing-
                # memory-6794e10db672
                update = count == 0
                # print(count, update, self.steps)
                batch_loss = self._train_batch(batch, update=update)
                count = self.batch_multiplier if update else count
                count -= 1

                # log learning progress
                if self.model.training and self.steps % self.logging_freq == 0 \
                        and update:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - processed_tokens
                    self.logger.info(
                        "Epoch %d Step: %d Loss: %f Tokens per Sec: %f",
                        epoch_no + 1, self.steps, batch_loss,
                        elapsed_tokens / elapsed)
                    start = time.time()
                    total_valid_duration = 0

                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and update:
                    valid_start_time = time.time()

                    valid_score, valid_loss, valid_ppl, valid_sources, \
                        valid_sources_raw, valid_references, valid_hypotheses, \
                        valid_hypotheses_raw, valid_attention_scores = \
                        validate_on_data(
                            batch_size=self.batch_size, data=valid_data,
                            eval_metric=self.eval_metric,
                            level=self.level, model=self.model,
                            use_cuda=self.use_cuda,
                            max_output_length=self.max_output_length,
                            criterion=self.criterion)

                    if self.ckpt_metric == "loss":
                        ckpt_score = valid_loss
                    elif self.ckpt_metric in ["ppl", "perplexity"]:
                        ckpt_score = valid_ppl
                    else:
                        ckpt_score = valid_score

                    new_best = False
                    if self.is_best(ckpt_score):
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            'Hooray! New best validation result [%s]!',
                                self.ckpt_metric)
                        new_best = True
                        self.save_checkpoint()

                    # pass validation score or loss or ppl to scheduler
                    if self.schedule_metric == "loss":
                        # schedule based on loss
                        schedule_score = valid_loss
                    elif self.schedule_metric in ["ppl", "perplexity"]:
                        # schedule based on perplexity
                        schedule_score = valid_ppl
                    else:
                        # schedule based on evaluation score
                        schedule_score = valid_score
                    if self.scheduler is not None:
                        self.scheduler.step(schedule_score)

                    # append to validation report
                    self._add_report(
                        valid_score=valid_score, valid_loss=valid_loss,
                        valid_ppl=valid_ppl, eval_metric=self.eval_metric,
                        new_best=new_best)

                    # always print first x sentences
                    for p in range(self.print_valid_sents):
                        self.logger.debug("Example #%d", p)
                        self.logger.debug("\tRaw source: %s",
                                          valid_sources_raw[p])
                        self.logger.debug("\tSource: %s",
                            valid_sources[p])
                        self.logger.debug("\tReference: %s",
                                          valid_references[p])
                        self.logger.debug("\tRaw hypothesis: %s",
                            valid_hypotheses_raw[p])
                        self.logger.debug("\tHypothesis: %s",
                            valid_hypotheses[p])
                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        'Validation result at epoch %d, step %d: %s: %f, '
                        'loss: %f, ppl: %f, duration: %.4fs',
                            epoch_no+1, self.steps, self.eval_metric,
                            valid_score, valid_loss, valid_ppl, valid_duration)

                    # store validation set outputs
                    self.store_outputs(valid_hypotheses)

                    # store attention plots for first three sentences of
                    # valid data and one randomly chosen example
                    store_attention_plots(attentions=valid_attention_scores,
                                          targets=valid_hypotheses_raw,
                                          sources=[s for s in valid_data.src],
                                          idx=[0, 1, 2,
                                               np.random.randint(0, len(
                                                   valid_hypotheses))],
                                          output_prefix="{}/att.{}".format(
                                              self.model_dir,
                                              self.steps))

                if self.stop:
                    break
            if self.stop:
                self.logger.info(
                    'Training ended since minimum lr %f was reached.',
                        self.learning_rate_min)
                break
        else:
            self.logger.info('Training ended after %d epochs.',
                epoch_no+1)
        self.logger.info('Best validation result at step %d: %f %s.',
            self.best_ckpt_iteration, self.best_ckpt_score, self.ckpt_metric)

    def _train_batch(self, batch, update=True):
        """
        Train the model on one batch: Compute the loss, make a gradient step.

        :param batch:
        :param update: if False, only store gradient. if True also make update
        :return:
        """
        batch_loss = self.model.get_loss_for_batch(
            batch=batch, criterion=self.criterion)

        # normalize batch loss
        if self.normalization == "batch":
            normalizer = batch.nseqs
        elif self.normalization == "tokens":
            normalizer = batch.ntokens
        else:
            raise NotImplementedError("Only normalize by 'batch' or 'tokens'")

        norm_batch_loss = batch_loss.sum() / normalizer
        # division needed since loss.backward sums the gradients until updated
        norm_batch_multiply = norm_batch_loss / self.batch_multiplier

        # compute gradients
        norm_batch_multiply.backward()

        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(params=self.model.parameters())

        if update:
            # make gradient step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # increment step counter
            self.steps += 1

        # increment token counter
        self.total_tokens += batch.ntokens

        return norm_batch_loss

    def _add_report(self, valid_score, valid_ppl, valid_loss, eval_metric,
                    new_best=False):
        """
        Add a one-line report to validation logging file.

        :param valid_score:
        :param valid_ppl:
        :param valid_loss:
        :param eval_metric:
        :param new_best:
        :return:
        """
        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        if current_lr < self.learning_rate_min:
            self.stop = True

        with open(self.valid_report_file, 'a') as opened_file:
            opened_file.write(
                "Steps: {}\tLoss: {:.5f}\tPPL: {:.5f}\t{}: {:.5f}\t"
                "LR: {:.8f}\t{}\n".format(
                    self.steps, valid_loss, valid_ppl, eval_metric,
                    valid_score, current_lr, "*" if new_best else ""))

    def store_outputs(self, hypotheses):
        """
        Write current validation outputs to file in model_dir.
        :param hypotheses:
        :return:
        """
        current_valid_output_file = "{}/{}.hyps".format(self.model_dir,
                                                        self.steps)
        with open(current_valid_output_file, 'w') as opened_file:
            for hyp in hypotheses:
                opened_file.write("{}\n".format(hyp))


def train(cfg_file):
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file:
    :return:
    """
    cfg = load_config(cfg_file)
    # set the random seed
    # torch.backends.cudnn.deterministic = True
    seed = cfg["training"].get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab = \
        load_data(cfg=cfg)

    # build an encoder-decoder model
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, trainer.model_dir+"/config.yaml")

    # print config
    log_cfg(cfg, trainer.logger)

    log_data_info(train_data=train_data, valid_data=dev_data,
                  test_data=test_data, src_vocab=src_vocab, trg_vocab=trg_vocab,
                  logging_function=trainer.logger.info)
    model.log_parameters_list(logging_function=trainer.logger.info)

    logging.info(model)

    # store the vocabs
    src_vocab_file = "{}/src_vocab.txt".format(cfg["training"]["model_dir"])
    src_vocab.to_file(src_vocab_file)
    trg_vocab_file = "{}/trg_vocab.txt".format(cfg["training"]["model_dir"])
    trg_vocab.to_file(trg_vocab_file)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    if test_data is not None:
        checkpoint_path = "{}/{}.ckpt".format(
                trainer.model_dir, trainer.best_ckpt_iteration)
        try:
            trainer.load_checkpoint(checkpoint_path)
        except AssertionError:
            trainer.logger.warning("Checkpoint %s does not exist. "
                                   "Skipping testing.", checkpoint_path)
            if trainer.best_ckpt_iteration == 0 \
                and trainer.best_ckpt_score in [np.inf, -np.inf]:
                trainer.logger.warning(
                    "It seems like no checkpoint was written, "
                    "since no improvement was obtained over the initial model.")
            return

        # test model
        if "testing" in cfg.keys():
            beam_size = cfg["testing"].get("beam_size", 0)
            beam_alpha = cfg["testing"].get("alpha", -1)
        else:
            beam_size = 0
            beam_alpha = -1

        # pylint: disable=unused-variable
        score, loss, ppl, sources, sources_raw, references, hypotheses, \
            hypotheses_raw, attention_scores = validate_on_data(
                data=test_data, batch_size=trainer.batch_size,
                eval_metric=trainer.eval_metric, level=trainer.level,
                max_output_length=trainer.max_output_length,
                model=model, use_cuda=trainer.use_cuda, criterion=None,
                beam_size=beam_size, beam_alpha=beam_alpha)

        if "trg" in test_data.fields:
            decoding_description = "Greedy decoding" if beam_size == 0 else \
                "Beam search decoding with beam size = {} and alpha = {}"\
                    .format(beam_size, beam_alpha)
            trainer.logger.info("Test data result: %f %s [%s]",
                                score, trainer.eval_metric,
                                decoding_description)
        else:
            trainer.logger.info(
                "No references given for %s.%s -> no evaluation.",
                cfg["data"]["test"], cfg["data"]["src"])

        output_path_set = "{}/{}.{}".format(
            trainer.model_dir, "test", cfg["data"]["trg"])
        with open(output_path_set, mode="w", encoding="utf-8") as f:
            for h in hypotheses:
                f.write(h + "\n")
        trainer.logger.info("Test translations saved to: %s", output_path_set)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Joey-NMT')
    parser.add_argument("config", default="configs/default.yaml", type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)
