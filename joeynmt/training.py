# coding: utf-8
import argparse
import logging
import time
import os
import sys
import numpy as np
import shutil


import torch
import torch.nn as nn

from joeynmt.model import build_model

from joeynmt.batch import Batch
from joeynmt.helpers import log_data_info, load_data, \
    load_config, log_cfg, store_attention_plots, make_data_iter
from joeynmt.prediction import validate_on_data


class TrainManager:

    def __init__(self, model, config):
        train_config = config["training"]
        self.model = model
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        # criterion = nn.CrossEntropyLoss(ignore_index=self.pad_index, reduction="sum")
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
        if self.schedule_metric == "eval_metric":
            # if we schedule after BLEU/chrf, we want to maximize it
            scheduler_mode = "max"
        else:
            # if we schedule after loss or perplexity, we want to minimize it
            scheduler_mode = "min"
        self.scheduler = None
        if "scheduling" in train_config.keys() and \
                train_config["scheduling"]:
            if train_config["scheduling"].lower() == "plateau":
                # learning rate scheduler
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer=self.optimizer,
                    mode=scheduler_mode,
                    verbose=True,
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
        self.criterion = criterion
        self.normalization = train_config.get("normalization", "batch")
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_tokens = 0
        self.best_valid_score = 0
        self.best_valid_iteration = 0
        self.max_output_length = train_config.get("max_output_length", None)
        self.overwrite = train_config.get("overwrite", False)
        self.model_dir = self._make_model_dir(train_config["model_dir"])
        self.logger = self._make_logger()
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.use_cuda = train_config["use_cuda"]
        if self.use_cuda:
            self.model.cuda()
        self.logging_freq = train_config["logging_freq"]
        self.validation_freq = train_config["validation_freq"]
        self.eval_metric = train_config["eval_metric"]
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
            self.logger.info("Loading model from {}".format(model_load_path))
            self.load_checkpoint(model_load_path)

    def save_checkpoint(self):
        model_path = "{}/{}.ckpt".format(self.model_dir, self.steps)
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_valid_score": self.best_valid_score,
            "best_valid_iteration": self.best_valid_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if \
            self.scheduler is not None else None,
        }
        torch.save(state, model_path)

    def load_checkpoint(self, path):

        assert os.path.isfile(path), "Checkpoint %s not found" % path
        checkpoint = torch.load(path)

        # restore model and optimizer parameters
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if checkpoint["scheduler_state"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        # restore counts
        self.steps = checkpoint["steps"]
        self.total_tokens = checkpoint["total_tokens"]
        self.best_valid_score = checkpoint["best_valid_score"]
        self.best_valid_iteration = checkpoint["best_valid_iteration"]

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda()

    def _make_model_dir(self, model_dir):
        if os.path.isdir(model_dir):
            if not self.overwrite:
                raise FileExistsError(
                    "Model directory exists and overwriting is disabled.")
        else:
            os.makedirs(model_dir)
        return model_dir

    def _make_logger(self):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            handlers=[
                                logging.FileHandler(
                                    "{}/train.log".format(self.model_dir)),
                                logging.StreamHandler(sys.stdout)
                            ])
        logging.info("Hello! This is Joey-NMT.")
        return logging

    def train_and_validate(self, train_data, valid_data):
        train_iter = make_data_iter(train_data, batch_size=self.batch_size,
                                    train=True, shuffle=self.shuffle)
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH {}".format(epoch_no + 1))
            self.model.train()

            start = time.time()
            total_valid_duration = 0
            processed_tokens = self.total_tokens

            for batch_no, batch in enumerate(iter(train_iter), 1):
                # reactivate training
                self.model.train()
                batch = Batch(batch, self.pad_index, use_cuda=self.use_cuda)
                batch_loss = self._train_batch(batch)

                # log learning progress
                if self.model.training and self.steps % self.logging_freq == 0:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - processed_tokens
                    self.logger.info(
                        "Epoch %d Step: %d Loss: %f Tokens per Sec: %f" %
                        (epoch_no + 1, self.steps, batch_loss,
                         elapsed_tokens / elapsed))
                    start = time.time()
                    total_valid_duration = 0

                # validate on whole dev set
                if self.steps % self.validation_freq == 0:
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

                    if valid_score > self.best_valid_score:
                        self.best_valid_score = valid_score
                        self.best_valid_iteration = self.steps
                        self.logger.info('Hooray! New best validation result!')
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
                        new_best=self.steps == self.best_valid_iteration)

                    # always print first x sentences
                    for p in range(self.print_valid_sents):
                        self.logger.debug("Example #{}".format(p))
                        self.logger.debug("\tRaw source: {}".format(
                            valid_sources_raw[p]))
                        self.logger.debug("\tSource: {}".format(
                            valid_sources[p]))
                        self.logger.debug("\tReference: {}".format(
                            valid_references[p]))
                        self.logger.debug("\tRaw hypothesis: {}".format(
                            valid_hypotheses_raw[p]))
                        self.logger.debug("\tHypothesis: {}".format(
                            valid_hypotheses[p]))
                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        'Validation result at epoch {}, step {}: {}: {}, '
                        'loss: {}, ppl: {}, duration: {:.4f}s'.format(
                            epoch_no+1, self.steps, self.eval_metric,
                            valid_score, valid_loss, valid_ppl, valid_duration))

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
                    'Training ended since minimum lr {} was reached.'.format(
                        self.learning_rate_min))
                break
        else:
            self.logger.info('Training ended after {} epochs.'.format(epoch_no))
        self.logger.info('Best validation result at step {}: {} {}.'.format(
            self.best_valid_iteration, self.best_valid_score, self.eval_metric))

    def _train_batch(self, batch):
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
        # compute gradient
        norm_batch_loss.backward()

        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(params=self.model.parameters())

        # make gradient step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # increment step and token counter
        self.steps += 1
        self.total_tokens += batch.ntokens
        return norm_batch_loss

    def _add_report(self, valid_score, valid_ppl, valid_loss, eval_metric,
                    new_best=False):
        """ Add a one-line report to validation logging file. """
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
        """ Write current validation outputs to file """
        current_valid_output_file = "{}/{}.hyps".format(self.model_dir,
                                                        self.steps)
        with open(current_valid_output_file, 'w') as opened_file:
            for hyp in hypotheses:
                opened_file.write("{}\n".format(hyp))


def train(cfg_file):
    cfg = load_config(cfg_file)
    # set the random seed
    # torch.backends.cudnn.deterministic = True
    seed = cfg["training"].get("random_seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # load the data
    train_data, dev_data, test_data, src_vocab, trg_vocab = \
        load_data(config=cfg)

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
        # test model
        beam_size = cfg["testing"].get("beam_size", 0)
        beam_alpha = cfg["testing"].get("alpha", -1)
        validate_on_data(
            data=test_data, batch_size=trainer.batch_size,
            eval_metric=trainer.eval_metric, level=trainer.level,
            max_output_length=trainer.max_output_length,
            model=model, use_cuda=trainer.use_cuda, criterion=None,
            beam_size=beam_size, beam_alpha=beam_alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Joey-NMT')
    parser.add_argument("config", default="configs/default.yaml", type=str,
                        help="Training configuration file (yaml).")
    args = parser.parse_args()
    train(cfg_file=args.config)
