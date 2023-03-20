# coding: utf-8
"""
Training module
"""
import heapq
import logging
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from joeynmt.batch import Batch
from joeynmt.builders import build_gradient_clipper, build_optimizer, build_scheduler
from joeynmt.config import (
    TestConfig,
    TrainConfig,
    log_config,
    parse_global_args,
    set_validation_args,
)
from joeynmt.helpers import (
    delete_ckpt,
    load_checkpoint,
    store_attention_plots,
    symlink_update,
    write_list_to_file,
)
from joeynmt.model import Model
from joeynmt.prediction import predict, prepare, test

logger = logging.getLogger(__name__)


class TrainManager:
    """
    Manages training loop, validations, learning rate scheduling
    and early stopping.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self,
                 model: Model,
                 model_dir: Path,
                 device: torch.device,
                 n_gpu: int = 0,
                 num_workers: int = 0,
                 fp16: bool = False,
                 seed: int = 42,
                 train_args: TrainConfig = None,
                 dev_args: TestConfig = None) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param model_dir: directory to save ckpts
        :param device: torch device
        :param n_gpu: number of gpus. 0 if cpu.
        :param num_workers: number of multiprocess workers.
        :param fp16: half-precision
        :param seed: random seed
        :param train_args: config args for training
        :param dev_args: config args for validation
        """
        self.args = train_args  # config for training
        self.dev_cfg = dev_args  # config for geedy decoding
        self.seed = seed

        # logging and storing
        self.model_dir = model_dir
        self.tb_writer = SummaryWriter(log_dir=(model_dir / "tensorboard").as_posix())

        # model
        self.model = model

        # CPU / GPU
        self.device = device
        self.n_gpu = n_gpu
        self.num_workers = num_workers

        # optimization
        self.clip_grad_fun = build_gradient_clipper(cfg=self.args._asdict())
        self.optimizer = build_optimizer(cfg=self.args._asdict(),
                                         parameters=self.model.parameters())

        # fp16
        self.fp16: bool = fp16  # True or False for scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16) \
            if self.device.type == "cuda" else None
        self.autocast = {"device_type": self.device.type, "enabled": self.fp16}
        if self.fp16:
            self.autocast["dtype"] = torch.float16 \
                if self.device.type == "cuda" else torch.bfloat16

        # save/delete checkpoints
        self.ckpt_queue: List[Tuple[float, Path]] = []  # heap queue

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            cfg=self.args._asdict(),
            scheduler_mode="min" if self.args.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=self.model.encoder._output_size,
        )

        # Placeholder so that we can use the train_iter in other functions.
        self.train_iter, self.train_iter_state = None, None

        # initialize training statistics
        self.stats = self.TrainStatistics(minimize_metric=self.args.minimize_metric)

        # load model parameters
        if self.args.load_model is not None:
            self.init_from_checkpoint(
                self.args.load_model,
                reset_best_ckpt=self.args.reset_best_ckpt,
                reset_scheduler=self.args.reset_scheduler,
                reset_optimizer=self.args.reset_optimizer,
                reset_iter_state=self.args.reset_iter_state,
            )
        for layer_name, load_path in [
            ("encoder", self.args.load_encoder),
            ("decoder", self.args.load_decoder),
        ]:
            if load_path is not None:
                self.init_layers(path=load_path, layer=layer_name)

    def _save_checkpoint(self, new_best: bool, score: float) -> None:
        """
        Save the model's current parameters and the training state to a checkpoint.

        The training state contains the total number of training steps, the total number
        of training tokens, the best checkpoint score and iteration so far, and
        optimizer and scheduler states.

        :param new_best: This boolean signals which symlink we will use for the new
            checkpoint. If it is true, we update best.ckpt.
        :param score: Validation score which is used as key of heap queue. if score is
            float('nan'), the queue won't be updated.
        """
        model_path = Path(self.model_dir) / f"{self.stats.steps}.ckpt"
        train_iter_state = self.train_iter.batch_sampler.sampler.generator.get_state() \
            if hasattr(self.train_iter.batch_sampler.sampler, 'generator') else None
        # yapf: disable
        state = {
            "steps": self.stats.steps,
            "total_tokens": self.stats.total_tokens,
            "best_ckpt_score": self.stats.best_ckpt_score,
            "best_ckpt_iteration": self.stats.best_ckpt_iter,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": (self.scaler.state_dict()
                             if self.scaler is not None else None),
            "scheduler_state": (self.scheduler.state_dict()
                                if self.scheduler is not None else None),
            "train_iter_state": train_iter_state,
            "total_correct": self.stats.total_correct,
        }
        torch.save(state, model_path.as_posix())

        # update symlink
        symlink_target = Path(f"{self.stats.steps}.ckpt")
        # last symlink
        last_path = Path(self.model_dir) / "latest.ckpt"
        prev_path = symlink_update(symlink_target, last_path)  # update always
        # best symlink
        best_path = Path(self.model_dir) / "best.ckpt"
        if new_best:
            prev_path = symlink_update(symlink_target, best_path)
            assert best_path.resolve().stem == str(self.stats.best_ckpt_iter)

        # push to and pop from the heap queue
        to_delete = None
        if not math.isnan(score) and self.args.keep_best_ckpts > 0:
            if len(self.ckpt_queue) < self.args.keep_best_ckpts:  # no pop, push only
                heapq.heappush(self.ckpt_queue, (score, model_path))
            else:  # push + pop the worst one in the queue
                if self.args.minimize_metric:
                    # pylint: disable=protected-access
                    heapq._heapify_max(self.ckpt_queue)
                    to_delete = heapq._heappop_max(self.ckpt_queue)
                    heapq.heappush(self.ckpt_queue, (score, model_path))
                    # pylint: enable=protected-access
                else:
                    to_delete = heapq.heappushpop(self.ckpt_queue, (score, model_path))

            if to_delete is not None:
                assert to_delete[1] != model_path  # don't delete the last ckpt
                if to_delete[1].stem != best_path.resolve().stem:
                    delete_ckpt(to_delete[1])  # don't delete the best ckpt

            assert len(self.ckpt_queue) <= self.args.keep_best_ckpts

            # remove old symlink target if not in queue after push/pop
            if prev_path is not None and prev_path.stem not in [
                    c[1].stem for c in self.ckpt_queue
            ]:
                delete_ckpt(prev_path)

    def init_from_checkpoint(
        self,
        path: Path,
        reset_best_ckpt: bool = False,
        reset_scheduler: bool = False,
        reset_optimizer: bool = False,
        reset_iter_state: bool = False,
    ) -> None:
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also
        scheduler and optimizer states, see `self._save_checkpoint`.

        :param path: path to checkpoint
        :param reset_best_ckpt: reset tracking of the best checkpoint,
                                use for domain adaptation with a new dev
                                set or when using a new metric for fine-tuning.
        :param reset_scheduler: reset the learning rate scheduler, and do not
                                use the one stored in the checkpoint.
        :param reset_optimizer: reset the optimizer, and do not use the one
                                stored in the checkpoint.
        :param reset_iter_state: reset the sampler's internal state and do not
                                use the one stored in the checkpoint.
        """
        logger.info("Loading model from %s", path)
        model_checkpoint = load_checkpoint(path=path, device=self.device)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
            if "scaler_state" in model_checkpoint and self.scaler is not None:
                self.scaler.load_state_dict(model_checkpoint["scaler_state"])
        else:
            logger.info("Reset optimizer.")

        if not reset_scheduler:
            if (model_checkpoint["scheduler_state"] is not None
                    and self.scheduler is not None):
                self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])
        else:
            logger.info("Reset scheduler.")

        if not reset_best_ckpt:
            self.stats.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.stats.best_ckpt_iter = model_checkpoint["best_ckpt_iteration"]
        else:
            logger.info("Reset tracking of the best checkpoint.")

        if not reset_iter_state:
            # restore counters
            assert "train_iter_state" in model_checkpoint
            self.stats.steps = model_checkpoint["steps"]
            self.stats.total_tokens = model_checkpoint["total_tokens"]
            self.stats.total_correct = model_checkpoint.get("total_correct", 0)
            self.train_iter_state = model_checkpoint["train_iter_state"]
        else:
            # reset counters if explicitly 'train_iter_state: True' in config
            logger.info("Reset data iterator (random seed: {%d}).", self.seed)

    def init_layers(self, path: Path, layer: str) -> None:
        """
        Initialize encoder decoder layers from a given checkpoint file.

        :param path: path to checkpoint
        :param layer: layer name; 'encoder' or 'decoder' expected
        """
        assert path is not None
        layer_state_dict = OrderedDict()
        logger.info("Loading %s laysers from %s", layer, path)
        ckpt = load_checkpoint(path=path, device=self.device)
        for k, v in ckpt["model_state"].items():
            if k.startswith(layer):
                layer_state_dict[k] = v
        self.model.load_state_dict(layer_state_dict, strict=False)

    def train_and_validate(self, train_data: Dataset, valid_data: Dataset) -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        # pylint: disable=too-many-branches,too-many-statements
        self.train_iter = train_data.make_iter(
            batch_size=self.args.batch_size,
            batch_type=self.args.batch_type,
            seed=self.seed,
            shuffle=self.args.shuffle,
            num_workers=self.num_workers,
            device=self.device,
            pad_index=self.model.pad_index,
        )

        if self.train_iter_state is not None:
            self.train_iter.batch_sampler.sampler.generator.set_state(
                self.train_iter_state.cpu())

        #################################################################
        # simplify accumulation logic:
        #################################################################
        # for epoch in range(epochs):
        #     self.model.zero_grad()
        #     epoch_loss = 0.0
        #     batch_loss = 0.0
        #     for i, batch in enumerate(self.train_iter):
        #
        #         # gradient accumulation:
        #         # loss.backward() inside _train_step()
        #         batch_loss += self._train_step(inputs)
        #
        #         if (i + 1) % self.batch_multiplier == 0:
        #             self.optimizer.step()     # update!
        #             self.model.zero_grad()    # reset gradients
        #             self.steps += 1           # increment counter
        #
        #             epoch_loss += batch_loss  # accumulate batch loss
        #             batch_loss = 0            # reset batch loss
        #
        #     # leftovers are just ignored.
        #################################################################

        logger.info(
            "Train stats:\n"
            "\tdevice: %s\n"
            "\tn_gpu: %d\n"
            "\t16-bits training: %r\n"
            "\tgradient accumulation: %d\n"
            "\tbatch size per device: %d\n"
            "\teffective batch size (w. parallel & accumulation): %d",
            self.device.type,
            self.n_gpu,
            self.fp16,
            self.args.batch_multiplier,
            self.args.batch_size // self.n_gpu \
                if self.n_gpu > 1 else self.args.batch_size,  # noqa:E131
            self.args.batch_size * self.args.batch_multiplier,
        )

        try:
            for epoch_no in range(self.args.epochs):
                logger.info("EPOCH %d", epoch_no + 1)

                if self.scheduler_step_at == "epoch":
                    self.scheduler.step(epoch=epoch_no)

                self.model.train()

                # Reset statistics for each epoch.
                start = time.time()
                total_valid_duration = 0
                start_tokens = self.stats.total_tokens
                start_correct = self.stats.total_correct
                self.model.zero_grad()
                epoch_loss = 0
                total_batch_loss = 0

                # subsample train data each epoch
                if train_data.random_subset > 0:
                    try:
                        train_data.reset_random_subset()
                        train_data.sample_random_subset(seed=epoch_no)
                        logger.info(
                            "Sample random subset from dev set: n=%d, seed=%d",
                            len(train_data),
                            epoch_no,
                        )
                    except AssertionError as e:
                        logger.warning(e)

                batch: Batch  # yield a joeynmt Batch object
                for i, batch in enumerate(self.train_iter):
                    # sort batch now by src length and keep track of order
                    batch.sort_by_src_length()

                    # get batch loss
                    norm_batch_loss = self._train_step(batch)
                    total_batch_loss += norm_batch_loss

                    # update!
                    if (i + 1) % self.args.batch_multiplier == 0:
                        # clip gradients (in-place)
                        if self.clip_grad_fun is not None:
                            self.clip_grad_fun(parameters=self.model.parameters())

                        # make gradient step
                        if self.scaler is None:
                            self.optimizer.step()
                        else:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()

                        # decay lr
                        if self.scheduler_step_at == "step":
                            self.scheduler.step(self.stats.steps)

                        # reset gradients
                        self.model.zero_grad()

                        # increment step counter
                        self.stats.steps += 1
                        if self.stats.steps >= self.args.max_updates:
                            self.stats.is_max_update = True

                        # log learning progress
                        if self.stats.steps % self.args.logging_freq == 0:
                            elapsed = time.time() - start - total_valid_duration
                            elapsed_tok = self.stats.total_tokens - start_tokens
                            elapsed_correct = self.stats.total_correct - start_correct
                            self.tb_writer.add_scalar("train/batch_loss",
                                                      total_batch_loss,
                                                      self.stats.steps)
                            self.tb_writer.add_scalar("train/batch_acc",
                                                      elapsed_correct / elapsed_tok,
                                                      self.stats.steps)
                            logger.info(
                                "Epoch %3d, "
                                "Step: %8d, "
                                "Batch Loss: %12.6f, "
                                "Batch Acc: %.6f, "
                                "Tokens per Sec: %8.0f, "
                                "Lr: %.6f",
                                epoch_no + 1,
                                self.stats.steps,
                                total_batch_loss,
                                elapsed_correct / elapsed_tok,
                                elapsed_tok / elapsed,
                                self.optimizer.param_groups[0]["lr"],
                            )
                            start = time.time()
                            total_valid_duration = 0
                            start_tokens = self.stats.total_tokens
                            start_correct = self.stats.total_correct

                        # update epoch_loss
                        epoch_loss += total_batch_loss  # accumulate loss
                        total_batch_loss = 0  # reset batch loss

                        # validate on the entire dev set
                        if self.stats.steps % self.args.validation_freq == 0:
                            valid_duration = self._validate(valid_data)
                            total_valid_duration += valid_duration

                        # check current_lr
                        current_lr = self.optimizer.param_groups[0]["lr"]
                        if current_lr < self.args.learning_rate_min:
                            self.stats.is_min_lr = True

                        self.tb_writer.add_scalar("train/learning_rate", current_lr,
                                                  self.stats.steps)

                    if self.stats.is_min_lr or self.stats.is_max_update:
                        break

                if self.stats.is_min_lr or self.stats.is_max_update:
                    log_str = (f"minimum lr {self.args.learning_rate_min}"
                               if self.stats.is_min_lr else
                               f"maximum num. of updates {self.args.max_updates}")
                    logger.info("Training ended since %s was reached.", log_str)
                    break

                logger.info(
                    "Epoch %3d: total training loss %.2f",
                    epoch_no + 1,
                    epoch_loss,
                )
            else:
                logger.info("Training ended after %3d epochs.", epoch_no + 1)
            logger.info(
                "Best validation result (greedy) "
                "at step %8d: %6.2f %s.",
                self.stats.best_ckpt_iter,
                self.stats.best_ckpt_score,
                self.args.early_stopping_metric,
            )
        except KeyboardInterrupt:
            self._save_checkpoint(False, float("nan"))

        self.tb_writer.close()  # close Tensorboard writer

    def _train_step(self, batch: Batch) -> Tensor:
        """
        Train the model on one batch: Compute the loss.

        :param batch: training batch
        :return:
            - losses for batch (sum)
            - number of correct tokens for batch (sum)
        """
        # reactivate training
        self.model.train()

        with torch.autocast(**self.autocast):
            # get loss (run as during training with teacher forcing)
            batch_loss, _, _, correct_tokens = self.model(return_type="loss",
                                                          **vars(batch))

        # normalize batch loss
        norm_batch_loss = batch.normalize(
            batch_loss,
            normalization=self.args.normalization,
            n_gpu=self.n_gpu,
            n_accumulation=self.args.batch_multiplier,
        )

        # sum over multiple gpus
        sum_correct_tokens = batch.normalize(correct_tokens, "sum", self.n_gpu)

        # accumulate gradients
        if self.scaler is None:
            norm_batch_loss.backward()
        else:
            self.scaler.scale(norm_batch_loss).backward()

        # increment token counter
        self.stats.total_tokens += batch.ntokens
        self.stats.total_correct += sum_correct_tokens.item()

        return norm_batch_loss.item()

    def _validate(self, valid_data: Dataset):
        if valid_data.random_subset > 0:  # subsample validation set each valid step
            try:
                valid_data.reset_random_subset()
                valid_data.sample_random_subset(seed=self.stats.steps)
                logger.info(
                    "Sample random subset from dev set: n=%d, seed=%d",
                    len(valid_data),
                    self.stats.steps,
                )
            except AssertionError as e:
                logger.warning(e)

        valid_start_time = time.time()
        (
            valid_scores,
            valid_references,
            valid_hypotheses,
            valid_hypotheses_raw,
            valid_sequence_scores,  # pylint: disable=unused-variable
            valid_attention_scores,
        ) = predict(
            model=self.model,
            data=valid_data,
            compute_loss=True,
            device=self.device,
            n_gpu=self.n_gpu,
            normalization=self.args.normalization,
            args=self.dev_cfg,
            fp16=self.fp16,
        )
        valid_duration = time.time() - valid_start_time

        # for eval_metric in ['loss', 'ppl', 'acc'] + self.eval_metrics:
        for eval_metric, score in valid_scores.items():
            if not math.isnan(score):
                self.tb_writer.add_scalar(f"valid/{eval_metric}", score,
                                          self.stats.steps)

        ckpt_score = valid_scores[self.args.early_stopping_metric]

        if self.scheduler_step_at == "validation":
            self.scheduler.step(metrics=ckpt_score)

        # update new best
        new_best = self.stats.is_best(ckpt_score)
        if new_best:
            self.stats.best_ckpt_score = ckpt_score
            self.stats.best_ckpt_iter = self.stats.steps
            logger.info(
                "Hooray! New best validation result [%s]!",
                self.args.early_stopping_metric,
            )

        # save checkpoints
        is_better = (self.stats.is_better(ckpt_score, self.ckpt_queue)
                     if len(self.ckpt_queue) > 0 else True)
        if self.args.keep_best_ckpts < 0 or is_better:
            self._save_checkpoint(new_best, ckpt_score)

        # append to validation report
        self._add_report(valid_scores=valid_scores, new_best=new_best)

        self._log_examples(
            references=valid_references,
            hypotheses=valid_hypotheses,
            hypotheses_raw=valid_hypotheses_raw,
            data=valid_data,
        )

        # store validation set outputs
        write_list_to_file(self.model_dir / f"{self.stats.steps}.hyps",
                           valid_hypotheses)

        # store attention plots for selected valid sentences
        if valid_attention_scores:
            store_attention_plots(
                attentions=valid_attention_scores,
                targets=valid_hypotheses_raw,
                sources=valid_data.get_list(lang=valid_data.src_lang, tokenized=True),
                indices=self.args.print_valid_sents,
                output_prefix=(self.model_dir / f"att.{self.stats.steps}").as_posix(),
                tb_writer=self.tb_writer,
                steps=self.stats.steps,
            )

        return valid_duration

    def _add_report(self, valid_scores: dict, new_best: bool = False) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_scores: validation evaluation score [eval_metric]
        :param new_best: whether this is a new best model
        """
        current_lr = self.optimizer.param_groups[0]["lr"]

        valid_file = self.model_dir / "validations.txt"
        with valid_file.open("a", encoding="utf-8") as opened_file:
            score_str = "\t".join([f"Steps: {self.stats.steps}"] + [
                f"{eval_metric}: {score:.5f}"
                for eval_metric, score in valid_scores.items() if not math.isnan(score)
            ] + [f"LR: {current_lr:.8f}", "*" if new_best else ""])
            opened_file.write(f"{score_str}\n")

    def _log_examples(
        self,
        hypotheses: List[str],
        references: List[str],
        hypotheses_raw: List[List[str]],
        data: Dataset,
    ) -> None:
        """
        Log the `print_valid_sents` sentences from given examples.

        :param hypotheses: decoded hypotheses (list of strings)
        :param references: decoded references (list of strings)
        :param hypotheses_raw: raw hypotheses (list of list of tokens)
        :param data: Dataset
        """
        for p in self.args.print_valid_sents:
            if p >= len(hypotheses):
                continue
            logger.info("Example #%d", p)

            # tokenized text
            tokenized_src = data.get_item(idx=p, lang=data.src_lang)
            tokenized_trg = data.get_item(idx=p, lang=data.trg_lang)
            logger.debug("\tTokenized source:     %s", tokenized_src)
            logger.debug("\tTokenized reference:  %s", tokenized_trg)
            logger.debug("\tTokenized hypothesis: %s", hypotheses_raw[p])

            # detokenized text
            detokenized_src = data.tokenizer[data.src_lang].post_process(data.src[p])
            logger.info("\tSource:     %s", detokenized_src)
            logger.info("\tReference:  %s", references[p])
            logger.info("\tHypothesis: %s", hypotheses[p])

    class TrainStatistics:

        def __init__(
            self,
            steps: int = 0,
            is_min_lr: bool = False,
            is_max_update: bool = False,
            total_tokens: int = 0,
            best_ckpt_iter: int = 0,
            minimize_metric: bool = True,
            total_correct: int = 0,
        ) -> None:
            """
            Train Statistics

            :param steps: global update step counter
            :param is_min_lr: stop by reaching learning rate minimum
            :param is_max_update: stop by reaching max num of updates
            :param total_tokens: number of total tokens seen so far
            :param best_ckpt_iter: store iteration point of best ckpt
            :param minimize_metric: minimize or maximize score
            :param total_correct: number of correct tokens seen so far
            """
            self.steps = steps
            self.is_min_lr = is_min_lr
            self.is_max_update = is_max_update
            self.total_tokens = total_tokens
            self.best_ckpt_iter = best_ckpt_iter
            self.minimize_metric = minimize_metric
            self.best_ckpt_score = float('inf') if minimize_metric else float('-inf')
            self.total_correct = total_correct

        def is_best(self, score):
            if self.minimize_metric:
                is_best = score < self.best_ckpt_score
            else:
                is_best = score > self.best_ckpt_score
            return is_best

        def is_better(self, score: float, heap_queue: list):
            assert len(heap_queue) > 0
            if self.minimize_metric:
                is_better = score < heapq.nlargest(1, heap_queue)[0][0]
            else:
                is_better = score > heapq.nsmallest(1, heap_queue)[0][0]
            return is_better


def train(cfg: Dict, skip_test: bool = False) -> None:
    """
    Main training function. After training, also test on test data if given.

    :param cfg: configuration dict
    :param skip_test: whether a test should be run or not after training
    """

    # write all entries of config to the log
    log_config(cfg)

    # parse args
    args = parse_global_args(cfg, mode="train")

    # prepare model and datasets
    model, train_data, dev_data, test_data = prepare(args, mode="train")
    dev_args = set_validation_args(args.test,
                                   args.train.batch_size,
                                   args.train.batch_type)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model,
                           model_dir=args.model_dir,
                           device=args.device,
                           n_gpu=args.n_gpu,
                           num_workers=args.num_workers,
                           fp16=args.fp16,
                           seed=args.seed,
                           train_args=args.train,
                           dev_args=dev_args)

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    if not skip_test:
        # predict with the best model on validation and test
        # (if test data is available)

        # load model checkpoint
        ckpt = args.model_dir / "best.ckpt"
        model_checkpoint = load_checkpoint(ckpt, device=args.device)
        model.load_state_dict(model_checkpoint["model_state"])

        prepared = {"dev": dev_data, "test": test_data, "model": model}
        test(
            cfg,
            output_path=(args.model_dir / "best.hyps").as_posix(),
            prepared=prepared,
        )
    else:
        logger.info("Skipping test after training.")
