.. _overview:

========
Overview
========

This page gives an overview of the particular organization of the code.
If you want to modify or contribute to the code, this is a must-read, so you know where to enter your code.

For a detailed documentation of the API, go to :ref:`modules`.


Modes
=====
When JoeyNMT is called from the command line, the mode ("train/test/translate") determines what happens next.

The **"train"** mode leads to ``training.py``, where executes the following steps:

1. load the configuration file
2. load the data and build the vocabularies
3. build the model
4. create a training manager
5. train and validate the model (includes saving checkpoints)
6. test the model with the best checkpoint (if test data given)

**"test"** and **"translate"** mode are handled by ``prediction.py``.
In **"test"** mode, JoeyNMT does the following:

1. load the configuration file
2. load the data and vocabulary files
3. load the model from checkpoint
4. predict hypotheses for the test set
5. evaluate hypotheses against references (if given)

The **"translate"** mode is similar, but it loads source sentences either from an *external* file or prompts lines of *inputs from the user* and does not perform an evaluation.

Training Management
===================

The training process is managed by the `TrainManager <https://github.com/joeynmt/joeynmt/blob/master/joeynmt/training.py#L37>`_.
The manager receives a model and then performs the following steps: parses the input configuration, sets up the logger, schedules the learning rate, sets up the optimizer and counters for update steps. It then keeps track of the current best checkpoint to determine when to stop training.
Most of the hyperparameters in the "training" section of the configuration file are turned into attributes of the TrainManager.


Encoder-Decoder Model
=====================

The encoder-decoder model architecture is defined in `model.py <https://github.com/joeynmt/joeynmt/blob/master/joeynmt/model.py>`_.
This is where encoder and decoder get connected. The forward pass as well as the computation of the training loss and the generation of predictions of the combined encoder-decoder model are defined here.

Individual encoders and decoders are defined with their forward functions in `encoders.py <https://github.com/joeynmt/joeynmt/blob/master/joeynmt/encoders.py>`_ and `decoders.py <https://github.com/joeynmt/joeynmt/blob/master/joeynmt/decoders.py>`_.

Data Handling
=============

Mini-Batching
-------------
The **training** data is split into buckets of similar source and target length and then split into batches (`data.py <https://github.com/joeynmt/joeynmt/blob/master/joeynmt/data.py>`_) to reduce the amount of padding, i.e. waste of computation time.
The samples within each mini-batch are sorted, so that we can make use of PyTorch's efficient RNN `sequence padding and packing <https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e>`_ functions.

For **inference**, we sort the data as well (when creating batches with `batch.py <https://github.com/joeynmt/joeynmt/blob/master/joeynmt/batch.py>`_), but we keep track of the original order so that we can revert the order of the model outputs.
This trick speeds up validation and also testing.

Vocabulary
----------
For the creation of the vocabulary (`vocabulary.py <https://github.com/joeynmt/joeynmt/blob/master/joeynmt/vocabulary.py>`_), all tokens occuring in the training set are collected, sorted and optionally filtered by frequency and then cut off as specified in the configuration.
The vocabularies are stored in the model directory. The vocabulary files contain one token per line, where the line number corresponds to the index of the token in the vocabulary.

Data Loading
------------
At the current state, we use `Torchtext <https://torchtext.readthedocs.io/en/latest/>`_ for data loading and the transformation of files of strings to PyTorch tensors.
Most importantly, the code (`data.py`) works with the `Dataset <https://torchtext.readthedocs.io/en/latest/datasets.html>`_ and `Field <https://torchtext.readthedocs.io/en/latest/data.html#fields>`_ objects: one field for source and one for target, creating a `TranslationDataset <https://torchtext.readthedocs.io/en/latest/datasets.html?highlight=TranslationDataset#machine-translation>`_.


Inference
=========
For inference we run either beam search or greedy decoding, both implemented in `search.py <https://github.com/joeynmt/joeynmt/blob/master/joeynmt/search.py>`_.
We chose to largely adopt the `implementation of beam search in OpenNMT-py <https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam_search.py>`_ for the neat solution of dropping hypotheses from the batch when they are finished.


Checkpoints
===========
The TrainManager takes care of saving checkpoints whenever the model has reached a new validation highscore (keeping a configurable number of checkpoints in total).
The checkpoints do not only contain the model parameters (``model_state``), but also the cumulative count of training tokens and steps, the highscore and iteration count for that highscore, the state of the optimizer, the scheduler and the data iterator.
This ensures a seamless continuation of training when training is interrupted.

From ``_save_checkpoint``:
::

    model_state_dict = self.model.module.state_dict() if \
    isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()
    state = {
        "steps": self.steps,
        "total_tokens": self.total_tokens,
        "best_ckpt_score": self.best_ckpt_score,
        "best_ckpt_iteration": self.best_ckpt_iteration,
        "model_state": model_state_dict,
        "optimizer_state": self.optimizer.state_dict(),
        "scheduler_state": self.scheduler.state_dict() if \
        self.scheduler is not None else None,
        'amp_state': amp.state_dict() if self.fp16 else None
        'train_iter_state': train_iter.state_dict()
    }


