.. _overview:

===============
Module Overview
===============

This page gives an overview of the particular organization of the code.
If you want to modify or contribute to the code, this is a must-read, so you know where to enter your code.

For detailed documentation of the API, go to :ref:`api`.


Modes
-----

When JoeyNMT is called from the command line, the mode ("train/test/translate") determines what happens next.

The **"train"** mode leads to :joeynmt:`training.py`, where executes the following steps:

1. load the configuration file
2. load the data and build the vocabularies
3. build the model
4. create a training manager
5. train and validate the model (includes saving checkpoints)
6. test the model with the best checkpoint (if test data is given)

**"test"** and **"translate"** mode are handled by :joeynmt:`prediction.py`.
In **"test"** mode, JoeyNMT does the following:

1. load the configuration file
2. load the data and vocabulary files
3. load the model from checkpoint
4. predict hypotheses for the test set
5. evaluate hypotheses against references (if given)

The **"translate"** mode is similar, but it loads source sentences either from an *external* file or prompts lines of *inputs from the user* and does not perform an evaluation.


Training Management
-------------------

The training process is managed by the `TrainManager :joeynmt:`training.py`.
The manager receives a model and then performs the following steps: parses the input configuration, sets up the logger, schedules the learning rate, sets up the optimizer and counters for update steps. It then keeps track of the current best checkpoint to determine when to stop training.
Most of the hyperparameters in the "training" section of the configuration file are turned into attributes of the TrainManager.

When "batch_multiplier" > 0 is set, the gradients are accumulated before the model parameters are updated. In the batch loop, we call ``loss.backward()``  for each batch, but ``optimizer.step()`` is called every (batch_multiplier)-th time steps only, and then the accumulated gradients (``model.zero_grad()``) are reset.

.. code-block:: python

    for epoch in range(epochs):
        model.zero_grad()
        epoch_loss = 0.0
        batch_loss = 0.0
        for i, batch in enumerate(train_iter):

            # gradient accumulation:
            # loss.backward() will be called in _train_step()
            batch_loss += _train_step(batch)

            if (i + 1) % args.batch_multiplier == 0:
                optimizer.step()     # update!
                model.zero_grad()    # reset gradients
                steps += 1           # increment counter

                epoch_loss += batch_loss  # accumulate batch loss
                batch_loss = 0            # reset batch loss

        # leftovers are just ignored.
        # (see `drop_last` arg in train_iter.batch_sampler)


Encoder-Decoder Model
---------------------

The encoder-decoder model architecture is defined in :joeynmt:`model.py`.
This is where encoder and decoder get connected. The forward pass as well as the computation of the training loss and the generation of predictions of the combined encoder-decoder model are defined here.

Individual encoders and decoders are defined with their forward functions in :joeynmt:`encoders.py` and :joeynmt:`decoders.py`.


Data Handling
-------------

Data Loading
^^^^^^^^^^^^

At the current state, we support the following input data formats:
- plain txt: one-sentence-per-line; requires language name in the file extension.
- tsv: requires header row with src and trg language names.
- Huggingface's datasets: requires `translation` field.  
- (stdin for interactive translation cli)

In the timing of data loading, we only apply preprocess operations such as lowercasing, punctuation deletion, etc. if needed.
Tokenization is applied on-the-fly when a batch is constructed during training/prediction. See ``get_item()`` in :joeynmt:`BaseDataset <datasets.py>` class for details.


Mini-Batching
^^^^^^^^^^^^^

The dataloader samples data points from the corpus and constructs a batch with :joeynmt:`batch.py`. The instances within each mini-batch are sorted by length, so that we can make use of PyTorch's efficient RNN `sequence padding and packing <https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e>`_ functions. For **inference**, we keep track of the original order so that we can revert the order of the model outputs.

Joey NMT v2.3 (or greater) supports DataParallel and DistributedDataParallel in PyTorch. Please refer to external documentation i.e. `PyTorch DDP tutorial <https://pytorch.org/tutorials/intermediate/dist_tuto.html>`_ to learn how those modules dispatch the minibatchs across multiple GPU devices.


Vocabulary
^^^^^^^^^^

For the creation of the vocabulary (:joeynmt:`vocabulary.py`, all tokens occurring in the training set are collected, sorted and optionally filtered by frequency and then cut off as specified in the configuration. By default, it creates src language vocab and trg language vocab separately. If you want to use joint vocabulary, you need to create vocabulary (:scripts:`build_vocab.py`) before you start training.
The vocabularies are stored in the model directory. The vocabulary files contain one token per line, where the line number corresponds to the index of the token in the vocabulary.

Token granularity should be specified in the "data" section of the configuration file. Currently, JoeyNMT supports word-based, character-based models and sub-word models with byte-pair-encodings (BPE) as learned with `subword-nmt <https://github.com/rsennrich/subword-nmt>`_ or `sentencepiece <https://github.com/google/sentencepiece>`_.


Inference
^^^^^^^^^

For inference we run either beam search or greedy decoding, both implemented in :joeynmt:`search.py`.
We chose to largely adopt the `implementation of beam search in OpenNMT-py <https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/translate/beam_search.py>`_ for the neat solution of dropping hypotheses from the batch when they are finished.


Checkpoints
-----------
The TrainManager takes care of saving checkpoints whenever the model has reached a new validation highscore (keeping a configurable number of checkpoints in total).
The checkpoints do not only contain the model parameters (``model_state``), but also the cumulative count of training tokens and steps, the highscore and iteration count for that highscore, the state of the optimizer, the scheduler and the data iterator.
This ensures a seamless continuation of training when training is interrupted.
