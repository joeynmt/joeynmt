.. _tutorial:

========
Tutorial
========

In this tutorial, you learn to build a recurrent neural translation system for a toy translation task, how to train, tune and test it.

Instead of following the synthetic example here, you might also run the :notebooks:`quick start guide <joey_v2_demo.ipynb>` that walks you step-by-step through the installation, data preparation, training, evaluation using "real" translation dataset from `Tatoeba <https://opus.nlpl.eu/Tatoeba.php>`_.

:notebooks:`Torchhub tutorial <torchhub.ipynb>` demonstrates how to generate translation from a pretrained model via `Torchhub <https://pytorch.org/hub/>`_ API. 


1. Data Preparation
-------------------
For training a translation model, you need parallel data, i.e. a collection of source sentences and reference translations that are aligned sentence-by-sentence and stored in two files,
such that each line in the reference file is the translation of the same line in the source file.


Synthetic Data
^^^^^^^^^^^^^^

For the sake of this tutorial, we'll simply generate synthetic data to mimic a real-world translation task.
Our machine translation task is here to learn to reverse a given input sequence of integers.

For example, the input would be a source sentence like this:

::

    14 46 43 2 36 6 20 8 38 17 3 24 13 49 8 25

And the correct "translation" would be:

::

    25 8 49 13 24 3 17 38 8 20 6 36 2 43 46 14

Why is this an interesting toy task?

Let's generate some data!

.. code-block:: bash

    python scripts/generate_reverse_task.py

This generates 50k training and 1k dev and test examples for integers between 0 and 50 of maximum length 25 for training and 30 for development and testing.
The generated files are placed under `test/data/reverse/`.

.. code-block:: bash

    wc -l test/data/reverse/*

::

       1000 test/data/reverse/dev.src
       1000 test/data/reverse/dev.trg
       1000 test/data/reverse/test.src
       1000 test/data/reverse/test.trg
      50000 test/data/reverse/train.src
      50000 test/data/reverse/train.trg


2. Configuration
----------------

Once you have the data, it's time to build the NMT model.

In Joey NMT, experiments are specified in configuration files, in `YAML <http://yaml.org/>`_ format.
Most importantly, the configuration contains the description of the model architecture (e.g. number of hidden units in the encoder RNN),
paths to the training, development and test data, and the training hyperparameters (learning rate, validation frequency etc.).

You can find examples in the ``configs`` directory. :configs:`rnn_small.yaml` contains a detailed explanation of all configuration options.

For the tutorial, we'll use :configs:`rnn_reverse.yaml`. We'll go through it section by section.


Top Section
^^^^^^^^^^^

Here we specify general settings applied both in training and prediction.
With ``use_cuda`` we can decide whether to train the model on GPU (True) or CPU (False). Note that for training on GPU you need the appropriate CUDA libraries installed.

.. code-block:: yaml

    name: "reverse_experiment"
    joeynmt_version: "2.3.0"
    model_dir: "reverse_model"
    use_cuda: False
    fp16: False
    random_seed: 42


Data Section
^^^^^^^^^^^^

Here we give the path to the data (".src" is the source suffix, ".trg" is the target suffix of the plain txt files)
and for each side separately, indicate which segmentation level we want to train on, here simply on the word level, as opposed to the character level.
The training set will be filtered by ``max_length``, i.e. only examples where source and target contain not more than 25 tokens are retained for training (that's the full data set for us).
Source and target vocabulary are created from the training data, by keeping ``voc_limit`` source tokens that occur at least ``voc_min_freq`` times, and equivalently for the target side.
If you want to use a pre-generated vocabulary, you can load it in ``voc_file`` field. This will be important when loading a trained model for testing.
``special_symbols`` section defines special tokens required to control training and generation.

.. code-block:: yaml

    data:
        train: "test/data/reverse/train"
        dev: "test/data/reverse/dev"
        test: "test/data/reverse/test"
        dataset_type: "plain"
        src:
            lang: "src"
            max_length: 25
            level: "word"
            voc_limit: 100
            voc_min_freq: 0
            #voc_file: src_vocab.txt
        trg:
            lang: "trg"
            max_length: 25
            level: "word"
            voc_limit: 100
            voc_min_freq: 0
            #voc_file: trg_vocab.txt
        special_symbols:
            unk_token: "<unk>"
            unk_id: 0
            pad_token: "<pad>"
            pad_id: 1
            bos_token: "<s>"
            bos_id: 2
            eos_token: "</s>"
            eos_id: 3


Training Section
^^^^^^^^^^^^^^^^

This section describes how the model is trained.
Training stops when either the learning rate decreased to ``learning_rate_min`` (when using a decreasing learning rate schedule) or the maximum number of epochs is reached.
For individual schedulers and optimizers, we refer to the `PyTorch documentation <https://pytorch.org/docs/stable/index.html>`_.

Here we're using the "plateau" scheduler that reduces the initial learning rate by ``decrease_factor`` whenever the ``early_stopping_metric`` has not improved for ``patience`` validations.
Validations (with greedy decoding) are performed every ``validation_freq`` batches and every ``logging_freq`` batches the training batch loss will be logged.

Checkpoints for the model parameters are saved whenever a new high score in ``early_stopping_metric``, here the ``eval_metric`` BLEU, has been reached.
In order not to waste much memory on old checkpoints, we're only keeping the ``keep_best_ckpts`` best checkpoints. Nevertheless, we always keep the latest checkpoint so that one can resume the training from that point. By setting ``keep_best_ckpts = -1``, you can prevent to delete any checkpoints.

At the beginning of each epoch, the training data is shuffled if we set ``shuffle`` to True (there is actually no good reason for not doing so).


.. code-block:: yaml

    training:
        #load_model: "reverse_model/best.ckpt"
        optimizer: "adamw"
        learning_rate: 0.001
        learning_rate_min: 0.0002
        weight_decay: 0.0
        clip_grad_norm: 1.0
        batch_size: 12
        batch_type: "sentence"
        batch_multiplier: 2
        scheduling: "plateau"
        patience: 5
        decrease_factor: 0.5
        early_stopping_metric: "bleu"
        epochs: 5
        validation_freq: 1000
        logging_freq: 100
        shuffle: True
        print_valid_sents: [0, 3, 6]
        keep_best_ckpts: 2
        overwrite: True

.. danger::

    In this example, we set ``overwrite: True`` which you shouldn't do if you're running serious experiments, since it overwrites the existing ``model_dir`` and all its content if it already exists and you re-start training.


Testing Section
^^^^^^^^^^^^^^^

Here we only specify which decoding strategy we want to use during testing. If ``beam_size: 1`` the model greedily decodes, otherwise it uses a beam of ``beam_size`` to search for the best output. ``beam_alpha`` is the length penalty for beam search (proposed in `Wu et al. 2018 <https://arxiv.org/pdf/1609.08144.pdf>`_).

.. code-block:: yaml

    testing:
        #load_model: "reverse_model/best.ckpt"
        n_best: 1
        beam_size: 1
        beam_alpha: 1.0
        eval_metrics: ["bleu"]
        min_output_length: 1
        max_output_length: 30
        batch_size: 12
        batch_type: "sentence"
        return_prob: "none"
        generate_unk: False
        sacrebleu_cfg:
            tokenize: "13a"
            lowercase: False


Model Section
^^^^^^^^^^^^^

Here we describe the model architecture and the initialization of parameters.

In this example we use a one-layer bidirectional LSTM encoder with 64 units, a one-layer LSTM decoder with also 64 units.
Source and target embeddings both have the size of 16.

We're not going into details for the initialization, just know that it matters for tuning but that our default configurations should generally work fine.
A detailed description for the initialization options is described in :joeynmt:`initialization.py`.

Dropout is applied onto the input of the encoder RNN with dropout probability of 0.1, as well as to the input of the decoder RNN and to the input of the attention vector layer (``hidden_dropout``).
Input feeding (`Luong et al. 2015 <https://aclweb.org/anthology/D15-1166>`_) means the attention vector is concatenated to the hidden state before feeding it to the RNN in the next step.

The first decoder state is simply initialized with zeros. For real translation tasks, the options are `last` (taking the last encoder state) or `bridge` (learning a projection of the last encoder state).

Encoder and decoder are connected through global attention, here through `luong` attention, aka the "general" (Luong et al. 2015) or bilinear attention mechanism.

.. code-block:: yaml

    model:
        initializer: "xavier_uniform"
        embed_initializer: "normal"
        embed_init_weight: 0.1
        bias_initializer: "zeros"
        init_rnn_orthogonal: False
        lstm_forget_gate: 0.
        encoder:
            type: "recurrent"
            rnn_type: "lstm"
            embeddings:
                embedding_dim: 16
                scale: False
            hidden_size: 64
            bidirectional: True
            dropout: 0.1
            num_layers: 1
            activation: "tanh"
        decoder:
            type: "recurrent"
            rnn_type: "lstm"
            embeddings:
                embedding_dim: 16
                scale: False
            hidden_size: 64
            dropout: 0.1
            hidden_dropout: 0.1
            num_layers: 1
            activation: "tanh"
            input_feeding: True
            init_hidden: "zero"
            attention: "luong"


That's it! We've specified all that we need to train a translation model for the reverse task.


3. Training
-----------

Start
^^^^^
For training, run the following command:

.. code-block:: bash

    python -m joeynmt train configs/reverse.yaml


This will train a model on the reverse data specified in the config, validate on validation data,
and store model parameters, vocabularies, validation outputs and a small number of attention plots in the ``reverse_model`` directory.


.. note::

    If you encounter a file IO error, please consider to use the absolute path in the configuration.


Progress Tracking
^^^^^^^^^^^^^^^^^

The Log File
""""""""""""

During training the Joey NMT will print the training log to stdout, and also save it to a log file ``reverse_model/train.log``.
It reports information about the model, like the total number of parameters, the vocabulary size, the data sizes.
You can doublecheck that what you specified in the configuration above is actually matching the model that is now training.

After the reports on the model should see something like this:

::

    2024-01-15 12:57:12,987 - INFO - joeynmt.training - Epoch   1, Step:      900, Batch Loss:    21.149554, Batch Acc: 0.390395, Tokens per Sec:     9462, Lr: 0.001000
    2024-01-15 12:57:16,549 - INFO - joeynmt.training - Epoch   1, Step:     1000, Batch Loss:    35.254892, Batch Acc: 0.414826, Tokens per Sec:     9317, Lr: 0.001000
    2024-01-15 12:57:16,550 - INFO - joeynmt.prediction - Predicting 1000 example(s)... (Greedy decoding with min_output_length=1, max_output_length=30, return_prob='none', generate_unk=True, repetition_penalty=-1, no_repeat_ngram_size=-1)
    2024-01-15 12:57:29,506 - INFO - joeynmt.prediction - Generation took 12.9554[sec].
    2024-01-15 12:57:29,548 - INFO - joeynmt.metrics - nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
    2024-01-15 12:57:29,549 - INFO - joeynmt.prediction - Evaluation result (greedy): bleu:  22.52, loss:  29.77, ppl:   5.88, acc:   0.50, 0.0398[sec]
    2024-01-15 12:57:29,549 - INFO - joeynmt.training - Hooray! New best validation result [bleu]!
    2024-01-15 12:57:29,576 - INFO - joeynmt.training - Checkpoint saved in reverse_model/1000.ckpt.
    2024-01-15 12:57:29,578 - INFO - joeynmt.training - Example #0
    2024-01-15 12:57:29,578 - INFO - joeynmt.training -     Source:     10 43 37 32 6 9 25 36 21 29 16 7 18 27 30 46 37 15 7 48 18
    2024-01-15 12:57:29,578 - INFO - joeynmt.training -     Reference:  18 48 7 15 37 46 30 27 18 7 16 29 21 36 25 9 6 32 37 43 10
    2024-01-15 12:57:29,578 - INFO - joeynmt.training -     Hypothesis: 18 15 48 7 7 37 37 30 27 18 18 21 36 29 36 25 9 32 37
    ...
    2024-01-15 13:02:15,428 - INFO - joeynmt.training - Epoch   5, total training loss: 3602.67, num. of seqs: 40000, num. of tokens: 558505, 61.0933[sec]
    2024-01-15 13:02:15,429 - INFO - joeynmt.training - Training ended after   5 epochs.
    2024-01-15 13:02:15,429 - INFO - joeynmt.training - Best validation result (greedy) at step     7000:  95.42 bleu.

The training batch loss is logged every 100 mini-batches, as specified in the configuration, and every 1000 batches the model is validated on the dev set.
So after 1000 batches the model achieves a BLEU score of 22.52 (which will not be that fast for a real translation task, our reverse task is much easier).
You can see that the model prediction is only partially correct.

The loss on individual batches might vary and not only decrease, but after every completed epoch, the accumulated training loss for the whole training set is reported.
This quantity should decrease if your model is properly learning.

Validation Reports
""""""""""""""""""

The scores on the validation set express how well your model is generalizing to unseen data.
The ``validations.txt`` file in the model directory reports the validation results (Loss, evaluation metric (here: BLEU), Perplexity (PPL)) and the current learning rate at every validation point.

For our example, the first lines should look like this:

::

    Steps: 1000     loss: 29.77000  acc: 0.50119    ppl: 5.88275    bleu: 22.51791  LR: 0.00100000  *
    Steps: 2000     loss: 25.81088  acc: 0.61057    ppl: 5.00362    bleu: 57.30290  LR: 0.00100000  *
    Steps: 3000     loss: 25.59565  acc: 0.71042    ppl: 4.86078    bleu: 83.38687  LR: 0.00100000  *
    Steps: 4000     loss: 19.88389  acc: 0.79269    ppl: 3.61883    bleu: 89.83186  LR: 0.00100000  *
    Steps: 5000     loss: 24.50622  acc: 0.76759    ppl: 4.37760    bleu: 89.38016  LR: 0.00100000

Models are saved whenever a new best validation score is reached, in ``batch_no.ckpt``, where ``batch_no`` is the number of batches the model has been trained on so far.
You can see when a checkpoint was saved by the asterisk at the end of the line in ``validations.txt``.
``best.ckpt`` links to the checkpoint that has so far achieved the best validation score.

Learning Curves
"""""""""""""""

Joey NMT provides a script :scripts:`plot_validations.py` to plot validation scores with matplotlib.
You can choose several models and metrics to plot. For now, we're interested in BLEU and perplexity and we want to save it as png.

.. code-block:: bash

    python scripts/plot_validations.py reverse_model --plot-values bleu ppl  --output-path reverse_model/bleu-ppl.png

It should look like this:

.. image:: ../images/bleu-ppl.png
    :width: 150px
    :align: center
    :height: 300px
    :alt: validation curves


Tensorboard
"""""""""""

Joey NMT additionally uses `Tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_ to visualize training and validation curves and attention matrices during training.
Launch Tensorboardlike this:

.. code-block:: bash

    tensorboard --logdir reverse_model/tensorboard

and then open the url (default: ``localhost:6006``) with a browser.

You should see something like that:

.. image:: ../images/tensorboard.png
    :width: 374px
    :align: center
    :height: 196px
    :alt: tensorboard

We can now inspect the training loss curves, both for individual batches

.. image:: ../images/train_train_batch_loss.png
    :width: 265px
    :align: center
    :height: 100px
    :alt: train batch loss

and for the whole training set:

.. image:: ../images/train_train_epoch_loss.png
    :width: 330px
    :align: center
    :height: 200px
    :alt: train epoch loss

and the validation loss:

.. image:: ../images/valid_valid_loss.png
    :width: 330px
    :align: center
    :height: 200px
    :alt: validation loss

Looks good! Training and validation loss are decreasing, that means the model is doing well.


Attention Visualization
^^^^^^^^^^^^^^^^^^^^^^^

Attention scores often allow us a more visual inspection of what the model has learned.
For every pair of source and target tokens, the model computes attention scores, so we can visualize this matrix.
Joey NMT automatically saves plots of attention scores for examples of the validation set (the ones you picked for ``print_valid_examples``) and saves them in your model directory.

Here's an example, target tokens as columns and source tokens as rows:

.. image:: ../images/attention_reverse.png
    :width: 300px
    :align: center
    :height: 300px
    :alt: attention for reverse model

The bright colors mean that these positions got high attention, the dark colors mean there was not much attention.
We can see here that the model has figured out to give "2" on the source high attention when it has to generate "2" on the target side.

Tensorboard (tab: "images") allows us to inspect how attention develops over time, here's what happened for a relatively short sentence:

.. image:: ../images/attention_0.gif
    :width: 400px
    :align: center
    :height: 400px
    :alt: attention over time

For real machine translation tasks, the attention looks less monotonic, for example for an IWSLT de-en model like this:

.. image:: ../images/attention_iwslt.png
    :width: 400px
    :align: center
    :height: 400px
    :alt: attention iwslt


4. Testing
----------

There are *three* options for testing what the model has learned.

In general, testing works by loading a trained model (``load_model`` in the configuration) and feeding it new sources that it will generate predictions for.

Test Set Evaluation
^^^^^^^^^^^^^^^^^^^

For testing and evaluating on the parallel test set specified in the configuration, run

.. code-block:: bash

    python -m joeynmt test reverse_model/config.yaml --output-path reverse_model/predictions

This will generate beam search translations for dev and test set (as specified in the configuration) in ``reverse_model/predictions.[dev|test]``
with the latest/best model in the ``reverse_model`` directory (or a specific checkpoint set with ``load_model``).
It will also evaluate the outputs with ``eval_metrics`` and print the evaluation result.
If ``--output-path`` is not specified, it will not store the translation, and solely do the evaluation and print the results.

The evaluation for our reverse model should look like this:

::

    2024-01-15 13:25:07,213 - INFO - joeynmt.prediction - Decoding on dev set... (device: cuda, n_gpu: 1, use_ddp: False, fp16: True)
    2024-01-15 13:25:07,213 - INFO - joeynmt.prediction - Predicting 1000 example(s)... (Greedy decoding with min_output_length=1, max_output_length=30, return_prob='none', generate_unk=True, repetition_penalty=-1, no_repeat_ngram_size=-1)
    2024-01-15 13:25:20,203 - INFO - joeynmt.prediction - Generation took 12.9892[sec].
    2024-01-15 13:25:20,301 - INFO - joeynmt.metrics - nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
    2024-01-15 13:25:20,302 - INFO - joeynmt.prediction - Evaluation result (greedy): bleu:  95.06, 0.0860[sec]
    2024-01-15 13:25:20,302 - INFO - joeynmt.prediction - Decoding on test set... (device: cuda, n_gpu: 1, use_ddp: False, fp16: True)
    2024-01-15 13:25:20,302 - INFO - joeynmt.prediction - Predicting 1000 example(s)... (Greedy decoding with min_output_length=1, max_output_length=30, return_prob='none', generate_unk=True, repetition_penalty=-1, no_repeat_ngram_size=-1)
    2024-01-15 13:25:32,532 - INFO - joeynmt.prediction - Generation took 12.2290[sec].
    2024-01-15 13:25:32,725 - INFO - joeynmt.metrics - nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.4.0
    2024-01-15 13:25:32,725 - INFO - joeynmt.prediction - Evaluation result (greedy): bleu:  95.19, 0.1821[sec]


Once again you can see that the reverse task is relatively easy to learn, while for translation high BLEU scores like this would be miraculous/suspicious.


File Translation
^^^^^^^^^^^^^^^^

In order to translate the contents of any file (one source sentence per line) not contained in the configuration (here ``my_input.txt``), simply run

.. code-block:: bash

    echo $'2 34 43 21 2 \n3 4 5 6 7 8 9 10 11 12' > my_input.txt
    python -m joeynmt translate reverse_model/config.yaml < my_input.txt

The translations will be written to stdout or alternatively ``--output-path`` if specified.

For this example, the output (all correct!) will be

::

        2 21 43 34 2
        12 11 10 9 8 7 6 5 4 3


Interactive Translation
^^^^^^^^^^^^^^^^^^^^^^^

If you just want to try a few examples, run

.. code-block:: bash

    python -m joeynmt translate reverse_model/config.yaml

and you'll be prompted to type input sentences that Joey NMT will then translate with the model specified in the configuration.

Let's try a challenging long one:

::

    Please enter a source sentence:
    1 23 23 43 34 2 2 2 2 2 4 5 32 47 47 47 21 20 0 10 10 10 10 10 8 7 33 36 37
    Joey NMT:
    33 10 10 37 10 10 0 20 21 47 47 47 32 5 4 2 2 2 2 2 2 34 43 23 1

.. warning::

    Interactive ``translate`` mode doesn't work with Multi-GPU. Please run it on single GPU or CPU.


5. Tuning
---------

Trying out different combinations of hyperparameters to improve the model is called "tuning".
Improving the model could mean in terms of generalization performance at the end of training, faster convergence or making it more efficient or smaller while achieving the same quality.
In our case, that means going back to the configuration and changing a few of the hyperparameters.

For example, let's try out what happens if we increase the batch size to 50 or reduce it to 2 (and change the "model_dir"!).
For a one-to-one comparison, we consequently need to divide or multiply the validation frequency by 5, respectively, since the "steps" are counted in terms of mini-batches.
In the plot below we can see that we reach approximately the same quality after 6 epochs, but that the shape of the curves looks quite different.
In this case, a small mini-batch size leads to the fastest progress but also takes noticeably longer to complete the full 6 epochs in terms of wall-clock time.

.. image:: ../images/reverse_comparison.png
    :width: 450px
    :align: center
    :height: 300px
    :alt: comparison of mini-batch sizes

You might have noticed that there are lots hyperparameters and that you can't possibly try out all combinations to find the best model.
What is commonly done instead of an exhaustive search is grid search over a small subset of hyperparameters,
or random search (`Bergstra & Bengio 2012 <http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf>`_), which is usually the more efficient solution.


6. What's next?
---------------

If you want to implement something new in Joey NMT or dive a bit deeper, you should take a look at the :ref:`overview` and explore the :ref:`api`.

Other than that, we hope that you found this tutorial helpful. Please leave an `issue on Github <https://github.com/joeynmt/joeynmt/issues>`_ if you had trouble with anything or have ideas for improvement.
