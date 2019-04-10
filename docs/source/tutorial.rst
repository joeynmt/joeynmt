.. _tutorial:

========
Tutorial
========

In this tutorial you learn to build a recurrent neural translation system for a toy translation task, how to train, tune and test it.


1. Data Preparation
===================
For training a translation model, you need parallel data, i.e. a collection of source sentences and reference translations that are aligned sentence-by-sentence and stored in two files,
such that each line in the reference file is the translation of the same line in the source file.


Synthetic Data
--------------

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

    cd scripts
    python3 generate_reverse_task.py

This generates 50k training and 1k dev and test examples for integers between 0 and 50 of maximum length 25 for training and 30 for development and testing.
Lets move it to a better directory.

.. code-block:: bash

    mkdir test/data/reverse
    mv train* test/data/reverse/
    mv test* test/data/reverse/
    mv dev* test/data/reverse/


Pre-processing
--------------

Before training a model on it, parallel data is most commonly filtered by length ratio, tokenized and true- or lowercased.
For our reverse task, this is not important, but for real translation data it matters.

The Moses toolkit provides a set of useful `scripts <https://github.com/moses-smt/mosesdecoder/tree/master/scripts>`_ for this purpose.
For a standard pipeline, follow for example the one described in the `Sockeye paper <https://arxiv.org/pdf/1712.05690.pdf>`_.

In addition, you might want to build the NMT model not on the basis of words, but rather sub-words or characters (the ``level`` in JoeyNMT configurations).
Currently, JoeyNMT supports word-based, character-based models and sub-word models with byte-pair-encodings (BPE) as learned with `subword-nmt <https://github.com/rsennrich/subword-nmt>`_.


2. Configuration
================

Once you have the data, it's time to build the NMT model.

In JoeyNMT, experiments are specified in configuration files, in `YAML <http://yaml.org/>`_ format.
Most importantly, the configuration contains the description of the model architecture (e.g. number of hidden units in the encoder RNN),
paths to the training, development and test data, and the training hyperparameters (learning rate, validation frequency etc.).

You can find examples in the `configs directory <https://github.com/joeynmt/joeynmt/tree/master/configs>`_.
`default.yaml <https://github.com/joeynmt/joeynmt/tree/master/configs/default.yaml>`_ contains a detailed explanation of all configuration options.

For the tutorial we'll use `reverse.yaml <https://github.com/joeynmt/joeynmt/tree/master/configs/reverse.yaml>`_. We'll go through it section by section.

1. Data Section
---------------

.. literalinclude:: ../../configs/reverse.yaml
    :linenos:
    :language: python
    :lines:
    :start-after: 2
    :end-before: 18


2. Training Section
-------------------

.. literalinclude:: ../../configs/reverse.yaml
    :linenos:
    :language: python
    :lines:
    :start-after: 22
    :end-before: 47


3. Testing Section
------------------

.. literalinclude:: ../../configs/reverse.yaml
    :linenos:
    :language: python
    :lines:
    :start-after: 18
    :end-before: 22


4. Model Section
----------------

.. literalinclude:: ../../configs/reverse.yaml
    :linenos:
    :language: python
    :lines:
    :start-after: 47
    :end-before: 76


Great! We've specified all that we need to train a translation model for the reverse task.

3. Training
===========

Start
-----
For training, run the following command:

.. code-block:: bash

    python3 -m joeynmt train configs/reverse.yaml


This will train a model on the reverse data specified in the config,
validate on validation data,
and store model parameters, vocabularies, validation outputs and a small number of attention plots in the ``reverse_model`` directory.

Progress Tracking
-----------------

The Log File
^^^^^^^^^^^^

During training the JoeyNMT will print the training log to stdout, and also save it to a log file ``reverse_model/train.log``.
It reports information about the model, like the total number of parameters, the vocabulary size, the data sizes.
You can doublecheck that what you specified in the configuration above is actually matching the model that is now training.

After the reports on the model should see something like this:

::
    2019-04-10 18:27:00,512 Epoch 1 Step: 400 Batch Loss: 3.057961 Tokens per Sec: 17251.341187
    2019-04-10 18:27:02,027 Epoch 1 Step: 500 Batch Loss: 2.064026 Tokens per Sec: 21625.927728
    2019-04-10 18:27:03,787 Hooray! New best validation result [eval_metric]!
    ...
    2019-04-10 18:27:03,790 Example #2
    2019-04-10 18:27:03,790         Raw source: ['6', '2', '4']
    2019-04-10 18:27:03,790         Source: 6 2 4
    2019-04-10 18:27:03,790         Reference: 4 2 6
    2019-04-10 18:27:03,790         Raw hypothesis: ['4', '6', '6']
    2019-04-10 18:27:03,790         Hypothesis: 4 6 6
    2019-04-10 18:27:03,790 Example #3
    2019-04-10 18:27:03,790         Raw source: ['7', '3']
    2019-04-10 18:27:03,790         Source: 7 3
    2019-04-10 18:27:03,790         Reference: 3 7
    2019-04-10 18:27:03,790         Raw hypothesis: ['3', '7']
    2019-04-10 18:27:03,790         Hypothesis: 3 7
    2019-04-10 18:27:03,790 Validation result at epoch 1, step 500: bleu: 22.986330, loss: 7783.417969, ppl: 3.297102, duration: 1.7626s

The training batch loss is logged every 100 batch, as specified in the configuration, and every 500 batches the model is validated on the dev set.
So after 500 batches the model achieves a BLEU score of 22.99 (which will not be that fast for a real translation task, our reverse task is much easier).
For example #2 it wrongly "translated" the 2nd number (compare "Hypothesis" with "Reference"), but for example #3 it's already correct.

The loss on individual batches might vary and not only decrease, but after every completed epoch, the accumulated training loss for the whole training set is reported.
This quantity should decrease if your model is properly learning.

Validation Reports
^^^^^^^^^^^^^^^^^^

The scores on the validation set express how well your model is generalizing to unseen data.
The ``validations.txt`` file in the model directory reports the validation results (Loss, evaluation metric (here: BLEU), Perplexity (PPL)) and the current learning rate at every validation point.

For our example, the first lines should look like this:

::
    Steps: 500      Loss: 7783.41797        PPL: 3.29710    bleu: 22.98633  LR: 0.00100000  *
    Steps: 1000     Loss: 1491.60852        PPL: 1.25688    bleu: 88.90597  LR: 0.00100000  *
    Steps: 1500     Loss: 319.96951 PPL: 1.05027    bleu: 97.31911  LR: 0.00100000  *
    Steps: 2000     Loss: 126.47280 PPL: 1.01957    bleu: 98.87817  LR: 0.00100000  *

Models are saved whenever a new best validation score is reached, in ``batch_no.ckpt``, where ``batch_no`` is the number of batches the model has been trained on so far.
You can see when a checkpoint was saved by the asterisk at the end of the line in ``validations.txt``.
``best.ckpt`` links to the checkpoint that has so far achieved the best validation score.

Learning Curves
^^^^^^^^^^^^^^^

JoeyNMT provides a `script <https://github.com/joeynmt/joeynmt/blob/master/scripts/plot_validations.py>`_ to plot validation scores with matplotlib.
You can choose several models and metrics to plot. For now, we're interested in BLEU and perplexity and we want to save it as png.

.. code-block:: bash

    python3 scripts/plot_validations.py reverse_model --plot_values bleu PPL  --output_path reverse_model/bleu-ppl.png

It should like this:

.. image:: ../images/bleu-ppl.png
    :width: 150px
    :align: center
    :height: 300px
    :alt: validation curves

Tensorboard
^^^^^^^^^^^

JoeyNMT additionally uses `TensorboardX <https://github.com/lanpa/tensorboardX>`_ to visualize training and validation curves and attention matrices during training.
Launch `Tensorboard <https://github.com/tensorflow/tensorboard>`_ (requires installation that is not included in JoeyNMTs requirements) like this:

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
-----------------------

Attention scores often allow us a more visual inspection of what the model has learned.
For every pair of source and target token the model computes attention scores, so we can visualize this matrix.
JoeyNMT automatically saves plots of attention scores for examples of the validation set (the ones you picked for ``print_valid_examples``) and saves them in your model directory.

Here's an example, target tokens as columns and source tokens as rows:

.. image:: ../images/attention_reverse.png
    :width: 330px
    :align: center
    :height: 200px
    :alt: attention for reverse model

The bright colors mean that these positions got high attention, the dark colors mean there was not much attention.
We can see here that the model has figured out to give "2" on the source high attention when it has to generate "2" on the target side.

Tensorboard (tab: "images") allows us to inspect how attention develops over time, here's happened for the first sentence of the validation set:

.. image:: ../images/attention_0.gif
    :alt: attention over time

For real machine translation tasks, this would look less monotone, for example for an IWSLT de-en model like this:

.. image:: ../images/attention_iwslt.png
    :width: 300px
    :align: center
    :height: 300px
    :alt: attention iwslt


4. Testing
==========

5. Tuning
=========
