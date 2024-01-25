.. _cli:

======================
Command-line Interface
======================

Joey NMT has 3 modes: ``train``, ``test``, and ``translate``, and all of them takes a `YAML <https://yaml.org/>`_-style config file as argument. You can find examples in the ``configs`` directory. :configs:`transformer_small.yaml` contains a detailed explanation of configuration options.

Most importantly, the configuration contains the description of the model architecture (e.g. number of hidden units in the encoder RNN), paths to the training, development and test data, and the training hyperparameters (learning rate, validation frequency etc.).

.. note::

    Note that subword model training and joint vocabulary creation is not included in the 3 modes above, has to be done separately.
    We provide a script that takes care of it: :scripts:`build_vocab.py`.

    .. code-block:: bash

        python scripts/build_vocab.py configs/transformer_small.yaml --joint


``train`` mode
--------------

For training, run 

.. code-block:: bash

    python -m joeynmt train configs/transformer_small.yaml

This will train a model on the training data, validate on validation data, and store model parameters, vocabularies, validation outputs. All needed information should be specified in the ``data``, ``training`` and ``model`` sections of the config file (here :configs:`transformer_small.yaml`).

::

    model_dir/
    ├── *.ckpt          # checkpoints
    ├── *.hyps          # translated texts at validation
    ├── config.yaml     # config file
    ├── spm.model       # sentencepiece model / subword-nmt codes file
    ├── src_vocab.txt   # src vocab
    ├── trg_vocab.txt   # trg vocab
    ├── train.log       # train log
    └── validation.txt  # validation scores

.. danger::

    Be careful not to overwrite ``model_dir``; set ``overwrite: False`` in the config file.


``test`` mode
-------------

This mode will generate translations for validation and test set (as specified in the configuration) in ``model_dir/out.[dev|test]``.

.. code-block:: bash

    python -m joeynmt test configs/transformer_small.yaml

You can specify the ckpt path explicitly in the config file. If ``load_model`` is not given in the config, the best model in ``model_dir`` will be used to generate translations.

You can specify i.e. `sacrebleu <https://github.com/mjpost/sacrebleu>`_ options in the ``test`` section of the config file.

.. note::

    :scripts:`average_checkpoints.py` will generate averaged checkpoints for you.

    .. code-block:: bash

        python scripts/average_checkpoints.py --inputs model_dir/*00.ckpt --output model_dir/avg.ckpt


If you want to output the log-probabilities of the hypotheses or references, you can specify ``return_score: 'hyp'`` or ``return_score: 'ref'`` in the testing section of the config. And run ``test`` with ``--output-path`` and ``--save-scores`` options.

.. code-block:: bash

    python -m joeynmt test configs/transformer_small.yaml --output-path model_dir/pred --save-scores

This will generate ``model_dir/pred.{dev|test}.{scores|tokens}`` which contains scores and corresponding tokens.

.. tip::

    - If you set ``return_score: 'hyp'`` with greedy decoding, then token-wise scores will be returned. The beam search will return sequence-level scores, because the scores are summed up per sequence during beam exploration.
    - If you set ``return_score: 'ref'``, the model looks up the probabilities of the given ground truth tokens, and both decoding and evaluation will be skipped.
    - If you specify ``n_best`` > 1 in config, the first translation in the nbest list will be used in the evaluation.


``translate`` mode
------------------

This mode accepts inputs from stdin and generate translations.


File translation
^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -m joeynmt translate configs/transformer_small.yaml < my_input.txt > output.txt


Interactive translation
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -m joeynmt translate configs/transformer_small.yaml

You'll be prompted to type an input sentence. Joey NMT will then translate with the model specified in the config file.

