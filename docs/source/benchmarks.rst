.. _benchmarks:

==========
Benchmarks
==========


We provide several pretrained models with their benchmark results.


JoeyNMT v2.x
------------

IWSLT14 de/en/fr multilingual
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


We trained this multilingual model with JoeyNMT v2.3.0 using DDP.

+-----------+--------------+---------------+-------+-------+---------+--------------------------------------------------------------------+
| Direction | Architecture | Tokenizer     | dev   | test  | #params | download                                                           |
+===========+==============+===============+=======+=======+=========+====================================================================+
| en->de    | Transformer  | sentencepiece |    \- | 28.88 | 200M    | `iwslt14_prompt <https://huggingface.co/may-ohta/iwslt14_prompt>`_ |
+-----------+              +               +-------+-------+         +                                                                    +
| de->en    |              |               |    \- | 35.28 |         |                                                                    |
+-----------+              +               +-------+-------+         +                                                                    +
| en->fr    |              |               |    \- | 38.86 |         |                                                                    |
+-----------+              +               +-------+-------+         +                                                                    +
| fr->en    |              |               |    \- | 40.35 |         |                                                                    |
+-----------+--------------+---------------+-------+-------+---------+--------------------------------------------------------------------+

sacrebleu signature: `nrefs:1|case:lc|eff:no|tok:13a|smooth:exp|version:2.4.0`


WMT14 ende / deen
^^^^^^^^^^^^^^^^^

We trained the models with JoeyNMT v2.1.0 from scratch.

cf) `wmt14 deen leaderboard <https://paperswithcode.com/sota/machine-translation-on-wmt2014-german-english>`_ in paperswithcode

+-----------+--------------+---------------+-------+-------+---------+----------------------------------------------------------------------------------------------------+
| Direction | Architecture | Tokenizer     | dev   | test  | #params | download                                                                                           |
+===========+==============+===============+=======+=======+=========+====================================================================================================+
| en->de    | Transformer  | sentencepiece | 24.36 | 24.38 | 60.5M   | `wmt14_ende.tar.gz <https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2/wmt14_ende.tar.gz>`_ (766M) |
+-----------+--------------+---------------+-------+-------+---------+----------------------------------------------------------------------------------------------------+
| de->en    | Transformer  | sentencepiece | 30.60 | 30.51 | 60.5M   | `wmt14_deen.tar.gz <https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2/wmt14_deen.tar.gz>`_ (766M) |
+-----------+--------------+---------------+-------+-------+---------+----------------------------------------------------------------------------------------------------+

sacrebleu signature: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.2.0`


JoeyNMT v1.x
------------

.. warning::

    The following models are trained with JoeynNMT v1.x, and decoded with Joey NMT v2.0. 
    See ``config_v1.yaml`` and ``config_v2.yaml`` in the linked tar.gz, respectively.
    Joey NMT v1.x benchmarks are archived `here <https://github.com/joeynmt/joeynmt/blob/main/docs/benchmarks_v1.md>`__.


IWSLT14 deen
^^^^^^^^^^^^

Pre-processing with Moses decoder tools as in this :scripts:`script <get_iwslt14_bpe.sh>`.

+-----------+--------------+-------------+-------+-------+---------+----------------------------------------------------------------------------------------------------------------------------------------+
| Direction | Architecture | Tokenizer   | dev   | test  | #params | download                                                                                                                               |
+===========+==============+=============+=======+=======+=========+========================================================================================================================================+
| de->en    | RNN          | subword-nmt | 31.77 | 30.74 | 61M     | `rnn_iwslt14_deen_bpe.tar.gz <https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2/rnn_iwslt14_deen_bpe.tar.gz>`_ (672M)                 |
+-----------+--------------+-------------+-------+-------+---------+----------------------------------------------------------------------------------------------------------------------------------------+
| de->en    | Transformer  | subword-nmt | 34.53 | 33.73 | 19M     | `transformer_iwslt14_deen_bpe.tar.gz <https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2/transformer_iwslt14_deen_bpe.tar.gz>`_ (221M) |
+-----------+--------------+-------------+-------+-------+---------+----------------------------------------------------------------------------------------------------------------------------------------+

sacrebleu signature: `nrefs:1|case:lc|eff:no|tok:13a|smooth:exp|version:2.0.0`

.. note::

    For interactive translate mode, you should specify ``pretokenizer: "moses"`` in both src's and trg's ``tokenizer_cfg``,
    so that you can input raw sentences. Then ``MosesTokenizer`` and ``MosesDetokenizer`` will be applied internally.
    For test mode, we used the preprocessed texts as input and set ``pretokenizer: "none"`` in the config.


Masakhane JW300 afen / enaf
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We picked the pretrained models and configs (bpe codes file etc.) from `masakhane.io <https://github.com/masakhane-io/masakhane-mt>`_.

+-----------+--------------+-------------+-------+-------+---------+----------------------------------------------------------------------------------------------------------------------------+
| Direction | Architecture | Tokenizer   | dev   | test  | #params | download                                                                                                                   |
+===========+==============+=============+=======+=======+=========+============================================================================================================================+
| af->en    | Transformer  | subword-nmt | \-    | 57.70 | 46M     | `transformer_jw300_afen.tar.gz <https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2/transformer_jw300_afen.tar.gz>`_ (525M) |
+-----------+--------------+-------------+-------+-------+---------+----------------------------------------------------------------------------------------------------------------------------+
| en->af    | Transformer  | subword-nmt | 47.24 | 47.31 | 24M     | `transformer_jw300_enaf.tar.gz <https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2/transformer_jw300_enaf.tar.gz>`_ (285M) |
+-----------+--------------+-------------+-------+-------+---------+----------------------------------------------------------------------------------------------------------------------------+

sacrebleu signature: `nrefs:1|case:mixed|eff:no|tok:intl|smooth:exp|version:2.0.0`


JParaCrawl enja / jaen
^^^^^^^^^^^^^^^^^^^^^^

For training, we split JparaCrawl v2 into train and dev set and trained a model on them.
Please check the preprocessing script `here <https://github.com/joeynmt/joeynmt/blob/v2.2/scripts/get_jparacrawl.sh>`__.
We tested then on `kftt <http://www.phontron.com/kftt/>`_ test set and `wmt20 <https://data.statmt.org/wmt20/translation-task/>`_ test set, respectively.

+-----------+--------------+---------------+-------+-------+---------+---------------------------------------------------------------------------------------------------------------+
| Direction | Architecture | Tokenizer     | kftt  | wmt20 | #params | download                                                                                                      |
+===========+==============+===============+=======+=======+=========+===============================================================================================================+
| af->en    | Transformer  | sentencepiece | 17.66 | 14.31 | 225M    | `jparacrawl_enja.tar.gz <https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2/jparacrawl_enja.tar.gz>`_ (2.3GB) |
+-----------+--------------+---------------+-------+-------+---------+---------------------------------------------------------------------------------------------------------------+
| en->af    | Transformer  | sentencepiece | 14.97 | 11.49 | 221M    | `jparacrawl_jaen.tar.gz <https://cl.uni-heidelberg.de/statnlpgroup/joeynmt2/jparacrawl_jaen.tar.gz>`_ (2.2GB) |
+-----------+--------------+---------------+-------+-------+---------+---------------------------------------------------------------------------------------------------------------+

sacrebleu signature:
    - en->ja: `nrefs:1|case:mixed|eff:no|tok:ja-mecab-0.996-IPA|smooth:exp|version:2.0.0`
    - ja->en: `nrefs:1|case:mixed|eff:no|tok:intl|smooth:exp|version:2.0.0`

(Note: In wmt20 test set, `newstest2020-enja` has 1000 examples, `newstest2020-jaen` has 993 examples.)
