.. _faq:

==========================
Frequently Asked Questions
==========================

Documentation
-------------
- **Are there any Notebooks for Joey?**
    Check out the `Colab Notebook <https://github.com/masakhane-io/masakhane/blob/master/starter_notebook.ipynb>`_ from the Masakhane project that walks you through the installation, data preparation, training, evaluation.

- **Is there a bunch of scripts to run all those Joey commands?**
    Check out the scripts compiled in `Joey Toy Models <https://github.com/bricksdont/joeynmt-toy-models>`_, that also walk you through the installation, data preparation, training, evaluation, and even data download and pre-processing.

- **I can't find the information I'm looking for. What now?**
    Open an issue on GitHub or post a question on gitter.

Usage
-----

Training
^^^^^^^^

- **How can I train the model on GPU/CPU?**
   First of all, make sure you have the correct version of pytorch installed. 
   When running on *GPU* you need to manually install the suitable PyTorch version for your `CUDA <https://developer.nvidia.com/cuda-zone>`_ version. This is described in the `PyTorch installation instructions <https://pytorch.org/get-started/locally/>`_.
   Then set the ``use_cuda`` flag in the configuration to True for training on GPU (requires CUDA) or to False for training on CPU.

- **Does Joey NMT support multi-GPU processing?**
   ~~At the current stage, the code does not support multi-GPU processing. Contributions welcome :)~~
   [UPDATE] In version 1.0.0, we integrated multi-gpu and half-precision support.

- **How can I stop training?**
   Simply press Control+C.

- **My training data is huge and I actually don't want to train on it all. What can I do?**
    You could use the ``random_train_subset`` parameter in the data section of the configuration to load only a random subset of the training data. If you change the random seed, this selection changes too. So you could train on multiple random subsets and then ensemble the models with ``scripts/average_checkpoints.py``.

- **How can I see how well my model is doing?**
   1. *Training log*: Validation results and training loss (after each epoch and batch) are reported in the training log file ``train.log`` in your model directory.
   2. *Validation reports*: ``validations.txt`` contains the validation results, learning rates and indicators when a checkpoint was saved. You can easily plot the validation results with `this script <https://github.com/joeynmt/joeynmt/blob/master/scripts/plot_validations.py>`_, e.g.
    ::

        python3 scripts/plot_validation.py model_dir --plot_values bleu PPL --output_path my_plot.pdf

   3. *Tensorboard*: Validation results, training losses and attention scores are also stored in summaries for Tensorboard. Launch Tensorboard with
    ::

        tensorboard --logdir model_dir/tensorboard

    and then open the url (default: ``localhost:6006``) with a browser.

   See :ref:`tutorial`, section "Progress Tracking", for a detailed description of the quantities being logged.

- **How often should I validate?**
   Depends on the size of your data. For most use-cases you want to validate at least once per epoch.
   Say you have 100k training examples and train with mini-batches of size 20, then you should set ``validation_freq`` to 5000 (100k/20) to validate once per epoch.

- **How can I perform domain adaptation or fine-tuning?**
   Both approaches are similar, so we call the fine-tuning data *in-domain* data in the following.
   
     1. First train your model on one dataset (the *out-of-domain* data).
   
     2. Modify the original configuration file (or better a copy of it) in the data section to point to the new *in-domain* data.
        Specify which vocabularies to use: ``src_vocab: out-of-domain-model/src_vocab.txt`` and likewise for ``trg_vocab``.
        You have to specify this, otherwise JoeyNMT will try to build a new vocabulary from the new in-domain data, which the out-of-domain model wasn't built with.
        In the training section, specify which checkpoint of the out-of-domain model you want to start adapting: ``load_model: out-of-domain-model/best.ckpt``.
        If you set ``reset_best_ckpt'': True'', previously stored high scores under your metric will be ignored, and if you set ``reset_scheduler'' and ``reset_optimizer'' you can also overwrite the stored scheduler and optimizer with the new ones in your configuration.
        Use this if the scores on your new dev set are lower than on the old dev set, or if you use a different metric or schedule for fine-tuning.
    
     3. Train the in-domain model.

- **What if training is interrupted and I need to resume it?**
   Modify the configuration to load the latest checkpoint (``load_model``) and the vocabularies (``src_vocab``, ``trg_vocab``) and to write the model into a new directory (``model_dir``).
   Then train with this configuration.


Tuning
^^^^^^
- **Which default hyperparameters should I use?**
   There is no universal answer to this question. We recommend you to check publications that used the same data as you're using (or at least the same language pair and data size)
   and find out how large their models where, how long they trained them etc.
   You might also get inspiration from the benchmarks that we report. Their configuration files can be found in the ``configs`` directory.

- **Which hyperparameters should I change first?**
    As above, there is no universal answer. Some things to consider:

    - The *learning rate* determines how fast you can possibly learn.
      If you use a learning rate scheduler, make sure to configure it in a way that it doesn't reduce the learning rate too fast.
      Different optimizers need individually tuned learning rates as well.
    - The *model size and depth* matters. Check the benchmarks and their model and data sizes to get an estimate what might work.

Tensorboard
^^^^^^^^^^^
- **How can I start Tensorboard for a model that I trained on a remote server?**
   Start jupyter notebook in the Joey NMT directory, remote_port_number should be a free port, e.g. 8889.

   Create an SSH tunnel on the local machine (with free ports yyyy (local) and xxxx (remote)):

   .. code-block:: bash

        ssh -N -L localhost:yyyy:localhost:xxxx <remote_user@remote_user>

   On the remote machine, launch tensorboard and pass it the path to the tensorboard logs of your model:

   .. code-block:: bash

        tensorboard --logdir model_dir/tensorboard --host=localhost --port=xxxx


   Then navigate to `localhost:yyyy` in a browser on your local machine.

Configurations
^^^^^^^^^^^^^^
- **Where can I find the default values for the settings in the configuration file?**
   Either check `the configuration file <https://github.com/joeynmt/joeynmt/blob/master/configs/small.yaml>`_ or :ref:`api`
   for individual modules.
   Please note that there is no guarantee that the default setting is a good setting.

- **What happens if I made a mistake when configuring my model?**
   JoeyNMT will complain by raising a ``ConfigurationError``.

- **How many parameters has my model?**
   The number of parameters is logged in the training log file. You can find it in the model directory in ``train.log``. Search for the line containing "Total params:".

- **What's the influence of the random seed?**
   The random seed is used for all random factors in NMT training, such as the initialization of model parameters and the order of training samples.
   If you train two identical models with the same random seed, they should behave exactly the same.

- **How do you count the number of hidden units for bi-directional RNNs?**
   A bi-directional RNN with *k* hidden units will have *k* hidden units in the forward RNN plus *k* for the backward RNN.
   This might be different in other toolkits where the number of hidden units is divided by two to use half of them each for backward and forward RNN.

- **My model with configs/small.yaml doesn't perform well.`**
  No surprise! This configuration is created for the purpose of documentation: it contains all parameter settings with a description. It does not perform well on the actual task that it uses. Try the reverse or copy task instead!

- **What does batch_type mean?**
  The code operates on mini-batches, i.e., blocks of inputs instead of single inputs. Several inputs are grouped into one mini-batch. This grouping can either be done by defining a maximum number of sentences to be in one mini-batch (`batch_type: "sentence"`), or by a maximum number of tokens (`batch_type: "token"`). For Transformer models, mini-batching is usually done by tokens.

- **Do I need a warm-up scheduler with the Transformer architecture?**
   No. The 'Noam scheduler' that was introduced with the original Transformer architecture works well for the data sets (several millions) described in the `paper (Vaswani et al. 2017) <https://arxiv.org/pdf/1706.03762.pdf>`_. However, on different data it might require a careful tuning of the warm-up schedule. We experienced good performance with the plateau scheduler as well, which is usally easier to tune. `Popel and Bojar (2018) <https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf>`_ give further tips on how to tune the hyper-parameters for the Transformer.

Data
^^^^
- **Does JoeyNMT pre-process my data?**
    JoeyNMT does *not* include any pre-processing like tokenization, filtering by length ratio, normalization or learning/applying of BPEs.
    For that purpose, you might find the  `tools provided by the Moses decoder <https://github.com/moses-smt/mosesdecoder/tree/master/scripts>`_ useful, as well as the `subwordnmt <https://github.com/rsennrich/subword-nmt>`_ or `sentencepiece <https://github.com/google/sentencepiece>`_ library for BPEs. An example of a pre-processing pipeline is show in the `data preparation script for IWLST 2014 <https://github.com/joeynmt/joeynmt/blob/master/scripts/get_iwslt14_bpe.sh>`_.
    However, the training data gets *filtered* by the ``max_sent_length`` (keeping all training instances where source and target are up to that length) that you specify in the data section of the configuration file.
    You can find an example of a data pre-processing pipeline `here <https://github.com/bricksdont/joeynmt-toy-models/blob/master/scripts/preprocess.sh>`_.

- **Does JoeyNMT post-process your data?**
  JoeyNMT does generally *not* perform any post-processing like detokenization, recasing or the like. The only exception is when you run it with ´level='bpe'´ -- then it *merges* the BPEs for your convenience. This holds for computing validation BLEU and test BLEU scores, so that they're not computed on subwords, but the previously split tokens.

Debugging
^^^^^^^^^
- **My model doesn't work. What can I do?**
   First of all, invest in diagnostics: what exactly is not working? Is the training loss going down? Is the validation loss going down? Are there any patterns in the weirdness of the model outputs? Answers to these questions will help you locate the source of the problem.
   Andrej Karpathy wrote this wonderful `recipe for training neural nets <http://karpathy.github.io/2019/04/25/recipe/>`_ by  - it has lots of advice on how to find out what's going wrong and how to fix it.
   Specifically for NMT, here's three things we can recommend:
   - *Synthetic data*: If you modified the code, it might help to inspect tensors and outputs manually for a synthetic task like the reverse task presented in the :ref:`tutorial`.
   - *Data*: If you're working with a standard model, doublecheck whether your data is properly aligned, properly pre-processed, properly filtered and whether the vocabularies cover a reasonable amount of tokens.
   - *Hyperparameters*: Try a smaller/larger/deeper/shallower model architecture with smaller/larger learning rates, different optimizers and turn off schedulers. It might be worth to try different initialization options. Train longer and validate less frequently, maybe training just takes longer than you'd expect.

- **My model takes too much memory. What can I do?**
   Consider reducing ``batch_size``. The mini-batch size can be virtually increased by a factor of *k* by setting ``batch_multiplier`` to *k*.
   Tensor operations are still performed with ``batch_size`` instances each, but model updates are done after *k* of these mini-batches.

- **My model performs well on the validation set, but terrible on the test set. What's wrong?**
   Make sure that your validation set is similar to the data you want to test on, that it's large enough and that you're not "over-tuning" your model.

- **My model produces translations that are generally too short. What's wrong?**
   Make sure that ``max_sent_length`` for the filtering of the data (data section in configuration) is set sufficiently high. The training log reports how many training sentences remain after filtering.
   ``max_output_length`` (training section) limits the length of the outputs during inference, so make sure this one is also set correctly.

- **Evaluation breaks because I get an empty iterator. What's wrong?**
   If you're using `batch_type: token`, try increasing the `eval_batch_size`.

Features
--------
- **Which models does Joey NMT implement?**
   For the exact description of the RNN and Transformer model, check out the `paper <https://www.cl.uni-heidelberg.de/~kreutzer/joeynmt/joeynmt_demo.pdf>`_.

- **Why is there no convolutional model?**
   We might add it in the future, but from our experience, the most popular models are recurrent and self-attentional.

- **How are the parameters initialized?**
   Check the description in `initialization.py <https://github.com/joeynmt/joeynmt/blob/master/joeynmt/initialization.py#L60>`_.

- **Is there the option to ensemble multiple models?**
   You can do checkpoint averaging to combine multiple models. Use the `average_checkpoints script <https://github.com/joeynmt/joeynmt/blob/master/scripts/average_checkpoints.py>`_.

- **What is a bridge?**
   We call the connection between recurrent encoder and decoder states the *bridge*.
   This can either mean that the decoder states are initialized by copying the last (forward) encoder state (``init_hidden: "last"``),
   by learning a projection of the last encoder state (``init_hidden: "bridge"``) or simply zeros (``init_hidden: "zero"``).

- **Does learning rate scheduling matter?**
   Yes. There's a whole branch of research on how to find and modify a good learning rate so that your model ends up in a good place.
   For JoeyNMT it's most important that you don't decrease your learning rate too quickly, which might happen if you train with very frequent validations (``validation_freq``) and low ``patience`` for a plateau-based scheduler. So if you change the validation frequency, adapt the patience as well.
   We recommend to start by finding a good constant learning rate and then add a scheduler that decays this initial rate at a point where the constant learning rate does not further improve the model.

- **What is early stopping?**
   Early stopping means that training should be stopped when the model's generalization starts to degrade.
   Jason Brownlee wrote a neat `blogpost <https://machinelearningmastery.com/early-stopping-to-avoid-overtraining-neural-network-models/>`_ describing intuition and techniques for early stopping.
   In JoeyNMT, model checkpoints are stored whenever a new high score is achieved on the validation set, so when training ends, the latest checkpoint automatically captures the model parameters at the early stopping point.
   There's three options for measuring the high score on the validation set: the evaluation metric (``eval_metric``), perplexity (``ppl``), and the loss (``loss``). Set ``early_stopping_metric`` in the training configuration to either of those.

- **Is validation performed with greedy decoding or beam search?**
   Greedy decoding, since it's faster and usually aligns with model selection by beam search validation.

- **What's the difference between "max_sent_length" and and "max_output_length"?**
   ``max_sent_length`` determines the maximum source and target length of the training data,
   ``max_output_length`` is the maximum length of the translations that your model will be asked to produce.

- **How is the vocabulary generated?**
    See the :ref:`tutorial`, section "Configuration - Data Section".

- **What does freezing mean?**
   *Freezing* means that you don't update a subset of your parameters. If you freeze all parts of your model, it won't get updated (which doesn't make much sense).
   It might, however, might sense to update only a subset of the parameters in the case where you have a pre-trained model and want to carefully fine-tune it to e.g. a new domain.
   For the modules you want to freeze, set ``freeze: True`` in the corresponding configuration section.


Model Extensions
----------------
- **I want to extend Joey NMT -- where do I start? Where do I have to modify the code?**
    Depends on the scope of your extension. In general, we can recommend describing the desired behavior in the config (e.g. 'use_my_feature:True') and then passing this value along the forward pass and modify the model according to it.
    If your just loading more/richer inputs, you will only have to modify the part from the corpus reading to the encoder input. If you want to modify the training objective, you will naturally work in 'loss.py'.
    Logging and unit tests are very useful tools for tracking the changes of your implementation as well.

- **How do I integrate a new learning rate scheduler?**
    1. Check out the existing schedulers in `builders.py <https://github.com/joeynmt/joeynmt/blob/master/joeynmt/builders.py>`_, some of them are imported from PyTorch. The "Noam" scheduler is implemented here directly, you can use its code as a template how to implement a new scheduler. 
  
    2. You basically need to implement the ``step`` function that implements whatever happens when the scheduler is asked to make a step (either after every validation (``scheduler_step_at="validation"``) or every batch (``scheduler_step_at="step"``)). In that step, the learning rate can be modified just as you like (``rate = self._compute_rate()``). In order to make an effective update of the learning rate, the learning rate for the optimizer's parameter groups have to be set to the new value (``for p in self.optimizer.param_groups: p['lr'] = rate``).
  
    3. The last thing that is missing is the parsing of configuration parameters to build the scheduler object. Once again, follow the example of existing schedulers and integrate the code for constructing your new scheduler in the ``build_scheduler`` function.
  
    4. Give the new scheduler a try! Integrate it in a basic configuration file and check in the training log and the validation reports whether the learning rate is behaving as desired.


Miscellaneous
-------------
- **Why should I use JoeyNMT rather than other NMT toolkits?**
    It's easy to use, it is well documented, and it works just as well as other toolkits out-of-the-box. It does and will not implement all latest features, but rather the core features that make up for 99% of the quality.
    That means for you, once you know how to work with it, we guarantee you the code won't completely change from one day to the next.

- **I found a bug in your code, what should I do?**
    Make a Pull Request on GitHub and describe what it did and how you fixed it.

- **How can I check whether my model is significantly better than my baseline model?**
    Repeat your experiment with multiple random seeds (``random_seed``) to measure the variance.
    You can use techniques like `approximate randomization <https://www.cl.uni-heidelberg.de/~riezler/publications/papers/ACL05WS.pdf>`_ or `bootstrap sampling <http://homepages.inf.ed.ac.uk/pkoehn/publications/bootstrap2004.pdf>`_ to test the significance of the difference in evaluation score between the baseline's output and your model's output, e.g. with `multeval <https://github.com/jhclark/multeval>`_.

- **Where can I find training data?**
    See :ref:`resources`, section "Data".


Contributing
------------
- **How can I contribute?**
    Check out the current issues and look for "beginner-friendly" tags and grab one of these.

- **What's in a Pull Request?**
    Opening a pull request means that you have written code that you want to contribute to Joey NMT. In order to communicate what your code does, please write a description of new features, defaults etc.
    Your new code should also pass tests and adher to style guidelines, this will be tested automatically. The code will only be pushed when all issues raised by reviewers have been addressed.
    See also `here <https://help.github.com/en/articles/about-pull-requests>`_.

Evaluation
----------
- **Which quality metrics does JoeyNMT report?**
    JoeyNMT reports `BLEU <https://www.aclweb.org/anthology/P02-1040.pdf>`_, `chrF <https://www.aclweb.org/anthology/W15-3049.pdf>`_, sentence- and token-level accuracy. You can choose which of those to report with setting `eval_metric` accordingly. As a default, we recommend BLEU since it is a standard metric. However, not all BLEU implementations compute the score in the same way, as discussed `in this paper by Matt Post <https://www.aclweb.org/anthology/W18-6319/>`_. So the scores that you obtain might not be comparable to those published in a paper, *even* if the data is identical! 
    
- **Which library is JoeyNMT using to compute BLEU scores?**
    JoeyNMT uses `sacrebleu <ttps://github.com/mjpost/sacrebleu>`_ to compute BLEU and chrF scores.
    It uses the `raw_corpus_bleu <https://github.com/mjpost/sacrebleu/blob/f54908ac00879f666c92f4174367bcd3a8723197/sacrebleu/sacrebleu.py#L653>`_ scoring function that excludes special de/tokenization or smoothing. This is done to respect the tokenization that is inherent in the provided input data. However, that means that the BLEU score you get out of Joey is *dependent on your input tokenization*, so be careful when comparing it to scores you find in literature.
    
- **Can I publish the BLEU scores JoeyNMT reports on my test set?**
    As described in the two preceding questions, BLEU reporting has to be handled with care, since it depends on tokenizers and implementations. Generally, whenever you report BLEU scores, report as well how you computed them. This is essential for reproducability of results and future comparisons. If you compare to previous benchmarks or scores, first find out how these were computed.
    Our recommendation is as follows:
      1. Use the scores that Joey reports on your validation set for tuning and selecting the best model. 
      2. Then translate your test set once (in "translate" mode), and post-process the produced translations accordingly, e.g., detokenize it, restore casing.
      3. Use the BLEU scoring library of your choice, this is the one that is reported in previous benchmarks, or e.g. sacrebleu (see above). Make sure to set tokenization flags correctly.
      4. Report these scores together with a description of how you computed them, ideally provide a script with your code.  

