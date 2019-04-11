.. _faq:

==========================
Frequently Asked Questions
==========================

Usage
-----

Training
^^^^^^^^

- **How can I train the model on GPU/CPU?**
   Set the ``use_cuda`` flag in the configuration to True for training on GPU (requires CUDA) or to False for training on CPU.

- **How can I stop training?**
   Simply press Control+C.

- **How can I see how well my model is doing?**
   1. *Training log*: Validation results and training loss (after each epoch and batch) are reported in the training log file ``train.log`` in your model directory.
   2. *Validation reports*: ``validations.txt`` contains the validation results, learning rates and indicators when a checkpoint was saved.
     You can easily plot the validation results with `this script <https://github.com/joeynmt/joeynmt/blob/master/scripts/plot_validations.py>`_, e.g.
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

- **How can I perform domain adaptation?**
   1. First train your model on one dataset (the *out-of-domain* data).
   2. Modify the original configuration file (or better a copy of it) in the data section to point to the new *in-domain* data.
     Specify which vocabularies to use: ``src_vocab: out-of-domain-model/src_vocab.txt`` and likewise for ``trg_vocab``.
     You have to specify this, otherwise JoeyNMT will try to build a new vocabulary from the new in-domain data, which the out-of-domain model wasn't built with.
     In the training section, specify which checkpoint of the out-of-domain model you want to start adapting: ``load_model: out-of-domain-model/best.ckpt``.
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
   Start jupyter notebook in the JoeyNMT directory, remote_port_number should be a free port, e.g. 8889.

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
   Either check `the default configuration file <https://github.com/joeynmt/joeynmt/blob/master/configs/default.yaml>`_ or :ref:`api`
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

Data
^^^^
- **Does JoeyNMT pre-process my data?**
   JoeyNMT does *not* include any pre-processing like tokenization, filtering by length ratio, normalization or learning/applying of BPEs.
   For that purpose, you might find the tools provided by the Moses decoder useful, as well as the subwordnmt library for BPEs.
   However, the training data gets *filtered* by the ``max_sent_length`` (keeping all training instances where source and target are up to that length)
   that you specify in the data section of the configuration file.

Debugging
^^^^^^^^^
- **My model doesn't get better. What can I do?**
   - *Synthetic data*: If you modified the code, it might help to inspect tensors and outputs manually for a synthetic task like the reverse task presented in the :ref:`tutorial`.
   - *Data*: If you're working with a standard model, doublecheck whether your data is properly aligned, properly pre-processed, properly filtered and whether the vocabularies cover a reasonable amount of tokens.
   - *Hyperparameters*: Try a smaller/larger/deeper/shallower model architecture with smaller/larger learning rates, different optimizers and turn off schedulers. It might be worth to try different initialization options. Train longer and validate less frequently, maybe training just takes longer than you'd expect.

- **My model takes too much memory. What can I do?**
   Consider reducing ``batch_size``. The mini-batch size can be virtually increased by a factor of *k* by setting ``batch_multiplier`` to *k*.
   Tensor operations are still performed with ``batch_size`` instances each, but model updates are done after *k* of these mini-batches.

- **My model is too slow. What can I do?**
- **My model stopped improving, but didn't stop training. What can I do?**
- **My model performs well on the validation set, but terrible on the test set. What's wrong?**
- **My model produces translations that are generally too short. What's wrong?**

Features
--------
- **Why is there no convolutional model?**
   We might add it in the future, but from our experience, the most popular models are recurrent and self-attentional.
- **How are the parameters initialized?**
   Check the description in `initialization.py <https://github.com/joeynmt/joeynmt/blob/master/joeynmt/initialization.py#L60>`_.
- **Is there the option to ensemble multiple models?**
   Not yet.
- **What is a bridge?**
   We call the connection between recurrent encoder and decoder states the *bridge*.
   This can either mean that the decoder states are initialized by copying the last (forward) encoder state (``init_hidden: "last"``),
   by learning a projection of the last encoder state (``init_hidden: "bridge"``) or simply zeros (``init_hidden: "zero"``).
- **Does learning rate scheduling matter?**

- **What is early stopping?**

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
- **How can I make my model multi-task?**
- **How can I feed my model multiple inputs?**
- **How can I add a regularizer to the loss?**

Contributing
------------
- **How can I contribute?**
- **What's in a Pull Request?**

Miscellaneous
-------------
- **Why should I use JoeyNMT rather than other NMT toolkits?**
- **I found a bug in your code, what should I do?**
- **How can I check whether my model is significantly better than my baseline model?**
- **Where can I find training data?**
