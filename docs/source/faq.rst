==========================
Frequently Asked Questions
==========================

Usage
-----
- **Which default hyperparameters should I use?**
  There is no universal answer to this question. We recommend you to check publications that used the same data as you're using (or at least the same language pair and data size)
  and find out how large their models where, how long they trained them etc.
  You might also get inspiration from the benchmarks that we report. Their configuration files can be found in the ``configs`` directory.
- **How can I train the model on GPU/CPU?**
  Set the ``use_cuda`` flag in the configuration to True for training on GPU (requires CUDA) or to False for training on CPU.
- **Which hyperparameters should I change first?**
- **How can I start Tensorboard for a model that I trained on a remote server?**
- **How can I perform domain adaptation?**
- **Where can I find the default values for the settings in the configuration file?**
  Either check `the default configuration file <https://github.com/joeynmt/joeynmt/blob/master/configs/default.yaml>`_ or :ref:`the API documentation <api>` for individual modules.
  Please note that there is no guarantee that the default setting is a good setting.
- **What happens if I made a mistake when configuring my model?**
  JoeyNMT will complain by raising a ``ConfigurationError`.
- **How can I stop training?**
  Simply press Control+C.
- **How often should I validate?**
  Depends on the size of your data. Please
- **Does JoeyNMT pre-process my data?**
  JoeyNMT does *not* include any pre-processing like tokenization, filtering by length ratio, normalization or learning/applying of BPEs.
  For that purpose, you might find the tools provided by the Moses decoder useful, as well as the subwordnmt library for BPEs.
  However, the training data gets *filtered* by the ``max_sent_length`` (keeping all training instances where source and target are up to that length)
  that you specify in the data section of the configuration file.
- **How many parameters has my model?**
   The number of parameters is logged in the training log file. You can find it in the model directory in ``train.log``.
- **How can I see how well my model is doing?**
- **What's the influence of the random seed?**
- **How can I check whether my model is significantly better than my baseline model?**

Debugging
---------
- **My model doesn't get better. What can I do?**
- **My model takes too much memory. What can I do?**
- **My model is too slow. What can I do?**
- **My model stopped improving, but didn't stop training. What can I do?**
- **My model performs well on the validation set, but terrible on the test set. What's wrong?**
- **My model produces translations that are generally too short. What's wrong?**

Features
--------
- **Why is there no convolutional model?**
  We might add it in the future, but from our experience, the most popular models are recurrent and self-attentional.
- **How are the parameters initialized?**
  Check the description in `initialization.py <https://github.com/joeynmt/joeynmt/blob/master/joeynmt/initialization.py#L60>_`.
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
- **What's the difference between ``max_sent_length`` and and ``max_output_length``?**
   ``max_sent_length`` determines the maximum source and target length of the training data,
   ``max_output_length`` is the maximum length of the translations that your model will be asked to produce.

Model Extensions
----------------
- **How can I make my model multi-task?**
- **How can I feed my model multiple inputs?**
- **How can I add a regularizer to the loss?**