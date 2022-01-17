.. _install:

============
Installation
============

Basics
------

First install `Python <https://www.python.org/>`_ >= 3.7, `PyTorch <https://pytorch.org/>`_ >=v.1.9.0 and `git <https://git-scm.com/>`_.

Create and activate a `virtual environment <https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments>`_ to install the package into:

.. code-block:: bash

   $ python3 -m venv jnmt
   $ source jnmt/bin/activate


Cloning
-------

Then clone JoeyNMT from GitHub and switch to its root directory:

.. code-block:: bash

   (jnmt)$ git clone https://github.com/joeynmt/joeynmt.git
   (jnmt)$ cd joeynmt


Installing JoeyNMT
------------------

Install JoeyNMT and it's requirements:

.. code-block:: bash

   (jnmt)$ pip3 install .

Run the unit tests to make sure your installation is working:

.. code-block:: bash

   (jnmt)$ python3 -m unittest

**Warning!** When running on *GPU* you need to manually install the suitable PyTorch version for your [CUDA](https://developer.nvidia.com/cuda-zone) version. This is described in the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

You're ready to go!
