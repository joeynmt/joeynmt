.. _install:

============
Installation
============

Basics
------

First install `Python <https://www.python.org/>`_ >= 3.10, `PyTorch <https://pytorch.org/>`_ >=v.2.0.0 and `git <https://git-scm.com/>`_.

Create and activate a `virtual environment <https://packaging.python.org/tutorials/installing-packages/#creating-virtual-environments>`_ to install the package into:

.. code-block:: bash

   $ python -m venv jnmt
   $ source jnmt/bin/activate


Cloning
-------

Then clone JoeyNMT from GitHub and switch to its root directory:

.. code-block:: bash

   (jnmt)$ git clone https://github.com/joeynmt/joeynmt.git
   (jnmt)$ cd joeynmt

.. note::
    For Windows users, we recommend to doublecheck whether txt files (i.e. `test/data/toy/*`) have utf-8 encoding.


Installing JoeyNMT
------------------

Install JoeyNMT and its requirements:

.. code-block:: bash

   (jnmt)$ python -m pip install -e .

Run the unit tests to make sure your installation is working:

.. code-block:: bash

   (jnmt)$ python -m unittest

.. warning::

    When running on *GPU* you need to manually install the suitable PyTorch version for your `CUDA <https://developer.nvidia.com/cuda-zone>`_ version. This is described in the `PyTorch installation instructions <https://pytorch.org/get-started/locally/>`_.

You're ready to go!
