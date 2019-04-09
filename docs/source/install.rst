============
Installation
============

Basics
------

First install `Python <https://www.python.org/>`_ >= 3.6, `PyTorch <https://pytorch.org/>`_ v.0.4.1 and `git <https://git-scm.com/>`_.

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


Requirements
------------

Install the python requirements with pip:

.. code-block:: bash

   (jnmt)$ pip install --upgrade -r requirements.txt


JoeyNMT
-------

Install JoeyNMT:

.. code-block:: bash

   (jnmt)$ python3 setup.py install

Run the unit tests to make sure your installation is working:

.. code-block:: bash

   (jnmt)$ python3 -m unittest


You're ready to go!