.. _getstarted:

Getting Started
===============

The ``chemcat`` package offers a simplified object-oriented
framework to compute atmospheric chemical compositions in
thermochemical equilibrium.  The code enables multiple way to
parameterize the atmospheric composition, including:

* Scaling the abundance of all metal elements (everything except H and
  He) relative to the solar metallicity.
* Setting the abundance of individual/custom elements.
* Setting the abundance of individual/custom elements as elemental
  ratios (e.g., C/O, Na/H, etc.).

``chemcat`` also provides an interface to access the NIST-JANAF and
NASA-ThermoBuild databases of thermochemical properties (see section
:ref:`tutorial_databases`).

Take a look at the :ref:`quick_example` section to get up to speed in
computing ``chemcat``.  The `Chemistry Tutorial <./chemistry_tutorial.ipynb>`_ section showcases the
multiple ways in which ``chemcat`` enables a parameterization of the
atmospheric composition.


System Requirements
-------------------

``chemcat`` is compatible with **Python3.6+** and has been `tested
<https://github.com/AtmoLib/chemcat/actions/workflows/python-package.yml?query=branch%3Amain>`_
to work in both Linux and OS X.  On installation (see below), the code
will automatically install the following required Python software:

* numpy (version 1.19.1+)
* scipy (version 1.5.2+)
* matplotlib (version 3.3.4+)
* more-itertools (version 8.4.0+)

.. _install:

Install
-------

To install ``bibmanager`` run the following command from the terminal:

.. code-block:: shell

    pip install chemcat

Or if you prefer conda:

.. code-block:: shell

    conda install -c conda-forge chemcat

Alternatively (e.g., for developers), clone the repository to your local machine with the following terminal commands:

.. code-block:: shell

  git clone https://github.com/atmolib/chemcat
  cd chemcat
  pip install -e .



.. _quick_example:

Quick Example
-------------


Once installed, ``chemcat`` is ready to use, for example, from the
Python Interpreter.

This example shows how to compute volume mixing ratios in
thermochemical equilibrium for an isothermal atmosphere (at
:math:`T=1200` K) between :math:`10^{-8}` and :math:`10^{2}` bars:

.. code-block:: python

    import chemcat as cat

    nlayers = 81
    temperature = np.tile(1200.0, nlayers)
    pressure = np.logspace(-8, 3, nlayers)
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH C2H2 C2H4 H He C N O'.split()

    net = cat.Network(pressure, temperature, molecules)
    vmr = net.thermochemical_equilibrium()

See section `Chemistry Tutorial <./chemistry_tutorial.ipynb>`_ for an in-depth tutorial of the
``chemcat`` capabilities.
Additionally, all low-and mid-level routines can be found in the
package's :ref:`API`.
