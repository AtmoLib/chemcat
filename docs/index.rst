.. chemcat documentation master file, created by
   sphinx-quickstart on Tue Dec 15 19:45:44 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. |br| raw:: html

   <br/>

chemcat: Chemistry Calculator for Atmospheres
=============================================

|Build Status|
|docs|
|PyPI|
|conda|
|License|

-------------------------------------------------------------------

:Author:        Jasmina Blecic and Patricio Cubillos (see :ref:`team`)
:Contact:       `jasmina[at]nyu.edu`_
:Contact:       `patricio.cubillos[at]oeaw.ac.at`_
:Organizations: `Space Research Institute (IWF) <http://iwf.oeaw.ac.at/>`_
:Web Site:      https://github.com/AtmoLib/chemcat
:Date:          |today|

-------------------------------------------------------------------


Features
========

``chemcat`` is a tool to compute atmospheric chemistry compositions
and other thermochemical properties.  The following features are
available when running ``chemcat``:

- Compute Thermochemical-equilibrium Chemistry via Gibbs Free-energy
  Optimization including charge conservation
- Set custom network of species in the system
- Incorporate NIST-JANAF and NASA-ThermoBuild databases
- Compute heat capacities for a mixture of gasses

.. _team:

Collaborators
=============

All of these people have made a direct or indirect contribution to
``chemcat``, and in many instances have been fundamental in the
development of this package.

- Jasmina Blecic (NYU-Abu Dhabi)
- `Patricio Cubillos <https://github.com/pcubillos>`_ (IWF) `patricio.cubillos[at]oeaw.ac.at`_

Documentation
=============

.. toctree::
   :maxdepth: 2

   get_started
   chemistry_tutorial
   database_tutorial
   api
   contributing
   license


Be Kind
=======

Please cite this paper if you found ``chemcat`` useful for your research:
  `Cubillos, Blecic, et al. (2024): Radiative and Thermochemical Equilibrium Calculations and Application for Warm-Jupiter Exoplanets. <http://ui.adsabs.harvard.edu/abs/2022journ.000....0C>`_

We welcome your feedback or inquiries, please refer them to:

  Patricio Cubillos (`patricio.cubillos[at]oeaw.ac.at`_) |br|
  Jasmina Blecic (`jasmina[at]nyu.edu`_)

``chemcat`` is open-source open-development software under the GPL-2.0
:ref:`license`. |br|
Thank you for using ``chemcat``!



.. _patricio.cubillos[at]oeaw.ac.at: patricio.cubillos@oeaw.ac.at
.. _jasmina[at]nyu.edu: jasmina@nyu.edu


.. |Build Status| image:: https://github.com/AtmoLib/chemcat/actions/workflows/python-package.yml/badge.svg
    :target: https://github.com/AtmoLib/chemcat/actions/workflows/python-package.yml

.. |docs| image:: https://readthedocs.org/projects/chemcat/badge/?version=latest
    :target: https://chemcat.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |PyPI| image:: https://img.shields.io/pypi/v/chemcat.svg
    :target:      https://pypi.org/project/chemcat/
    :alt: Latest Version

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/chemcat.svg
    :target: https://anaconda.org/conda-forge/chemcat


.. |License| image:: https://img.shields.io/github/license/atmolib/chemcat.svg?color=blue
    :target: https://chemcat.readthedocs.io/en/latest/license.html

