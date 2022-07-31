Databases Tutorial
==================

constructing the right chemical network is not easy, particularly
finding which species exist in the thermodynamic databases is not
trivial (until now). This tutorial shows how to look up for all
available chemical species in the JANAF or CEA databases, searching by
atoms, atomic numbers, total number of atoms, and charge.  |br|
You can also find this tutorial as a `Python scrip here
<https://github.com/AtmoLib/chemcat/blob/main/docs/database_tutorial.py>`_
or as a `jupyter notebook here
<https://github.com/AtmoLib/chemcat/blob/main/docs/database_tutorial.ipynb>`_.

Let’s start off importing the necessary modules:

.. code:: python

    import chemcat.cea as cea
    import chemcat.janaf as janaf

Look up by Atom
---------------

To search for species containing specific atoms call the
``find_species()`` function with a list of the atomic species requested:

.. code:: python

    # Get all sodium-bearing species:
    janaf_salts = janaf.find_species(['Na'])
    print(f'JANAF salts with any amount of sodium:\n{janaf_salts}')
    
    cea_salts = cea.find_species(['Na'])
    print(f'\nCEA salts with any amount of sodium:\n{cea_salts}')


.. parsed-literal::

    JANAF salts with any amount of sodium:
    ['LiONa' 'Na2' 'Na2SO4' 'NaAlF4' 'NaBO2' '(NaBr)2' 'NaBr' '(NaCl)2' 'NaCl'
     '(NaCN)2' 'NaCN' '(NaF)2' 'NaF' 'Na' 'NaH' 'NaO' '(NaOH)2' 'NaOH']
    
    CEA salts with any amount of sodium:
    ['KNa' 'Na' 'NaCN' 'NaH' 'NaNO2' 'NaNO3' 'NaO' 'NaOH' 'Na2' 'Na2O' 'Na2O2'
     'Na2O2H2']


Multiple atomic species can be requested:

.. code:: python

    # Get all sodium- and oxigen-bearing species:
    janaf_sodium_oxide = janaf.find_species(['Na', 'O'])
    print(f'JANAF salts with any amount of sodium and oxygen:\n{janaf_sodium_oxide}')
    
    cea_sodium_oxide = cea.find_species(['Na', 'O'])
    print(f'\nCEA salts with any amount of sodium and oxigen:\n{cea_sodium_oxide}')


.. parsed-literal::

    JANAF salts with any amount of sodium and oxygen:
    ['LiONa' 'Na2SO4' 'NaBO2' 'NaO' '(NaOH)2' 'NaOH']
    
    CEA salts with any amount of sodium and oxigen:
    ['NaNO2' 'NaNO3' 'NaO' 'NaOH' 'Na2O' 'Na2O2' 'Na2O2H2']


Look up by Atomic Number
------------------------

To search for species containing specific atoms call the
``find_species()`` function with a dict of the atomic species requested
and their atomic values:

.. code:: python

    # Get species containing exactly two Na atoms:
    janaf_salts1 = janaf.find_species({'Na':1})
    print(f'JANAF salts with one sodium atom:\n{janaf_salts1}')
    
    cea_salts1 = cea.find_species({'Na':1})
    print(f'\nCEA salts with one sodium atom:\n{cea_salts1}')


.. parsed-literal::

    JANAF salts with one sodium atom:
    ['LiONa' 'NaAlF4' 'NaBO2' 'NaBr' 'NaCl' 'NaCN' 'NaF' 'Na' 'NaH' 'NaO'
     'NaOH']
    
    CEA salts with one sodium atom:
    ['KNa' 'Na' 'NaCN' 'NaH' 'NaNO2' 'NaNO3' 'NaO' 'NaOH']


Another example:

.. code:: python

    # Get species containing exactly two Na atoms:
    janaf_salts2 = janaf.find_species({'Na':2})
    print(f'JANAF salts with two sodium atoms:\n{janaf_salts2}')
    
    cea_salts2 = cea.find_species({'Na':2})
    print(f'\nCEA salts with two sodium atoms:\n{cea_salts2}')


.. parsed-literal::

    JANAF salts with two sodium atoms:
    ['Na2' 'Na2SO4' '(NaBr)2' '(NaCl)2' '(NaCN)2' '(NaF)2' '(NaOH)2']
    
    CEA salts with two sodium atoms:
    ['Na2' 'Na2O' 'Na2O2' 'Na2O2H2']


Use ``None`` as atomic number to request any non-zero amount of atoms
for a given species:

.. code:: python

    # Species containing exactly two carbon atoms and any amount of hydrogen:
    janaf_ethane = janaf.find_species({'H':None, 'C':2})
    print(f'JANAF ethane hydrocarbons:\n{janaf_ethane}')
    
    cea_ethane = cea.find_species({'H':None, 'C':2})
    print(f'\nCEA ethane hydrocarbons:\n{cea_ethane}')


.. parsed-literal::

    JANAF ethane hydrocarbons:
    ['C2H2' 'C2H4' 'C2H4O' 'C2HCl' 'C2HF' 'C2H']
    
    CEA ethane hydrocarbons:
    ['C2H' 'C2H2,acetylene' 'C2H2,vinylidene' 'CH2CO,ketene' 'O(CH)2O'
     'HO(CO)2OH' 'C2H3,vinyl' 'CH3CN' 'CH3CO,acetyl' 'C2H4' 'C2H4O,ethylen-o'
     'CH3CHO,ethanal' 'CH3COOH' 'OHCH2COOH' 'C2H5' 'C2H6' 'CH3N2CH3' 'C2H5OH'
     'CH3OCH3' 'CH3O2CH3' 'HCCN' 'HCCO' '(HCOOH)2']


Look up by Number of Total Atoms
--------------------------------

To search for species containing a specific number of atoms call the
``find_species()`` function with the ``num_atoms`` argument:

.. code:: python

    # Species containing exactly two atoms and at least one oxygen:
    janaf_diatomic_monoxides = janaf.find_species(['O'], num_atoms=2)
    print(f'JANAF diatomic monoxides (and O2):\n{janaf_diatomic_monoxides}')
    
    cea_diatomic_monoxides = cea.find_species(['O'], num_atoms=2)
    print(f'\nCEA diatomic monoxides (and O2):\n{cea_diatomic_monoxides}')


.. parsed-literal::

    JANAF diatomic monoxides (and O2):
    ['AlO' 'BaO' 'BeO' 'BO' 'CaO' 'ClO' 'CO' 'CrO' 'CsO' 'CuO' 'FeO' 'HgO'
     'KO' 'LiO' 'MgO' 'MoO' 'NaO' 'NbO' 'NO' 'O2' 'OD' 'OF' 'OH' 'PbO' 'PO'
     'SiO' 'SO' 'SrO' 'TaO' 'TiO' 'VO' 'WO' 'ZrO']
    
    CEA diatomic monoxides (and O2):
    ['CO' 'FeO' 'KO' 'NO' 'NaO' 'OD' 'OH' 'O2' 'AlO' 'MgO' 'SO' 'SiO' 'TiO'
     'VO']


Look up Ionic Species
---------------------

So far we have seen only neutral species, if we want to look up for
charged species we need to call the ``find_species()`` function with the
``charge`` argument:

.. code:: python

    # Look up sulfuric species in JANAF:
    janaf_sulfur = janaf.find_species(['S'], charge='all')
    print(f'All JANAF sulfuric species:\n{janaf_sulfur}')
    
    janaf_sulfur_neutral = janaf.find_species(['S'], charge='neutral')
    print(f'\nNeutral JANAF sulfuric species:\n{janaf_sulfur_neutral}')
    
    janaf_sulfur_ions = janaf.find_species(['S'], charge='ion')
    print(f'\nCharged JANAF sulfuric species:\n{janaf_sulfur_ions}')


.. parsed-literal::

    All JANAF sulfuric species:
    ['AlS' 'BaS' 'BeS' 'BS' 'CaS' 'CF3SF5' 'ClSSCl' 'COS' 'CS2' 'Cs2SO4' 'CS'
     'D2S' 'FeS' 'FSSF' 'H2S' 'HBS' 'HBS+' 'HS' 'HSO3F' 'K2SO4' 'Li2SO4' 'MgS'
     'Na2SO4' 'NiS' 'NS' 'O2S(OH)2' 'OSF2' 'P4S3' 'PbS' 'PSBr3' 'PSF3' 'PSF'
     'PS' 'S2Cl' 'S2F10' 'S2' 'S3' 'S4' 'S5' 'S6' 'S7' 'S8' 'SBrF5' 'SCl2'
     'SCl2+' 'SClF5' 'SCl' 'SCl+' 'SD' 'SF2' 'SF2-' 'SF2+' 'SF3' 'SF3-' 'SF3+'
     'SF4' 'SF4-' 'SF4+' 'SF5' 'SF5-' 'SF5+' 'SF6' 'SF6-' 'SF' 'SF-' 'SF+' 'S'
     'S-' 'S+' 'SiS' 'SO2Cl2' 'SO2ClF' 'SO2F2' 'SO2' 'SO3' 'SO' 'SPCl3' 'SrS'
     'SSF2' 'SSO']
    
    Neutral JANAF sulfuric species:
    ['AlS' 'BaS' 'BeS' 'BS' 'CaS' 'CF3SF5' 'ClSSCl' 'COS' 'CS2' 'Cs2SO4' 'CS'
     'D2S' 'FeS' 'FSSF' 'H2S' 'HBS' 'HS' 'HSO3F' 'K2SO4' 'Li2SO4' 'MgS'
     'Na2SO4' 'NiS' 'NS' 'O2S(OH)2' 'OSF2' 'P4S3' 'PbS' 'PSBr3' 'PSF3' 'PSF'
     'PS' 'S2Cl' 'S2F10' 'S2' 'S3' 'S4' 'S5' 'S6' 'S7' 'S8' 'SBrF5' 'SCl2'
     'SClF5' 'SCl' 'SD' 'SF2' 'SF3' 'SF4' 'SF5' 'SF6' 'SF' 'S' 'SiS' 'SO2Cl2'
     'SO2ClF' 'SO2F2' 'SO2' 'SO3' 'SO' 'SPCl3' 'SrS' 'SSF2' 'SSO']
    
    Charged JANAF sulfuric species:
    ['HBS+' 'SCl2+' 'SCl+' 'SF2-' 'SF2+' 'SF3-' 'SF3+' 'SF4-' 'SF4+' 'SF5-'
     'SF5+' 'SF6-' 'SF-' 'SF+' 'S-' 'S+']


Now the same look up, but for CEA:

.. code:: python

    # Look up sulfuric species in CEA:
    cea_sulfur = cea.find_species(['S'], charge='all')
    print(f'All CEA sulfuric species:\n{cea_sulfur}')
    
    cea_sulfur_neutral = cea.find_species(['S'], charge='neutral')
    print(f'\nNeutral CEA sulfuric species:\n{cea_sulfur_neutral}')
    
    cea_sulfur_ions = cea.find_species(['S'], charge='ion')
    print(f'\nCharged CEA sulfuric species:\n{cea_sulfur_ions}')


.. parsed-literal::

    All CEA sulfuric species:
    ['AlS' 'AlS2' 'Al2S' 'Al2S2' 'COS' 'CS' 'CS2' 'C2S2' 'C3OS' 'C3S2' 'D2S'
     'H2S' 'H2SO4' 'MgS' 'S' 'S+' 'S-' 'SD' 'SH' 'SH-' 'SN' 'SO' 'SO-' 'SO2'
     'SO2-' 'SO3' 'S2' 'S2-' 'S2O' 'S3' 'S4' 'S5' 'S6' 'S7' 'S8' 'SiS' 'SiS2']
    
    Neutral CEA sulfuric species:
    ['AlS' 'AlS2' 'Al2S' 'Al2S2' 'COS' 'CS' 'CS2' 'C2S2' 'C3OS' 'C3S2' 'D2S'
     'H2S' 'H2SO4' 'MgS' 'S' 'SD' 'SH' 'SN' 'SO' 'SO2' 'SO3' 'S2' 'S2O' 'S3'
     'S4' 'S5' 'S6' 'S7' 'S8' 'SiS' 'SiS2']
    
    Charged CEA sulfuric species:
    ['S+' 'S-' 'SH-' 'SO-' 'SO2-' 'S2-']


.. |br| raw:: html

   <br/>

