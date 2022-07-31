#!/usr/bin/env python
# coding: utf-8

# Databases Tutorial
# This tutorial shows how to compute chemistry with ``chemcat`` under different assumptions for temperature, metallicity, and custom elemental abundances.
# 
# Let's start off importing the necessary modules and create an utility function to plot the results:


import chemcat.cea as cea
import chemcat.janaf as janaf


## Look up by Atom
# To search for species containing specific atoms call the `find_species()` function with a list of the atomic species requested:

# Get all sodium-bearing species:
janaf_salts = janaf.find_species(['Na'])
print(f'JANAF salts with any amount of sodium:\n{janaf_salts}')

cea_salts = cea.find_species(['Na'])
print(f'\nCEA salts with any amount of sodium:\n{cea_salts}')


# Get all sodium- and oxigen-bearing species:
janaf_sodium_oxide = janaf.find_species(['Na', 'O'])
print(
    'JANAF salts with any amount of sodium and oxygen:'
    f'\n{janaf_sodium_oxide}'
)

cea_sodium_oxide = cea.find_species(['Na', 'O'])
print(
    '\nCEA salts with any amount of sodium and oxigen:'
    f'\n{cea_sodium_oxide}'
)


## Look up by Atomic Number
# To search for species containing specific atoms call the `find_species()` function with a dict of the atomic species requested and their atomic values:

# Get species containing exactly two Na atoms:
janaf_salts1 = janaf.find_species({'Na':1})
print(f'JANAF salts with one sodium atom:\n{janaf_salts1}')

cea_salts1 = cea.find_species({'Na':1})
print(f'\nCEA salts with one sodium atom:\n{cea_salts1}')


# Get species containing exactly two Na atoms:
janaf_salts2 = janaf.find_species({'Na':2})
print(f'JANAF salts with two sodium atoms:\n{janaf_salts2}')

cea_salts2 = cea.find_species({'Na':2})
print(f'\nCEA salts with two sodium atoms:\n{cea_salts2}')


# Use `None` as atomic number to request any non-zero amount of atoms for a given species:

# Species containing exactly two carbon atoms and any amount of hydrogen:
janaf_ethane = janaf.find_species({'H':None, 'C':2})
print(f'JANAF ethane hydrocarbons:\n{janaf_ethane}')

cea_ethane = cea.find_species({'H':None, 'C':2})
print(f'\nCEA ethane hydrocarbons:\n{cea_ethane}')


## Look up by Number of Total Atoms
# To search for species containing a specific number of atoms call the `find_species()` function with the `num_atoms` argument:

# Species containing exactly two atoms and at least one oxygen:
janaf_diatomic_monoxides = janaf.find_species(['O'], num_atoms=2)
print(f'JANAF diatomic monoxides (and O2):\n{janaf_diatomic_monoxides}')

cea_diatomic_monoxides = cea.find_species(['O'], num_atoms=2)
print(f'\nCEA diatomic monoxides (and O2):\n{cea_diatomic_monoxides}')


## Look up Ionic Species
# So far we have seen only neutral species, if we want to look up for charged species we need to call the `find_species()` function with the `charge` argument:

# Look up sulfuric species in JANAF:
janaf_sulfur = janaf.find_species(['S'], charge='all')
print(f'All JANAF sulfuric species:\n{janaf_sulfur}')

janaf_sulfur_neutral = janaf.find_species(['S'], charge='neutral')
print(f'\nNeutral JANAF sulfuric species:\n{janaf_sulfur_neutral}')

janaf_sulfur_ions = janaf.find_species(['S'], charge='ion')
print(f'\nCharged JANAF sulfuric species:\n{janaf_sulfur_ions}')


# Look up sulfuric species in CEA:
cea_sulfur = cea.find_species(['S'], charge='all')
print(f'All CEA sulfuric species:\n{cea_sulfur}')

cea_sulfur_neutral = cea.find_species(['S'], charge='neutral')
print(f'\nNeutral CEA sulfuric species:\n{cea_sulfur_neutral}')

cea_sulfur_ions = cea.find_species(['S'], charge='ion')
print(f'\nCharged CEA sulfuric species:\n{cea_sulfur_ions}')

