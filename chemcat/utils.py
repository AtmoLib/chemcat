# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'ROOT',
    'de_aliasing',
    'resolve_sources',
]

import os
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).parents[1]) + os.path.sep
import tea.janaf as janaf
import tea.cea as cea


def de_aliasing(input_species, source):
    """
    Get the right species names as given in the selected database.

    Parameters
    ----------
    input_species: List of strings
        List of species names.
    source: String
        The desired database source.

    Returns
    -------
    output_species: List of strings
        Species names with aliases replaced with the names
        as given in source database.

    Examples
    --------
    >>> import chemcat.utils as u
    >>> input_species = ['H2O', 'C2H2', 'HO2', 'CO']
    >>> source = 'janaf'
    >>> output_species = u.de_aliasing(input_species, source)
    >>> print(output_species)
    ['H2O', 'C2H2', 'HOO', 'CO']

    >>> source = 'cea'
    >>> output_species = u.de_aliasing(input_species, source)
    >>> print(output_species)
    ['H2O', 'C2H2,acetylene', 'HO2', 'CO']
    """
    # Get lists of species names aliases:
    whites = f'{ROOT}chemcat/data/white_pages.txt'
    aliases = []
    for line in open(whites, 'r'):
        if line.startswith('#'):
            continue
        aliases.append(line.split())
    all_aliases = np.concatenate(aliases)

    source_index = {
        'janaf': 0,
        'cea': 1,
    }

    output_species = []
    for species in input_species:
        if species not in all_aliases:
            output_species.append(species)
            continue
        for alias in aliases:
            if species in alias:
                output_species.append(alias[source_index[source]])
    return output_species


def resolve_sources(species, sources):
    r"""
    For each species in input, assign the right database proritizing
    by the order in sources.
    Parameters
    ----------
    species: 1D interable of strings
        Species to assign a database source.
    sources: 1D iterable of strings
        List of database sources in order of priority.
    Returns
    -------
    source_names: 1D array of strings
        Array with the assigned database to each species.
        If none found, leave value as None.
    Examples
    --------
    >>> import chemcat.utils as u
    >>> species = 'H2O CO (KOH)2 HO2'.split()
    >>> # Prioritize JANAF:
    >>> sources1 = u.resolve_sources(species, sources=['janaf', 'cea'])
    >>> # Prioritize CEA:
    >>> sources2 = u.resolve_sources(species, sources=['cea', 'janaf'])
    >>> # CEA exclusively:
    >>> sources3 = u.resolve_sources(species, sources=['cea'])
    >>> print(sources1, sources2, sources3, sep='\n')
    ['janaf' 'janaf' 'janaf' 'cea']
    ['cea' 'cea' 'janaf' 'cea']
    ['cea' 'cea' None 'cea']
    """
    if isinstance(sources, str):
        sources = [sources]

    nspecies = len(species)
    source_names = np.tile(None, nspecies)

    for source in sources:
        if source == 'janaf':
            in_janaf = janaf.is_in(species)
            is_missing = [name is None for name in source_names]
            source_names[is_missing & in_janaf] = 'janaf'

        elif source == 'cea':
            in_cea = cea.is_in(species)
            is_missing = [name is None for name in source_names]
            source_names[is_missing & in_cea] = 'cea'

    return source_names

