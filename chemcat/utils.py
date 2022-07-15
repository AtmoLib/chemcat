# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'ROOT',
    'de_aliasing',
]

import os
from pathlib import Path

import numpy as np


ROOT = str(Path(__file__).parents[1]) + os.path.sep


def de_aliasing(input_species, source):
    """
    Get the right name as in the database for the input species list.

    Parameters
    ----------
    input_species: List of strings
        List of species names.
    source: String
        The desired database source.

    Returns
    -------
    output_species: List of strings
        List of species names with aliases replaces with the names
        as given in source database.

    Examples
    --------
    >>> import chemcat.utils as u
    >>> input_species = 'H2O C2H2 HO2 CO'.split()
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

