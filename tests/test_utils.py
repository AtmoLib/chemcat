# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

import numpy as np

import chemcat.utils as u


def test_de_aliasing_janaf():
    input_species = ['H2O', 'C2H2', 'HO2', 'CO']
    source = 'janaf'
    output_species = u.de_aliasing(input_species, source)
    assert output_species == ['H2O', 'C2H2', 'HOO', 'CO']


def test_de_aliasing_cea():
    input_species = ['H2O', 'C2H2', 'HO2', 'CO']
    source = 'cea'
    output_species = u.de_aliasing(input_species, source)
    assert output_species == ['H2O', 'C2H2,acetylene', 'HO2', 'CO']

