# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

import numpy as np
import pytest

import chemcat.utils as u


def test_stoich_matrix_neutrals():
    stoich_data = [
        {'H': 2.0, 'O': 1.0},
        {'C': 1.0, 'H': 4.0},
        {'C': 1.0, 'O': 2.0},
        {'H': 2.0},
        {'H': 1.0},
        {'He': 1.0},
    ]
    expected_stoich = np.array([
        [0, 2, 0, 1],
        [1, 4, 0, 0],
        [1, 0, 0, 2],
        [0, 2, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])

    elements, stoich_matrix = u.stoich_matrix(stoich_data)
    np.testing.assert_equal(elements, np.array(['C', 'H', 'He', 'O']))
    np.testing.assert_equal(stoich_matrix, expected_stoich)


def test_stoich_matrix_ions():
    stoich_data = [
        {'H': 2.0, 'O': 1.0},
        {'C': 1.0, 'H': 4.0},
        {'C': 1.0, 'O': 2.0},
        {'H': 2.0},
        {'H': 1.0, 'e': -1.0},
        {'He': 1.0},
        {'e': 1.0},
    ]
    expected_stoich = np.array([
        [ 0,  2,  0,  1,  0],
        [ 1,  4,  0,  0,  0],
        [ 1,  0,  0,  2,  0],
        [ 0,  2,  0,  0,  0],
        [ 0,  1,  0,  0, -1],
        [ 0,  0,  1,  0,  0],
        [ 0,  0,  0,  0,  1],
    ])

    elements, stoich_matrix = u.stoich_matrix(stoich_data)
    np.testing.assert_equal(elements, np.array(['C', 'H', 'He', 'O', 'e']))
    np.testing.assert_equal(stoich_matrix, expected_stoich)


@pytest.mark.parametrize('sources', ('janaf', ['janaf']))
def test_de_aliasing_janaf_only(sources):
    input_species = ['H2O', 'C2H2', 'HO2', 'CO']
    output_species = u.de_aliasing(input_species, sources)
    assert output_species == ['H2O', 'C2H2', 'HOO', 'CO']


@pytest.mark.parametrize('sources', ('cea', ['cea']))
def test_de_aliasing_cea_only(sources):
    input_species = ['H2O', 'C2H2', 'HO2', 'CO']
    output_species = u.de_aliasing(input_species, sources)
    assert output_species == ['H2O', 'C2H2,acetylene', 'HO2', 'CO']


def test_de_aliasing_janaf_cea():
    input_species = ['H2O', 'C2H2', 'HO2', 'CO']
    sources = ('janaf', 'cea')
    output_species = u.de_aliasing(input_species, sources)
    assert output_species == ['H2O', 'C2H2', 'HOO', 'CO']


def test_de_aliasing_not_found():
    input_species = ['H2O', 'C4H2', 'HO2', 'CO']
    sources = 'janaf'
    output_species = u.de_aliasing(input_species, sources)
    assert output_species == ['H2O', 'C4H2', 'HOO', 'CO']


def test_de_aliasing_default_cea():
    input_species = ['H2O', 'C4H2', 'HO2', 'CO']
    sources = ('janaf', 'cea')
    output_species = u.de_aliasing(input_species, sources)
    assert output_species == ['H2O', 'C4H2,butadiyne', 'HOO', 'CO']


def test_resolve_sources_with_missing_species():
    species = 'H2O CO (KOH)2 HO2'.split()
    sources = u.resolve_sources(species, sources=['cea'])
    assert list(sources) == ['cea', 'cea', None, 'cea']


def test_resolve_sources_cea_priority():
    species = 'H2O CO (KOH)2 HO2'.split()
    sources = u.resolve_sources(species, sources=['cea', 'janaf'])
    assert list(sources) == ['cea', 'cea', 'janaf', 'cea']


def test_resolve_sources_janaf_priority():
    species = 'H2O CO (KOH)2 HO2'.split()
    sources = u.resolve_sources(species, sources=['janaf', 'cea'])
    assert list(sources) == ['janaf', 'janaf', 'janaf', 'cea']


@pytest.mark.parametrize('sources', ('cea', ['cea']))
def test_resolve_sources_list_or_string(sources):
    species = 'H2O CO (KOH)2 HO2'.split()
    source_names = u.resolve_sources(species, sources=['cea'])
    assert list(source_names) == ['cea', 'cea', None, 'cea']

