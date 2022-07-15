# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

import numpy as np
import pytest

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

