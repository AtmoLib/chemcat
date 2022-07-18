# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

import os

import numpy as np
import pytest

import chemcat as cat
import chemcat.utils as u

from conftest import *


# Ddefault values:
element_file = f'{u.ROOT}chemcat/data/asplund_2021_solar_abundances.dat'


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


@pytest.mark.parametrize('sun',
    [
        'asplund_2009_solar_abundances.dat',
        'asplund_2021_solar_abundances.dat',
    ])
def test_read_elemental(sun):
    elements, dex = u.read_elemental(f'{u.ROOT}chemcat/data/{sun}')

    expected_elements_asplund = (
        'D   H   He  Li  Be  B   C   N   O   F   Ne  Na '
        'Mg  Al  Si  P   S   Cl  Ar  K   Ca  Sc  Ti  V '
        'Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se '
        'Br  Kr  Rb  Sr  Y   Zr  Nb  Mo  Ru  Rh  Pd '
        'Ag  Cd  In  Sn  Sb  Te  I   Xe  Cs  Ba  La '
        'Ce  Pr  Nd  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm '
        'Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg '
        'Tl  Pb  Bi  Th  U').split()
    if '2009' in sun:
        expected_dex_asplund = expected_dex_asplund_2009
    elif '2021' in sun:
        expected_dex_asplund = expected_dex_asplund_2021

    assert len(elements) == 84
    for element, expected_element in zip(elements, expected_elements_asplund):
        assert element == expected_element
    np.testing.assert_allclose(dex[dex>0], expected_dex_asplund)


def test_set_element_abundance_solar():
    sun_elements, sun_dex = u.read_elemental(element_file)
    elements = 'H He C N O'.split()
    e_abundances = u.set_element_abundance(
        elements, sun_elements, sun_dex)
    expected_abundance = np.array([
        1.0, 8.20351544e-02, 2.88403150e-04, 6.76082975e-05, 4.89778819e-04,
    ])
    np.testing.assert_allclose(e_abundances, expected_abundance)

def test_set_element_abundance_metallicity():
    sun_elements, sun_dex = u.read_elemental(element_file)
    elements = 'H He C N O'.split()
    e_abundances = u.set_element_abundance(
        elements, sun_elements, sun_dex, metallicity=0.5)
    expected_abundance = np.array([
        1.0, 8.20351544e-02, 9.12010839e-04, 2.13796209e-04, 1.54881662e-03,
    ])
    np.testing.assert_allclose(e_abundances, expected_abundance)


def test_set_element_abundance_custom_element():
    sun_elements, sun_dex = u.read_elemental(element_file)
    elements = 'H He C N O'.split()
    e_abundances = u.set_element_abundance(
        elements, sun_elements, sun_dex, e_abundances={'C': 8.8})
    expected_abundance = np.array([
        1.0, 8.20351544e-02, 6.30957344e-04, 6.76082975e-05, 4.89778819e-04,
    ])
    np.testing.assert_allclose(e_abundances, expected_abundance)


def test_set_element_abundance_custom_e_scale():
    sun_elements, sun_dex = u.read_elemental(element_file)
    elements = 'H He C N O'.split()
    e_abundances = u.set_element_abundance(
        elements, sun_elements, sun_dex, e_scale={'C': np.log10(3.0)})
    expected_abundance = np.array([
        1.0, 8.20351544e-02, 8.65209451e-04, 6.76082975e-05, 4.89778819e-04,
    ])
    np.testing.assert_allclose(e_abundances, expected_abundance)


def test_set_element_abundance_custom_e_ratio():
    sun_elements, sun_dex = u.read_elemental(element_file)
    elements = 'H He C N O'.split()
    e_abundances = u.set_element_abundance(
        elements, sun_elements, sun_dex, e_ratio={'C_O': np.log10(0.6)})
    expected_abundance = np.array([
        1.0, 8.20351544e-02, 2.93867292e-04, 6.76082975e-05, 4.89778819e-04
    ])
    np.testing.assert_allclose(e_abundances, expected_abundance)


def test_write_file_read_file(tmpdir):
    atm_file = "atmfile.dat"
    atm = f'{tmpdir}/{atm_file}'

    nlayers = 11
    temperature = np.tile(1200.0, nlayers)
    pressure = np.logspace(-8, 3, nlayers)
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    net = cat.Network(pressure, temperature, molecules)
    vmr = net.thermochemical_equilibrium()

    u.write_file(atm, net.species, pressure, temperature, vmr)
    assert atm_file in os.listdir(str(tmpdir))

    # Now, open file and check values:
    species, pressure, temperature, read_vmr = u.read_file(atm)
    np.testing.assert_equal(species, molecules)
    np.testing.assert_allclose(pressure, pressure)
    np.testing.assert_allclose(temperature, temperature)
    np.testing.assert_allclose(read_vmr, vmr)



def test_write_file_read_file_tiny_abundances(tmpdir):
    atm_file = "atmfile.dat"
    atm = f'{tmpdir}/{atm_file}'

    nlayers = 11
    temperature = np.tile(1200.0, nlayers)
    pressure = np.logspace(-8, 3, nlayers)
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    net = cat.Network(pressure, temperature, molecules)
    vmr = net.thermochemical_equilibrium()
    # Very small values change the print format:
    vmr[0:0] = 1.0e-200
    vmr[:,3] = 2.0e-200
    vmr[3,:] = 3.0e-200

    u.write_file(atm, net.species, pressure, temperature, vmr)
    assert atm_file in os.listdir(str(tmpdir))
    # Now, open file and check values:
    species, pressure, temperature, read_vmr = u.read_file(atm)
    np.testing.assert_allclose(pressure, pressure)
    np.testing.assert_allclose(temperature, temperature)
    np.testing.assert_equal(species, molecules)
    np.testing.assert_allclose(read_vmr, vmr)


