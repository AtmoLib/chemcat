# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

import os

import numpy as np
import pytest

import chemcat as cat
import chemcat.utils as u
import chemcat.janaf as janaf

from conftest import *


# Ddefault values:
element_file = f'{u.ROOT}chemcat/data/asplund_2021_solar_abundances.dat'


def test_thermo_eval_heat_capacity_single_temp():
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    janaf_data = janaf.setup_network(molecules)
    heat_capacity = janaf_data[1]
    temperature = 1500.0
    cp = u.thermo_eval(temperature, heat_capacity)

    expected_cp = np.array([
        5.6636252 , 10.41029396,  4.23563153,  7.02137982,  8.00580904,
        4.19064967,  3.88455652,  6.65454913,  3.95900511,  2.49998117,
        2.49998117,  2.5033488 ,  2.49998117,  2.50707724])
    np.testing.assert_allclose(cp, expected_cp)


def test_thermo_eval_heat_capacity_temp_array():
    molecules = 'H2O CH4 CO C He'.split()
    janaf_data = janaf.setup_network(molecules)
    heat_capacity = janaf_data[1]
    temperatures = np.arange(100.0, 4501.0, 200.0)
    cp = u.thermo_eval(temperatures, heat_capacity)

    np.testing.assert_allclose(cp, expected_cp)


def test_thermo_eval_gibbs_free_energy_temp_array():
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    janaf_data = janaf.setup_network(molecules)
    gibbs_funcs = janaf_data[2]
    temperatures = np.arange(100.0, 4101.0, 500.0)
    gibbs = u.thermo_eval(temperatures, gibbs_funcs)

    np.testing.assert_allclose(gibbs, expected_gibbs)


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
        elements, sun_elements, sun_dex, e_ratio={'C_O': 0.6},
    )
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


def test_resolve_colors_defaults():
    species = 'H He C H2 CH4 CO CO2'.split()
    colors = u.resolve_colors(species)
    assert colors['H'] == 'blue'
    assert colors['He'] == 'olive'
    assert colors['C'] == 'coral'
    assert colors['H2'] == 'deepskyblue'
    assert colors['CH4'] == 'darkorange'
    assert colors['CO'] == 'limegreen'
    assert colors['CO2'] == 'red'


def test_resolve_colors_ions():
    species = 'H He C H2 CH4 CO CO2 e- H+ H- H3+'.split()
    colors = u.resolve_colors(species)
    # Ions not really counted as extra species-color pairs:
    assert 'H+' not in colors
    assert 'H-' not in colors
    # But neutral version of ions exist indeed:
    assert colors['e'] == 'darkgreen'
    assert colors['H3'] == 'royalblue'

    assert colors['H'] == 'blue'
    assert colors['He'] == 'olive'
    assert colors['C'] == 'coral'
    assert colors['H2'] == 'deepskyblue'
    assert colors['CH4'] == 'darkorange'
    assert colors['CO'] == 'limegreen'
    assert colors['CO2'] == 'red'


def test_resolve_colors_custom_list():
    species = 'H He C H2 CH4 CO CO2'.split()
    color_list = 'red green blue black orange brown magenta'.split()
    colors = u.resolve_colors(species, color_list=color_list)
    # If only color_list is input, do not default color_dict:
    assert colors['H'] == 'red'
    assert colors['He'] == 'green'
    assert colors['C'] == 'blue'
    assert colors['H2'] == 'black'
    assert colors['CH4'] == 'orange'
    assert colors['CO'] == 'brown'
    assert colors['CO2'] == 'magenta'


def test_resolve_colors_more_species_than_colors():
    species = 'H He C H2 CH4 CO CO2'.split()
    color_list = 'red green blue black'.split()
    colors = u.resolve_colors(species, color_list=color_list)
    assert colors['H'] == 'red'
    assert colors['He'] == 'green'
    assert colors['C'] == 'blue'
    assert colors['H2'] == 'black'
    # Colors cycle over remaining species:
    assert colors['CH4'] == 'red'
    assert colors['CO'] == 'green'
    assert colors['CO2'] == 'blue'


def test_resolve_colors_not_all_colors_in_dict():
    species = 'H He C H2 CH4 CO CO2'.split()
    color_dict = {
        'H': 'red',
        'He': 'green',
        'H2': 'blue',
    }
    colors = u.resolve_colors(species, color_dict=color_dict)
    # Colors as in input color_dict:
    assert colors['H'] == 'red'
    assert colors['He'] == 'green'
    assert colors['H2'] == 'blue'
    # Remaining species get assigned a color by default from u.COLORS:
    assert colors['C'] == 'royalblue'
    assert colors['CH4'] == 'darkorange'
    assert colors['CO'] == 'darkgreen'
    assert colors['CO2'] == 'magenta'


@pytest.mark.skip(reason='TBI')
def test_plot_vmr():
    nlayers = 81
    temp = 1700.0
    temperature = np.tile(temp, nlayers)
    pressure = np.logspace(-10, 3, nlayers)

    species = (
        'H2O CH4 CO CO2 NH3 N2 H2 HCN C2H2 C2H4 OH H He C N O'.split()
        + 'e- H- H+ H2+ He+'.split()
        + 'Na Na- Na+ K K- K+'.split()
        #+ 'Mg Mg+ Fe Fe+'.split()
        #+ 'Ti TiO TiO2 Ti+ TiO+ V VO VO2 V+'.split()
        + 'H3O+'.split()
    )

    net = cat.Network(pressure, temperature, species)
    vmr = net.thermochemical_equilibrium()
    species = net.species
    ax = u.plot_vmr(pressure, vmr, species, vmr_range=(1e-30, 3))


