# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

import os
from pathlib import Path
import sys

import numpy as np
import pytest

import chemcat as cat
import chemcat.janaf as janaf
from conftest import *


# Some default values:
nlayers = 11
net_temperature = np.tile(1200.0, nlayers)
net_pressure = np.logspace(-8, 3, nlayers)
net_molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()

element_file = f'{cat.ROOT}chemcat/data/asplund_2021_solar_abundances.dat'


def test_setup_janaf_network_missing_species():
    molecules = 'Ti Ti+ TiO TiO2 TiO+'.split()
    janaf_data = janaf.setup_network(molecules)

    expected_stoich_vals = np.array([
        [ 0,  1,  0],
        [ 0,  1, -1],
        [ 1,  1,  0],
        [ 2,  1,  0]
    ])

    assert len(janaf_data) == 5
    np.testing.assert_equal(janaf_data[0], ['Ti', 'Ti+', 'TiO', 'TiO2'])
    np.testing.assert_equal(janaf_data[1], ['O', 'Ti', 'e'])
    np.testing.assert_equal(janaf_data[4], expected_stoich_vals)


def test_thermo_eval_heat_capacity_single_temp():
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    janaf_data = janaf.setup_network(molecules)
    heat_capacity = janaf_data[2]
    temperature = 1500.0
    cp = cat.thermo_eval(temperature, heat_capacity)

    expected_cp = np.array([
        5.6636252 , 10.41029396,  4.23563153,  7.02137982,  8.00580904,
        4.19064967,  3.88455652,  6.65454913,  3.95900511,  2.49998117,
        2.49998117,  2.5033488 ,  2.49998117,  2.50707724])
    np.testing.assert_allclose(cp, expected_cp)


def test_thermo_eval_heat_capacity_temp_array():
    molecules = 'H2O CH4 CO C He'.split()
    janaf_data = janaf.setup_network(molecules)
    heat_capacity = janaf_data[2]
    temperatures = np.arange(100.0, 4501.0, 200.0)
    cp = cat.thermo_eval(temperatures, heat_capacity)

    np.testing.assert_allclose(cp, expected_cp)


def test_thermo_eval_gibbs_free_energy_temp_array():
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    janaf_data = janaf.setup_network(molecules)
    gibbs_funcs = janaf_data[3]
    temperatures = np.arange(100.0, 4101.0, 500.0)
    gibbs = cat.thermo_eval(temperatures, gibbs_funcs)

    np.testing.assert_allclose(gibbs, expected_gibbs)


def test_network_init():
    nlayers = 81
    temperature = np.tile(1200.0, nlayers)
    pressure = np.logspace(-8, 3, nlayers)
    molecules = 'H2O CH4 CO CO2 H2 H C O'.split()
    net = cat.Network(pressure, temperature, molecules)

    expected_stoich_vals = np.array([
        [0, 2, 1],
        [1, 4, 0],
        [1, 0, 1],
        [1, 0, 2],
        [0, 2, 0],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]])
    expected_element_rel_abundance = [2.88403150e-04, 1.0, 4.89778819e-04]
    expected_element_file = \
        f'{cat.ROOT}chemcat/data/asplund_2021_solar_abundances.dat'

    np.testing.assert_equal(net.pressure, pressure)
    np.testing.assert_equal(net.temperature, temperature)
    np.testing.assert_equal(net.input_species, molecules)
    assert net.metallicity == 0.0
    assert net.e_abundances == {}
    assert net.e_scale == {}
    assert net.e_ratio == {}
    assert net.element_file == expected_element_file

    np.testing.assert_equal(net.species, molecules)
    np.testing.assert_equal(net.elements, ['C', 'H', 'O'])
    np.testing.assert_equal(net.stoich_vals, expected_stoich_vals)
    np.testing.assert_allclose(
        net.element_rel_abundance, expected_element_rel_abundance,
    )


def test_network_cp_default_temp():
    net = cat.Network(net_pressure, net_temperature, net_molecules)
    cp = net.heat_capacity()

    expected_cp = np.array([
        5.26408044, 9.48143057, 4.11030773, 6.77638503, 7.34238673,
        4.05594463, 3.72748083, 6.3275286 , 3.79892261, 2.49998117,
        2.49998117, 2.50082308, 2.49998117, 2.51092596])

    assert np.shape(cp) == (len(net_pressure), len(net_molecules))
    np.testing.assert_allclose(cp[0], expected_cp)
    np.testing.assert_equal(net.temperature, net_temperature)


def test_network_cp_input_temp():
    net = cat.Network(net_pressure, net_temperature, net_molecules)
    temps = [100.0, 600.0, 1200.0]
    cp = net.heat_capacity(temps)

    expected_cp = np.array([
        [4.00494915, 4.3688933 , 5.26408044],
        [4.00001798, 6.28146429, 9.48143057],
        [3.50040662, 3.6614513 , 4.11030773],
        [3.51291495, 5.69140811, 6.77638503],
        [4.00314507, 5.44749578, 7.34238673],
        [3.50040662, 3.62140061, 4.05594463],
        [3.38614788, 3.52722736, 3.72748083],
        [3.50798378, 5.26865079, 6.3275286 ],
        [3.92412613, 3.55128183, 3.79892261],
        [2.49998117, 2.49998117, 2.49998117],
        [2.49998117, 2.49998117, 2.49998117],
        [2.55831326, 2.50154471, 2.50082308],
        [2.49998117, 2.49998117, 2.49998117],
        [2.85081563, 2.54063323, 2.51092596]]).T

    assert np.shape(cp) == (len(temps), len(net_molecules))
    np.testing.assert_allclose(cp, expected_cp)
    # Network's temperature is not updated:
    np.testing.assert_equal(net.temperature, net_temperature)


def test_network_gibbs_default_temp():
    net = cat.Network(net_pressure, net_temperature, net_molecules)
    gibbs = net.gibbs_free_energy()

    expected_gibbs = np.array([
        -49.70275118, -33.59076581, -37.17544326, -68.58699428,
        -31.08382889, -25.35365299, -17.97578591, -13.94856633,
        -20.48067821,   6.44982554, -16.77498672,  51.21052551,
         27.33544072,   3.95818325])

    assert np.shape(gibbs) == (len(net_temperature), len(net_molecules))
    np.testing.assert_allclose(gibbs[0], expected_gibbs)
    np.testing.assert_equal(net.temperature, net_temperature)


def test_network_vmr_default():
    net = cat.Network(net_pressure, net_temperature, net_molecules)
    vmr = net.thermochemical_equilibrium()

    assert np.shape(vmr) == (len(net_temperature), len(net_molecules))
    np.testing.assert_allclose(vmr, expected_vmr_1200K)
    np.testing.assert_allclose(net.vmr, expected_vmr_1200K)

    expected_e_abundance = np.array(
        [2.88403150e-04, 1.0, 8.20351544e-02, 6.76082975e-05, 4.89778819e-04]
    )
    np.testing.assert_allclose(net.element_rel_abundance, expected_e_abundance)

    assert np.all(vmr>=0)
    elem_fractions = np.sum(net.vmr[0]*net.stoich_vals.T, axis=1)
    elem_fractions /= elem_fractions[net.elements == 'H']
    np.testing.assert_allclose(elem_fractions, net.element_rel_abundance)


def test_network_vmr_update_temp():
    net = cat.Network(net_pressure, net_temperature, net_molecules)
    vmr = net.thermochemical_equilibrium()

    assert np.shape(vmr) == (len(net_temperature), len(net_molecules))
    np.testing.assert_allclose(vmr, expected_vmr_1200K)

    thot = np.tile(2900.0, nlayers)
    vmr = net.thermochemical_equilibrium(temperature=thot)
    assert np.shape(vmr) == (len(net_temperature), len(net_molecules))
    np.testing.assert_allclose(vmr, expected_vmr_2900K)
    np.testing.assert_equal(net.temperature, thot)



def test_network_vmr_update_metallicity():
    net = cat.Network(net_pressure, net_temperature, net_molecules)
    vmr = net.thermochemical_equilibrium(metallicity=0.0)
    np.testing.assert_allclose(vmr, expected_vmr_1200K)

    vmr = net.thermochemical_equilibrium(metallicity=1.0)
    expected_e_abundance = np.array(
        [2.88403150e-03, 1.0, 8.20351544e-02, 6.76082975e-04, 4.89778819e-03]
    )
    np.testing.assert_allclose(vmr, expected_vmr_10x_solar)
    np.testing.assert_equal(net.metallicity, 1.0)
    np.testing.assert_allclose(net.element_rel_abundance, expected_e_abundance)

    vmr = net.thermochemical_equilibrium()
    np.testing.assert_allclose(vmr, expected_vmr_10x_solar)


def test_network_vmr_update_e_abundances():
    net = cat.Network(net_pressure, net_temperature, net_molecules)
    e_abundances = {'C': 8.0}
    vmr = net.thermochemical_equilibrium(e_abundances=e_abundances)

    np.testing.assert_allclose(vmr, expected_vmr_e_abund)
    expected_e_abundance = np.array(
        [1.0e-04, 1.0, 8.20351544e-02, 6.76082975e-05, 4.89778819e-04]
    )
    np.testing.assert_allclose(net.element_rel_abundance, expected_e_abundance)


def test_network_vmr_update_e_scale():
    net = cat.Network(net_pressure, net_temperature, net_molecules)
    e_scale = {'C': 1.0}
    vmr = net.thermochemical_equilibrium(e_scale=e_scale)

    np.testing.assert_allclose(vmr, expected_vmr_e_scale)
    expected_e_abundance = np.array(
        [2.88403150e-03, 1.0, 8.20351544e-02, 6.76082975e-05, 4.89778819e-04]
    )
    np.testing.assert_allclose(net.element_rel_abundance, expected_e_abundance)


def test_network_vmr_update_e_ratio():
    net = cat.Network(net_pressure, net_temperature, net_molecules)
    e_ratio = {'C_O': np.log10(0.8)}
    vmr = net.thermochemical_equilibrium(e_ratio=e_ratio)

    np.testing.assert_allclose(vmr, expected_vmr_e_ratio)
    expected_e_abundance = np.array(
        [3.91823055e-04, 1.0, 8.20351544e-02, 6.76082975e-05, 4.89778819e-04]
    )
    np.testing.assert_allclose(net.element_rel_abundance, expected_e_abundance)
    C_to_O = (
        net.element_rel_abundance[net.elements=='C']
        / net.element_rel_abundance[net.elements=='O'])
    np.testing.assert_allclose(C_to_O, 0.8)


def test_network_vmr_update_metal_e_ratio():
    net = cat.Network(net_pressure, net_temperature, net_molecules)
    metallicity = 1.0
    e_ratio = {'C_O': np.log10(1.1)}
    vmr = net.thermochemical_equilibrium(
        metallicity=metallicity,
        e_ratio=e_ratio,
    )

    np.testing.assert_allclose(vmr, expected_vmr_metal_e_ratio)
    expected_e_abundance = np.array(
        [5.38756701e-03, 1.0, 8.20351544e-02, 6.76082975e-04, 4.89778819e-03]
    )
    np.testing.assert_allclose(
        net.element_rel_abundance, expected_e_abundance)
    C_to_O = (
        net.element_rel_abundance[net.elements=='C']
        / net.element_rel_abundance[net.elements=='O'])
    np.testing.assert_allclose(C_to_O, 1.1)


def test_network_vmr_update_temp_error():
    net = cat.Network(net_pressure, net_temperature, net_molecules)
    match = (
        'Temperature profile does not match size of '
        'pressure profile [(]11 layers[)]')
    bad_temp = np.tile(1200.0, nlayers-1)
    with pytest.raises(ValueError, match=match):
        net.thermochemical_equilibrium(bad_temp)


@pytest.mark.parametrize('sun',
    [
        'asplund_2009_solar_abundances.dat',
        'asplund_2021_solar_abundances.dat',
    ])
def test_read_elemental(sun):
    elements, dex = cat.read_elemental(f'{cat.ROOT}chemcat/data/{sun}')

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
    sun_elements, sun_dex = cat.read_elemental(element_file)
    elements = 'H He C N O'.split()
    e_abundances = cat.set_element_abundance(
        elements, sun_elements, sun_dex)
    expected_abundance = np.array([
        1.0, 8.20351544e-02, 2.88403150e-04, 6.76082975e-05, 4.89778819e-04,
    ])
    np.testing.assert_allclose(e_abundances, expected_abundance)

def test_set_element_abundance_metallicity():
    sun_elements, sun_dex = cat.read_elemental(element_file)
    elements = 'H He C N O'.split()
    e_abundances = cat.set_element_abundance(
        elements, sun_elements, sun_dex, metallicity=0.5)
    expected_abundance = np.array([
        1.0, 8.20351544e-02, 9.12010839e-04, 2.13796209e-04, 1.54881662e-03,
    ])
    np.testing.assert_allclose(e_abundances, expected_abundance)


def test_set_element_abundance_custom_element():
    sun_elements, sun_dex = cat.read_elemental(element_file)
    elements = 'H He C N O'.split()
    e_abundances = cat.set_element_abundance(
        elements, sun_elements, sun_dex, e_abundances={'C': 8.8})
    expected_abundance = np.array([
        1.0, 8.20351544e-02, 6.30957344e-04, 6.76082975e-05, 4.89778819e-04,
    ])
    np.testing.assert_allclose(e_abundances, expected_abundance)


def test_set_element_abundance_custom_e_scale():
    sun_elements, sun_dex = cat.read_elemental(element_file)
    elements = 'H He C N O'.split()
    e_abundances = cat.set_element_abundance(
        elements, sun_elements, sun_dex, e_scale={'C': np.log10(3.0)})
    expected_abundance = np.array([
        1.0, 8.20351544e-02, 8.65209451e-04, 6.76082975e-05, 4.89778819e-04,
    ])
    np.testing.assert_allclose(e_abundances, expected_abundance)


def test_set_element_abundance_custom_e_ratio():
    sun_elements, sun_dex = cat.read_elemental(element_file)
    elements = 'H He C N O'.split()
    e_abundances = cat.set_element_abundance(
        elements, sun_elements, sun_dex, e_ratio={'C_O': np.log10(0.6)})
    expected_abundance = np.array([
        1.0, 8.20351544e-02, 2.93867292e-04, 6.76082975e-05, 4.89778819e-04
    ])
    np.testing.assert_allclose(e_abundances, expected_abundance)

