# Copyright (c) 2022-2024 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

import numpy as np
import pytest

import chemcat.cea as cea



def test_is_in():
    species = 'H2O (KOH)2 HO2 CO'.split()
    in_cea = cea.is_in(species)
    np.testing.assert_equal(in_cea, np.array([True, False, True, True]))


def test_read_thermo_build():
    species = ['H2O']
    cea_data = cea.read_thermo_build(species)

    expected_t_coeffs = np.array([200.0, 1000.0, 6000.0, 0.0])
    expected_a_coeffs = np.array([
        [-3.94796083e+04,  5.75573102e+02,  9.31782653e-01,
          7.22271286e-03, -7.34255737e-06,  4.95504349e-09,
         -1.33693325e-12],
        [ 1.03497210e+06, -2.41269856e+03,  4.64611078e+00,
          2.29199831e-03, -6.83683048e-07,  9.42646893e-11,
         -4.82238053e-15],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00]
    ])
    expected_b_coeffs = np.array([
        [-3.30397431e+04,  1.72420578e+01],
        [-1.38428651e+04, -7.97814851e+00],
        [ 0.00000000e+00,  0.00000000e+00]
    ])

    assert len(cea_data) == 1
    data = cea_data[0]
    assert data['name'] == 'H2O'
    assert data['stoich'] == {'H':2.0, 'O':1.0}

    assert np.shape(data['t_coeffs']) == (4,)
    assert np.shape(data['a_coeffs']) == (3,7)
    assert np.shape(data['b_coeffs']) == (3,2)
    np.testing.assert_allclose(data['t_coeffs'], expected_t_coeffs)
    np.testing.assert_allclose(data['a_coeffs'], expected_a_coeffs)
    np.testing.assert_allclose(data['b_coeffs'], expected_b_coeffs)


def test_read_thermo_build_stoich_neutral():
    species = 'H H2 H2O'.split()
    cea_data = cea.read_thermo_build(species)

    stoich = cea_data[0]['stoich']
    assert len(stoich) == 1
    assert stoich['H'] == 1.0

    stoich = cea_data[1]['stoich']
    assert len(stoich) == 1
    assert stoich['H'] == 2.0

    stoich = cea_data[2]['stoich']
    assert len(stoich) == 2
    assert stoich['H'] == 2.0
    assert stoich['O'] == 1.0


def test_read_thermo_cea_stoich_ions():
    species = 'e- H- H+ H3O+'.split()
    cea_data = cea.read_thermo_build(species)

    stoich = cea_data[0]['stoich']
    assert len(stoich) == 1
    assert stoich['e'] == 1.0

    stoich = cea_data[1]['stoich']
    assert len(stoich) == 2
    assert stoich['H'] == 1.0
    assert stoich['e'] == 1.0

    stoich = cea_data[2]['stoich']
    assert len(stoich) == 2
    assert stoich['H'] == 1.0
    assert stoich['e'] == -1.0

    stoich = cea_data[3]['stoich']
    assert len(stoich) == 3
    assert stoich['H'] == 3.0
    assert stoich['O'] == 1.0
    assert stoich['e'] == -1.0


@pytest.mark.parametrize(
    'temp',
    [300, 300.0],
)
def test_heat_func_cea_single_value(temp):
    data = cea.read_thermo_build(['H2O'])[0]
    heat = cea.heat_func(data['a_coeffs'], data['t_coeffs'])
    np.testing.assert_allclose(heat(temp), np.array([4.04063805]))


@pytest.mark.parametrize(
    'temp',
    ([300, 1000.0, 3000.0], np.array([300, 1000.0, 3000.0]))
)
def test_heat_func_cea_array(temp):
    data = cea.read_thermo_build(['H2O'])[0]
    heat = cea.heat_func(data['a_coeffs'], data['t_coeffs'])
    expected_heat = np.array([4.04063805, 4.96614188, 6.8342561])
    np.testing.assert_allclose(heat(temp), expected_heat)


def test_setup_network_cea_neutrals():
    molecules = 'H2O CH4 CO2 H2 H He'.split()
    cea_data = cea.setup_network(molecules)

    expected_stoich = [
        {'H': 2.0, 'O': 1.0},
        {'C': 1.0, 'H': 4.0},
        {'C': 1.0, 'O': 2.0},
        {'H': 2.0},
        {'H': 1.0},
        {'He': 1.0},
    ]

    assert len(cea_data) == 4
    np.testing.assert_equal(cea_data[0], molecules)
    np.testing.assert_equal(cea_data[3], expected_stoich)


def test_setup_network_cea_ions():
    molecules = 'H2O CH4 CO2 H2 H He e- H- H2+'.split()
    cea_data = cea.setup_network(molecules)

    expected_stoich = [
        {'H': 2.0, 'O': 1.0},
        {'C': 1.0, 'H': 4.0},
        {'C': 1.0, 'O': 2.0},
        {'H': 2.0},
        {'H': 1.0},
        {'He': 1.0},
        {'e': 1.0},
        {'H': 1.0, 'e': 1.0},
        {'H': 2.0, 'e': -1.0},
    ]

    np.testing.assert_equal(cea_data[0], molecules)
    np.testing.assert_equal(cea_data[3], expected_stoich)


def test_setup_network_cea_missing_species():
    molecules = 'Ti Ti+ TiO TiO2+ TiO2'.split()
    cea_data = cea.setup_network(molecules)

    expected_stoich= [
        {'Ti': 1.0},
        {'Ti': 1.0, 'e': -1.0},
        {'Ti': 1.0, 'O': 1.0},
        {'Ti': 1.0, 'O': 2.0},
    ]

    np.testing.assert_equal(cea_data[0], ['Ti', 'Ti+', 'TiO', 'TiO2'])
    np.testing.assert_equal(cea_data[3], expected_stoich)


def test_find_species_single():
    specs = cea.find_species(['K'])
    assert list(specs) == [
       'K', 'KCN', 'KH', 'KNO2', 'KNO3', 'KNa', 'KO', 'KOH', 'K2',
       'K2CO3', 'K2C2N2', 'K2O', 'K2O2', 'K2O2H2']


def test_find_species_dict_without_stoich_values():
    specs = cea.find_species({'K':None})
    assert list(specs) == [
       'K', 'KCN', 'KH', 'KNO2', 'KNO3', 'KNa', 'KO', 'KOH', 'K2',
       'K2CO3', 'K2C2N2', 'K2O', 'K2O2', 'K2O2H2']


def test_find_species_with_stoich_values():
    specs = cea.find_species({'K':2})
    assert list(specs) == ['K2', 'K2CO3', 'K2C2N2', 'K2O', 'K2O2', 'K2O2H2']


def test_find_species_multiple():
    specs = cea.find_species('N O'.split())
    assert list(specs) == [
        'OCCN', 'CNCOCN', 'HNCO', 'HNO', 'HNO2', 'HNO3', 'KNO2', 'KNO3',
        'NCO', 'NH2OH', 'NO', 'NO2', 'NO3', 'N2O', 'NH2NO2', 'N2O3',
        'N2O4', 'N2O5', 'NaNO2', 'NaNO3']


def test_find_species_ions():
    specs = cea.find_species('N O'.split(), charge='ion')
    assert list(specs) == ['NO+', 'NO2-', 'NO3-', 'N2O+']


def test_find_species_neutral_and_ions():
    specs = cea.find_species('N O'.split(), charge='all')
    assert list(specs) == [
        'OCCN', 'CNCOCN', 'HNCO', 'HNO', 'HNO2', 'HNO3', 'KNO2', 'KNO3',
        'NCO', 'NH2OH', 'NO', 'NO+', 'NO2', 'NO2-', 'NO3', 'NO3-', 'N2O',
        'NH2NO2', 'N2O+', 'N2O3', 'N2O4', 'N2O5', 'NaNO2', 'NaNO3',
    ]


def test_find_species_num_atoms():
    specs = cea.find_species(['Na'], num_atoms=2, charge='all')
    assert list(specs) == ['KNa', 'NaH', 'NaO', 'Na2']
