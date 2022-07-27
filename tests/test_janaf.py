# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

import numpy as np

import chemcat.janaf as janaf


def test_is_in():
    species = 'H2O (KOH)2 HO2 CO'.split()
    in_janaf = janaf.is_in(species)
    np.testing.assert_equal(in_janaf, np.array([True, True, False, True]))


def test_get_janaf_filenames_single():
    janaf_species = janaf.get_filenames('H2O')
    assert len(janaf_species) == 1
    assert janaf_species[0] == 'H-064.txt'


def test_get_janaf_filenames_multiple():
    species = 'H2O e- H+'.split()
    janaf_species = janaf.get_filenames(species)
    assert len(janaf_species) == len(species)
    assert janaf_species[0] == 'H-064.txt'
    assert janaf_species[1] == 'D-020.txt'
    assert janaf_species[2] == 'H-002.txt'


def test_get_janaf_filenames_gas_over_ref():
    janaf_species = janaf.get_filenames('Na')
    assert janaf_species[0] == 'Na-005.txt'


def test_get_janaf_filenames_ref_when_no_gas():
    janaf_species = janaf.get_filenames('H2')
    assert janaf_species[0] == 'H-050.txt'


def test_get_janaf_filenames_missing():
    species = 'H2O H2O+'.split()
    janaf_species = janaf.get_filenames(species)
    assert janaf_species[0] == 'H-064.txt'
    assert janaf_species[1] is None


def test_read_file():
    janaf_file = 'H-064.txt'  # Water
    janaf_data = janaf.read_file(janaf_file)

    expected_temp = np.array([
         100.  ,  200.  ,  298.15,  300.  ,  400.  ,  500.  ,  600.  ,
         700.  ,  800.  ,  900.  , 1000.  , 1100.  , 1200.  , 1300.  ,
        1400.  , 1500.  , 1600.  , 1700.  , 1800.  , 1900.  , 2000.  ,
        2100.  , 2200.  , 2300.  , 2400.  , 2500.  , 2600.  , 2700.  ,
        2800.  , 2900.  , 3000.  , 3100.  , 3200.  , 3300.  , 3400.  ,
        3500.  , 3600.  , 3700.  , 3800.  , 3900.  , 4000.  , 4100.  ,
        4200.  , 4300.  , 4400.  , 4500.  , 4600.  , 4700.  , 4800.  ,
        4900.  , 5000.  , 5100.  , 5200.  , 5300.  , 5400.  , 5500.  ,
        5600.  , 5700.  , 5800.  , 5900.  , 6000.  ])
    expected_heat = np.array([
        4.00494915, 4.01096277, 4.03994841, 4.04067004, 4.12077143,
        4.23671398, 4.3688933 , 4.50961195, 4.65706586, 4.80933066,
        4.96339955, 5.11590489, 5.26408044, 5.405641  , 5.53902304,
        5.6636252 , 5.77908666, 5.88552769, 5.98342939, 6.07327284,
        6.15553913, 6.23287426, 6.30323358, 6.36806038, 6.42783574,
        6.48316103, 6.53427678, 6.58166409, 6.62568377, 6.66669664,
        6.70494325, 6.74054387, 6.77409985, 6.80537067, 6.83483739,
        6.86250003, 6.88871941, 6.91325497, 6.93670808, 6.95883819,
        6.97976558, 6.99973079, 7.01873382, 7.03677468, 7.05397362,
        7.07045094, 7.08620661, 7.10136093, 7.11579361, 7.12914385,
        7.14297517, 7.15728758, 7.17159999, 7.18579212, 7.19998426,
        7.2141764 , 7.22836854, 7.24256068, 7.25663254, 7.27082468,
        7.28501682,
    ])
    expected_gibbs = np.array([
        -317.13342415, -168.5046965 , -120.2630193 , -119.66157194,
         -95.58332709,  -81.34464379,  -71.99422991,  -65.41982799,
         -60.56969923,  -56.86231056,  -53.94984867,  -51.61209304,
         -49.70275118,  -48.12096925,  -46.79482908,  -45.67190338,
         -44.71296187,  -43.88805447,  -43.17402029,  -42.55270104,
         -42.00957008,  -41.53271883,  -41.11268819,  -40.74174781,
         -40.41317506,  -40.12170303,  -39.86246799,  -39.63180789,
         -39.42641198,  -39.24333909,  -39.08005623,  -38.93449306,
         -38.80486807,  -38.68952461,  -38.58701504,  -38.49619122,
         -38.41581874,  -38.34491693,  -38.28262098,  -38.22828742,
         -38.18112061,  -38.14052282,  -38.10584443,  -38.07673679,
         -38.05266426,  -38.03313884,  -38.01795547,  -38.0067065 ,
         -37.99913851,  -37.99478805,  -37.99357992,  -37.99522317,
         -37.99944921,  -38.00613   ,  -38.01515589,  -38.02619341,
         -38.03916486,  -38.05400644,  -38.07042649,  -38.0885058 ,
         -38.10809524])

    assert len(janaf_data) == 3
    np.testing.assert_allclose(janaf_data[0], expected_temp)
    np.testing.assert_allclose(janaf_data[1], expected_heat)
    np.testing.assert_allclose(janaf_data[2], expected_gibbs)


def test_read_file_missing_cp_values():
    janaf_file = 'D-020.txt'  # electron
    janaf_data = janaf.read_file(janaf_file)
    expected_temp = np.array([
         298.15,  300.  ,  350.  ,  400.  ,  450.  ,  500.  ,  600.  ,
         700.  ,  800.  ,  900.  , 1000.  , 1100.  , 1200.  , 1300.  ,
        1400.  , 1500.  , 1600.  , 1700.  , 1800.  , 1900.  , 2000.  ,
        2100.  , 2200.  , 2300.  , 2400.  , 2500.  , 2600.  , 2700.  ,
        2800.  , 2900.  , 3000.  , 3100.  , 3200.  , 3300.  , 3400.  ,
        3500.  , 3600.  , 3700.  , 3800.  , 3900.  , 4000.  , 4100.  ,
        4200.  , 4300.  , 4400.  , 4500.  , 4600.  , 4700.  , 4800.  ,
        4900.  , 5000.  , 5100.  , 5200.  , 5300.  , 5400.  , 5500.  ,
        5600.  , 5700.  , 5800.  , 5900.  , 6000.  ])
    expected_heat = np.array([
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117, 2.49998117, 2.49998117, 2.49998117, 2.49998117,
        2.49998117,
    ])
    expected_gibbs = np.array([
        -2.52319374, -2.52319374, -2.55362264, -2.62121571, -2.70865371,
        -2.80643513, -3.01378467, -3.2216153 , -3.42234986, -3.61334236,
        -3.79387117, -3.96441737, -4.12546205, -4.27784712, -4.42217395,
        -4.55916416, -4.68941912, -4.81354019, -4.93212874, -5.04542529,
        -5.15403123, -5.25830736, -5.35849423, -5.45483239, -5.54780292,
        -5.63740583, -5.72400192, -5.80771148, -5.88877505, -5.9673129 ,
        -6.04356557, -6.11741279, -6.18933566, -6.2592139 , -6.32716778,
        -6.39343785, -6.45790383, -6.52080627, -6.58214517, -6.64204081,
        -6.70061344, -6.75774281, -6.81366946, -6.86839338, -6.92191458,
        -6.97435332, -7.02570962, -7.07598346, -7.12529513, -7.17376489,
        -7.2211522 , -7.26781787, -7.31352137, -7.35838295, -7.40252291,
        -7.44594123, -7.48851764, -7.53049269, -7.57162584, -7.6122779 ,
        -7.65208805])

    assert len(janaf_data) == 3
    np.testing.assert_allclose(janaf_data[0], expected_temp)
    np.testing.assert_allclose(janaf_data[1], expected_heat)
    np.testing.assert_allclose(janaf_data[2], expected_gibbs)


def test_read_stoich_from_species_neutral():
    stoich = janaf.read_stoich('H')
    assert len(stoich) == 1
    assert stoich['H'] == 1.0

    stoich = janaf.read_stoich('H2')
    assert len(stoich) == 1
    assert stoich['H'] == 2.0

    stoich = janaf.read_stoich('H2O')
    assert len(stoich) == 2
    assert stoich['H'] == 2.0
    assert stoich['O'] == 1.0


def test_read_stoich_from_species_ions():
    stoich = janaf.read_stoich('e-')
    assert len(stoich) == 1
    assert stoich['e'] == 1.0

    stoich = janaf.read_stoich('H-')
    assert len(stoich) == 2
    assert stoich['H'] == 1.0
    assert stoich['e'] == 1.0

    stoich = janaf.read_stoich('H+')
    assert len(stoich) == 2
    assert stoich['H'] == 1.0
    assert stoich['e'] == -1.0

    stoich = janaf.read_stoich('H3O+')
    assert len(stoich) == 3
    assert stoich['H'] == 3.0
    assert stoich['O'] == 1.0
    assert stoich['e'] == -1.0


def test_read_stoich_from_janaf():
    stoich = janaf.read_stoich(janaf_file='H-064.txt')
    assert len(stoich) == 2
    assert stoich['H'] == 2.0
    assert stoich['O'] == 1.0


def test_read_stoich_from_formula():
    stoich = janaf.read_stoich(formula='H3O1+')
    assert len(stoich) == 3
    assert stoich['H'] == 3.0
    assert stoich['O'] == 1.0
    assert stoich['e'] == -1.0


def test_setup_janaf_network_neutrals():
    molecules = 'H2O CH4 CO2 H2 H He'.split()
    janaf_data = janaf.setup_network(molecules)

    expected_stoich = [
        {'H': 2.0, 'O': 1.0},
        {'C': 1.0, 'H': 4.0},
        {'C': 1.0, 'O': 2.0},
        {'H': 2.0},
        {'H': 1.0},
        {'He': 1.0},
    ]

    assert len(janaf_data) == 4
    np.testing.assert_equal(janaf_data[3], expected_stoich)


def test_setup_janaf_network_ions():
    molecules = 'H2O CH4 CO2 H2 H He e- H- H2+'.split()
    janaf_data = janaf.setup_network(molecules)

    expected_stoich = [
        {'H': 2.0, 'O': 1.0},
        {'C': 1.0, 'H': 4.0},
        {'C': 1.0, 'O': 2.0},
        {'H': 2.0},
        {'H': 1.0},
        {'He': 1.0},
        {'e': 1.0},
        {'e': 1, 'H': 1.0},
        {'e': -1, 'H': 2.0},
    ]

    np.testing.assert_equal(janaf_data[0], molecules)
    np.testing.assert_equal(janaf_data[3], expected_stoich)


def test_setup_janaf_network_missing_species():
    molecules = 'Ti Ti+ TiO TiO+ TiO2'.split()
    janaf_data = janaf.setup_network(molecules)

    expected_stoich = [
        {'Ti': 1.0},
        {'e': -1, 'Ti': 1.0},
        {'O': 1.0, 'Ti': 1.0},
        {'O': 2.0, 'Ti': 1.0},
    ]

    np.testing.assert_equal(janaf_data[0], ['Ti', 'Ti+', 'TiO', 'TiO2'])
    np.testing.assert_equal(janaf_data[3], expected_stoich)


def test_find_species_single():
    species = janaf.find_species(['K'])
    assert list(species) == [
        'K2', 'K2SO4', 'KBF4', 'KBO2', '(KBr)2', 'KBr', '(KCl)2', 'KCl',
        '(KCN)2', 'KCN', '(KF)2', 'KF', 'K', 'KH', '(KI)2', 'KI', 'KO',
        '(KOH)2', 'KOH']


def test_find_species_dict_without_stoich_values():
    specs = janaf.find_species({'K':None})
    assert list(specs) == [
       'K2', 'K2SO4', 'KBF4', 'KBO2', '(KBr)2', 'KBr', '(KCl)2', 'KCl',
       '(KCN)2', 'KCN', '(KF)2', 'KF', 'K', 'KH', '(KI)2', 'KI', 'KO',
       '(KOH)2', 'KOH']


def test_find_species_multiple():
    specs = janaf.find_species('H C O'.split())
    assert list(specs) == ['C2H4O', 'H2CO', 'HCOF', 'HCO', 'HNCO']


def test_find_species_ions():
    specs = janaf.find_species('C H'.split(), charge='ion')
    assert list(specs) == ['CH+', 'HCO+']


def test_find_species_neutral_and_ions():
    specs = janaf.find_species('H C O'.split(), charge='all')
    assert list(specs) == ['C2H4O', 'H2CO', 'HCOF', 'HCO', 'HCO+', 'HNCO']


def test_find_species_num_atoms():
    specs = janaf.find_species(['Na'], num_atoms=2, charge='all')
    assert list(specs) == ['Na2', 'NaBr', 'NaCl', 'NaF', 'NaH', 'NaO', 'NaO-']


def test_find_species_with_stoich_values():
    specs = janaf.find_species({'K': 2})
    assert list(specs) == [
        'K2', 'K2SO4', '(KBr)2', '(KCl)2', '(KCN)2', '(KF)2', '(KI)2',
        '(KOH)2',
    ]

