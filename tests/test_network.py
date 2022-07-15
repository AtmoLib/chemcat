# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = str(Path(__file__).parents[1]) + os.path.sep
sys.path.append(ROOT+'chemcat')
import chemcat as cat
import chemcat.janaf as janaf


# Some default values:
element_file = f'{cat.ROOT}chemcat/data/asplund_2021_solar_abundances.dat'

expected_dex_asplund_2009 = np.array([
    7.3 , 12.  , 10.93,  1.05,  1.38,  2.7 ,  8.43,  7.83,  8.69,
    4.56,  7.93,  6.24,  7.6 ,  6.45,  7.51,  5.41,  7.12,  5.5 ,
    6.4 ,  5.03,  6.34,  3.15,  4.95,  3.93,  5.64,  5.43,  7.5 ,
    4.99,  6.22,  4.19,  4.56,  3.04,  3.65,  3.25,  2.52,  2.87,
    2.21,  2.58,  1.46,  1.88,  1.75,  0.91,  1.57,  0.94,  0.8 ,
    2.04,  2.24,  2.18,  1.1 ,  1.58,  0.72,  1.42,  0.96,  0.52,
    1.07,  0.3 ,  1.1 ,  0.48,  0.92,  0.1 ,  0.84,  0.1 ,  0.85,
    0.85,  1.4 ,  1.38,  0.92,  0.9 ,  1.75,  0.02])

expected_dex_asplund_2021 = np.array([
    7.3  , 12.   , 10.914,  0.96 ,  1.38 ,  2.7  ,  8.46 ,  7.83 ,
    8.69 ,  4.4  ,  8.06 ,  6.22 ,  7.55 ,  6.43 ,  7.51 ,  5.41 ,
    7.12 ,  5.31 ,  6.38 ,  5.07 ,  6.3  ,  3.14 ,  4.97 ,  3.9  ,
    5.62 ,  5.42 ,  7.46 ,  4.94 ,  6.2  ,  4.18 ,  4.56 ,  3.02 ,
    3.62 ,  3.12 ,  2.32 ,  2.83 ,  2.21 ,  2.59 ,  1.47 ,  1.88 ,
    1.75 ,  0.78 ,  1.57 ,  0.96 ,  0.8  ,  2.02 ,  2.22 ,  2.27 ,
    1.11 ,  1.58 ,  0.75 ,  1.42 ,  0.95 ,  0.52 ,  1.08 ,  0.31 ,
    1.1  ,  0.48 ,  0.93 ,  0.11 ,  0.85 ,  0.1  ,  0.85 ,  0.79 ,
    1.35 ,  0.91 ,  0.92 ,  1.95 ,  0.03 ])

expected_vmr_1200K = np.array([
       [3.45326436e-04, 3.65395656e-20, 4.94635978e-04, 1.45474668e-07,
        1.16062671e-14, 5.79940469e-05, 8.56535735e-01, 1.65782682e-16,
        6.08505289e-09, 1.82749748e-03, 1.40738660e-01, 3.03465124e-25,
        3.19583868e-16, 1.28121276e-11],
       [3.45557686e-04, 5.81581847e-18, 4.94960884e-04, 1.45460679e-07,
        1.46474360e-13, 5.80321280e-05, 8.57754803e-01, 2.09219739e-15,
        1.71492943e-09, 5.15425171e-04, 1.40831074e-01, 3.03893146e-25,
        9.01005390e-17, 1.01693762e-12],
       [3.45622917e-04, 9.22850884e-16, 4.95052539e-04, 1.45456761e-07,
        1.84528276e-12, 5.80428696e-05, 8.58098696e-01, 2.63573904e-14,
        4.83327142e-10, 1.45295668e-04, 1.40857144e-01, 3.04013891e-25,
        2.53961322e-17, 8.07610960e-14],
       [3.45641307e-04, 1.46311439e-13, 4.95078377e-04, 1.45455659e-07,
        2.32352746e-11, 5.80458872e-05, 8.58195643e-01, 3.31884411e-13,
        1.36219650e-10, 4.09521962e-05, 1.40864493e-01, 3.04047930e-25,
        7.15778860e-18, 6.41469848e-15],
       [3.45646517e-04, 2.31910058e-11, 4.95085633e-04, 1.45455352e-07,
        2.92530554e-10, 5.80466042e-05, 8.58222969e-01, 4.17840080e-12,
        3.83918811e-11, 1.15420808e-05, 1.40866565e-01, 3.04057484e-25,
        2.01735138e-18, 5.09529069e-16],
       [3.45651682e-04, 3.67555846e-09, 4.95083990e-04, 1.45455739e-07,
        3.68274412e-09, 5.80451261e-05, 8.58230663e-01, 5.26020266e-11,
        1.08204154e-11, 3.25301494e-06, 1.40867150e-01, 3.04054657e-25,
        5.68559631e-19, 4.04735745e-17],
       [3.46230255e-04, 5.80888687e-07, 4.94507267e-04, 1.45529249e-07,
        4.63545270e-08, 5.80236237e-05, 8.58232065e-01, 6.60222833e-10,
        3.05470954e-12, 9.16824927e-07, 1.40867483e-01, 3.03193457e-25,
        1.60212193e-19, 3.22030639e-18],
       [4.12584149e-04, 6.68879327e-05, 4.28256395e-04, 1.50200689e-07,
        5.82159354e-07, 5.77607869e-05, 8.58147227e-01, 6.02592776e-09,
        1.02598004e-12, 2.58383601e-07, 1.40886287e-01, 2.20323360e-25,
        4.50515452e-20, 3.04851019e-19],
       [8.05463713e-04, 4.59371258e-04, 3.62931535e-05, 2.48647063e-08,
        7.11546329e-06, 5.45414987e-05, 8.57639318e-01, 3.19722081e-09,
        5.64678058e-13, 7.28008392e-08, 1.40997796e-01, 9.55852395e-27,
        1.23383383e-20, 4.73018593e-20],
       [8.41655603e-04, 4.95497252e-04, 2.58175601e-07, 1.84843339e-10,
        6.27216583e-05, 2.67473271e-05, 8.57557289e-01, 1.91862029e-10,
        1.66306844e-13, 2.05170830e-08, 1.41015811e-01, 6.50655859e-29,
        2.43519271e-21, 3.92652341e-21],
       [8.41957038e-04, 4.95780387e-04, 1.63068618e-09, 1.16797005e-12,
        1.15086139e-04, 5.68254588e-07, 8.57523333e-01, 2.22276834e-12,
        4.68893522e-14, 5.78238518e-09, 1.41023268e-01, 4.10803211e-31,
        1.00037912e-22, 3.12018899e-22],
])
expected_vmr_2900K = np.array([
       [5.90659151e-16, 7.03878906e-33, 2.19547811e-04, 2.84201394e-14,
        3.36563349e-26, 7.30027802e-07, 6.44002149e-07, 2.83136249e-15,
        2.65565082e-10, 9.23665276e-01, 7.57731293e-02, 4.68405356e-05,
        6.09874683e-05, 2.32844242e-04],
       [7.69538698e-14, 2.03165903e-29, 2.61021274e-04, 3.49649831e-13,
        5.51322288e-23, 6.19323058e-06, 8.10812656e-06, 4.23240743e-14,
        2.74819723e-09, 9.23700585e-01, 7.57772507e-02, 5.38156166e-06,
        5.00644593e-05, 1.91392903e-04],
       [1.18790929e-11, 4.23877765e-26, 2.65985414e-04, 4.36966170e-12,
        5.41497035e-20, 1.89032990e-05, 1.02056520e-04, 2.74414529e-13,
        3.37008298e-08, 9.23615765e-01, 7.57857091e-02, 4.47154555e-07,
        2.46512932e-05, 1.86448291e-04],
       [1.87172027e-09, 8.46075302e-23, 2.66713342e-04, 5.49894479e-11,
        3.63146053e-17, 2.71062206e-05, 1.28130877e-03, 1.17117448e-12,
        4.22368711e-07, 9.22354473e-01, 7.58757513e-02, 3.57272802e-08,
        8.31965638e-06, 1.85867317e-04],
       [2.83134690e-07, 1.62177578e-19, 2.70561380e-04, 6.93236794e-10,
        2.05845584e-14, 3.04702874e-05, 1.55964140e-02, 4.45195486e-12,
        5.16129163e-06, 9.06949982e-01, 7.69611602e-02, 2.91635129e-09,
        2.48604612e-06, 1.83478091e-04],
       [2.54066417e-05, 2.42566288e-16, 3.04228718e-04, 7.68806506e-09,
        7.65272168e-12, 3.52830298e-05, 1.41898528e-01, 2.07398370e-11,
        4.32748938e-05, 7.71009811e-01, 8.65389628e-02, 3.32485871e-10,
        7.53969611e-07, 1.43742520e-04],
       [1.98608997e-04, 2.55392507e-13, 3.95709395e-04, 2.28714548e-08,
        6.97064427e-10, 4.62622406e-05, 4.84985731e-01, 3.14330728e-10,
        5.15717812e-05, 4.01731087e-01, 1.12564647e-01, 1.89081380e-10,
        2.43323602e-07, 2.61147481e-05],
       [3.01665652e-04, 1.05500874e-10, 4.60816745e-04, 2.69049159e-08,
        1.74759034e-08, 5.39681006e-05, 7.29239103e-01, 6.04196910e-09,
        1.80039825e-05, 1.38837228e-01, 1.31086994e-01, 2.17979020e-10,
        7.40695101e-08, 2.09542345e-06],
       [3.33253114e-04, 2.26150916e-08, 4.84821165e-04, 2.78198126e-08,
        2.68629508e-07, 5.66536594e-05, 8.19688906e-01, 8.84499410e-08,
        5.28721998e-06, 4.14854197e-02, 1.37945066e-01, 2.33345146e-10,
        2.13887246e-08, 1.63584082e-07],
       [3.47110276e-04, 3.81884141e-06, 4.87332669e-04, 2.81814858e-08,
        3.51339228e-06, 5.53848839e-05, 8.47181730e-01, 1.11640346e-06,
        1.52671114e-06, 1.18866438e-02, 1.40031776e-01, 2.32743197e-10,
        5.96027819e-09, 1.30950383e-08],
       [5.64389401e-04, 2.15671788e-04, 2.74870970e-04, 2.56146662e-08,
        3.68874143e-05, 3.74994283e-05, 8.54803902e-01, 4.06597512e-06,
        6.96503486e-07, 3.36514829e-03, 1.40696840e-01, 8.14627154e-11,
        1.38223928e-09, 1.67620927e-09],
])

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

    expected_cp = np.array([
        [ 4.00494915,  4.00001798,  3.50040662,  2.55831326,  2.49998117],
        [ 4.04067004,  4.29468525,  3.50497697,  2.50623533,  2.49998117],
        [ 4.23671398,  5.57366148,  3.58339455,  2.50214607,  2.49998117],
        [ 4.50961195,  6.95102049,  3.74900958,  2.50106362,  2.49998117],
        [ 4.80933066,  8.13053147,  3.91811251,  2.50070281,  2.49998117],
        [ 5.11590489,  9.0840507 ,  4.05438109,  2.50058253,  2.49998117],
        [ 5.405641  ,  9.83154339,  4.15805586,  2.5011839 ,  2.49998117],
        [ 5.6636252 , 10.41029396,  4.23563153,  2.5033488 ,  2.49998117],
        [ 5.88552769, 10.85854903,  4.2949258 ,  2.5076786 ,  2.49998117],
        [ 6.07327284, 11.20794022,  4.34074957,  2.51513549,  2.49998117],
        [ 6.23287426, 11.48324364,  4.37695154,  2.52559918,  2.49998117],
        [ 6.36806038, 11.70262042,  4.40617773,  2.53894941,  2.49998117],
        [ 6.48316103, 11.87954105,  4.43035247,  2.55470509,  2.49998117],
        [ 6.58166409, 12.02374761,  4.45043795,  2.57226486,  2.49998117],
        [ 6.66669664, 12.14269697,  4.46811799,  2.59090707,  2.49998117],
        [ 6.74054387, 12.24156084,  4.48363312,  2.61003038,  2.49998117],
        [ 6.80537067, 12.32478931,  4.4972239 ,  2.62903341,  2.49998117],
        [ 6.86250003, 12.39526891,  4.50937141,  2.64743508,  2.49998117],
        [ 6.91325497, 12.45540509,  4.52091755,  2.66511512,  2.49998117],
        [ 6.95883819, 12.5071222 ,  4.53102043,  2.68183297,  2.49998117],
        [ 6.99973079, 12.55198379,  4.54100304,  2.69722783,  2.49998117],
        [ 7.03677468, 12.5910723 ,  4.55014374,  2.71141997,  2.49998117],
        [ 7.07045094, 12.62522965,  4.55868307,  2.72440939,  2.49998117]])
    np.testing.assert_allclose(cp, expected_cp)


def test_thermo_eval_gibbs_free_energy_temp_array():
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    janaf_data = janaf.setup_network(molecules)
    gibbs_funcs = janaf_data[3]
    temperatures = np.arange(100.0, 4101.0, 500.0)
    gibbs = cat.thermo_eval(temperatures, gibbs_funcs)

    expected_gibbs = np.array([
        [-3.17133424e+02, -1.16088681e+02, -1.59818988e+02,
         -5.02592674e+02, -8.20487182e+01, -2.61580345e+01,
         -1.86912862e+01,  1.34767459e+02,  2.15155216e+01,
          2.46172614e+02, -1.73952313e+01,  8.40705686e+02,
          5.47846591e+02,  2.77901183e+02],
        [-7.19942299e+01, -3.83538117e+01, -4.66207721e+01,
         -1.05556070e+02, -3.32912275e+01, -2.37361101e+01,
         -1.64041870e+01,  1.90286902e+00, -1.49815655e+01,
          2.94108805e+01, -1.56631891e+01,  1.24152943e+02,
          7.58228197e+01,  3.00677680e+01],
        [-5.16120930e+01, -3.38239976e+01, -3.79394442e+01,
         -7.17936083e+01, -3.11258185e+01, -2.51133488e+01,
         -1.77460657e+01, -1.23649275e+01, -1.98927195e+01,
          8.59717728e+00, -1.66138218e+01,  5.79016593e+01,
          3.18033564e+01,  6.39222419e+00],
        [-4.47129619e+01, -3.34225599e+01, -3.52753255e+01,
         -6.01006070e+01, -3.13341056e+01, -2.62131192e+01,
         -1.87902703e+01, -1.86146276e+01, -2.22838364e+01,
          4.23524064e-01, -1.73388235e+01,  3.26889136e+01,
          1.49277086e+01, -2.85681422e+00],
        [-4.15327188e+01, -3.40086693e+01, -3.42015976e+01,
         -5.45090464e+01, -3.20565745e+01, -2.71075847e+01,
         -1.96337403e+01, -2.23952165e+01, -2.38381927e+01,
         -4.04703875e+00, -1.79077118e+01,  1.92925792e+01,
          5.89877483e+00, -7.89133267e+00],
        [-3.98624680e+01, -3.48977825e+01, -3.37428163e+01,
         -5.14044786e+01, -3.29091716e+01, -2.78585653e+01,
         -2.03446702e+01, -2.50470695e+01, -2.49885061e+01,
         -6.91376381e+00, -1.83734063e+01,  1.09318319e+01,
          2.26972437e-01, -1.11052554e+01],
        [-3.89344931e+01, -3.58722619e+01, -3.35705245e+01,
         -4.95326757e+01, -3.37755811e+01, -2.85049090e+01,
         -2.09616674e+01, -2.70695166e+01, -2.59028825e+01,
         -8.93367922e+00, -1.87669375e+01,  5.18817306e+00,
         -3.69351747e+00, -1.33607688e+01],
        [-3.84158187e+01, -3.68512157e+01, -3.35468397e+01,
         -4.83493264e+01, -3.46155197e+01, -2.90721134e+01,
         -2.15080647e+01, -2.86960551e+01, -2.66629713e+01,
         -1.04488680e+01, -1.91073082e+01,  9.81302145e-01,
         -6.58180440e+00, -1.50464658e+01],
        [-3.81405228e+01, -3.78024079e+01, -3.36057563e+01,
         -4.75832286e+01, -3.54147523e+01, -2.95772573e+01,
         -2.19996178e+01, -3.00526088e+01, -2.73143300e+01,
         -1.16368930e+01, -1.94072675e+01, -2.24469576e+00,
         -8.80940438e+00, -1.63642684e+01]])
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
    nlayers = 81
    temperature = np.tile(1200.0, nlayers)
    pressure = np.logspace(-8, 3, nlayers)
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    nspecies = len(molecules)
    net = cat.Network(pressure, temperature, molecules)
    cp = net.heat_capacity()

    expected_cp = np.array([
        5.26408044, 9.48143057, 4.11030773, 6.77638503, 7.34238673,
        4.05594463, 3.72748083, 6.3275286 , 3.79892261, 2.49998117,
        2.49998117, 2.50082308, 2.49998117, 2.51092596])
    assert np.shape(cp) == (nlayers, nspecies)
    np.testing.assert_allclose(cp[0], expected_cp)


def test_network_cp_input_temp():
    nlayers = 81
    temperature = np.tile(1200.0, nlayers)
    pressure = np.logspace(-8, 3, nlayers)
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    nspecies = len(molecules)
    net = cat.Network(pressure, temperature, molecules)

    temps = [100.0, 600.0, 1200.0]
    cp = net.heat_capacity(temps)

    expected_cp = np.array([
       [4.00494915, 4.00001798, 3.50040662, 3.51291495, 4.00314507,
        3.50040662, 3.38614788, 3.50798378, 3.92412613, 2.49998117,
        2.49998117, 2.55831326, 2.49998117, 2.85081563],
       [4.3688933 , 6.28146429, 3.6614513 , 5.69140811, 5.44749578,
        3.62140061, 3.52722736, 5.26865079, 3.55128183, 2.49998117,
        2.49998117, 2.50154471, 2.49998117, 2.54063323],
       [5.26408044, 9.48143057, 4.11030773, 6.77638503, 7.34238673,
        4.05594463, 3.72748083, 6.3275286 , 3.79892261, 2.49998117,
        2.49998117, 2.50082308, 2.49998117, 2.51092596]])

    assert np.shape(cp) == (len(temps), nspecies)
    np.testing.assert_allclose(cp, expected_cp)
    np.testing.assert_equal(net.temperature, temperature)


def test_network_gibbs_default_temp():
    nlayers = 81
    temperature = np.tile(1200.0, nlayers)
    pressure = np.logspace(-8, 3, nlayers)
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    nspecies = len(molecules)
    net = cat.Network(pressure, temperature, molecules)
    gibbs = net.gibbs_free_energy()

    expected_gibbs = np.array([
        -49.70275118, -33.59076581, -37.17544326, -68.58699428,
        -31.08382889, -25.35365299, -17.97578591, -13.94856633,
        -20.48067821,   6.44982554, -16.77498672,  51.21052551,
         27.33544072,   3.95818325])

    assert np.shape(gibbs) == (nlayers, nspecies)
    np.testing.assert_allclose(gibbs[0], expected_gibbs)
    np.testing.assert_equal(net.temperature, temperature)


def test_network_vmr_default():
    nlayers = 11
    temperature = np.tile(1200.0, nlayers)
    pressure = np.logspace(-8, 3, nlayers)
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    nspecies = len(molecules)
    net = cat.Network(pressure, temperature, molecules)
    vmr = net.thermochemical_equilibrium()

    assert np.shape(vmr) == (nlayers, nspecies)
    np.testing.assert_allclose(vmr, expected_vmr_1200K)
    np.testing.assert_allclose(net.vmr, expected_vmr_1200K)

    assert np.all(vmr>=0)
    elem_fractions = np.sum(net.vmr[0]*net.stoich_vals.T, axis=1)
    elem_fractions /= elem_fractions[net.elements == 'H']
    np.testing.assert_allclose(elem_fractions, net.element_rel_abundance)


def test_network_vmr_update_temp():
    nlayers = 11
    temperature = np.tile(1200.0, nlayers)
    pressure = np.logspace(-8, 3, nlayers)
    molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    nspecies = len(molecules)
    net = cat.Network(pressure, temperature, molecules)

    vmr = net.thermochemical_equilibrium()
    assert np.shape(vmr) == (nlayers, nspecies)
    np.testing.assert_allclose(vmr, expected_vmr_1200K)

    thot = np.tile(2900.0, nlayers)
    vmr = net.thermochemical_equilibrium(temperature=thot)
    assert np.shape(vmr) == (nlayers, nspecies)
    np.testing.assert_allclose(vmr, expected_vmr_2900K)
    np.testing.assert_equal(net.temperature, thot)


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

