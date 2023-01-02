# Copyright (c) 2022-2023 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

import numpy as np

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
     -8.80940438e+00, -1.63642684e+01]
])
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

expected_vmr_1200K_cea = np.array([
       [3.45325582e-04, 3.71921644e-20, 4.94636246e-04, 1.45118536e-07,
        1.15854802e-14, 5.79940366e-05, 8.56535405e-01, 2.03310760e-16,
        7.23395879e-09, 1.82785217e-03, 1.40738635e-01, 3.02703256e-25,
        3.19803133e-16, 1.28228761e-11],
       [3.45557701e-04, 5.91968008e-18, 4.94961215e-04, 1.45104925e-07,
        1.46212094e-13, 5.80321251e-05, 8.57754709e-01, 2.56580135e-15,
        2.03872645e-09, 5.15525276e-04, 1.40831067e-01, 3.03129562e-25,
        9.01623622e-17, 1.01779305e-12],
       [3.45623177e-04, 9.39331175e-16, 4.95052888e-04, 1.45101114e-07,
        1.84197898e-12, 5.80428688e-05, 8.58098670e-01, 3.23238103e-14,
        5.74584864e-10, 1.45323893e-04, 1.40857142e-01, 3.03249823e-25,
        2.54135584e-17, 8.08290814e-14],
       [3.45641635e-04, 1.48924254e-13, 4.95078731e-04, 1.45100042e-07,
        2.31936752e-11, 5.80458870e-05, 8.58195636e-01, 4.07011731e-13,
        1.61939516e-10, 4.09601519e-05, 1.40864493e-01, 3.03283726e-25,
        7.16270014e-18, 6.42009958e-15],
       [3.45646866e-04, 2.36051474e-11, 4.95085987e-04, 1.45099743e-07,
        2.92006823e-10, 5.80466040e-05, 8.58222966e-01, 5.12424809e-12,
        4.56407209e-11, 1.15443231e-05, 1.40866565e-01, 3.03293239e-25,
        2.01873565e-18, 5.09958114e-16],
       [3.45652113e-04, 3.74119461e-09, 4.95084268e-04, 1.45100141e-07,
        3.67615066e-09, 5.80451235e-05, 8.58230662e-01, 6.45093032e-11,
        1.28634404e-11, 3.25364691e-06, 1.40867150e-01, 3.03290302e-25,
        5.68949754e-19, 4.05076644e-17],
       [3.46241108e-04, 5.91231620e-07, 4.94497139e-04, 1.45174788e-07,
        4.62715227e-08, 5.80235917e-05, 8.58232051e-01, 8.09632782e-10,
        3.63158490e-12, 9.17003035e-07, 1.40867486e-01, 3.02415796e-25,
        1.60322084e-19, 3.22311583e-18],
       [4.13488633e-04, 6.77901056e-05, 4.27354149e-04, 1.49845533e-07,
        5.81115714e-07, 5.77607473e-05, 8.58146068e-01, 7.35827658e-09,
        1.22237062e-12, 2.58433623e-07, 1.40886541e-01, 2.18826621e-25,
        4.50824434e-20, 3.05776688e-19],
       [8.06032896e-04, 4.59938961e-04, 3.57257788e-05, 2.44333869e-08,
        7.10310046e-06, 5.45474154e-05, 8.57638596e-01, 3.85714561e-09,
        6.71769908e-13, 7.28149519e-08, 1.40997955e-01, 9.37881226e-27,
        1.23474744e-20, 4.73751353e-20],
       [8.41660084e-04, 4.95501712e-04, 2.53649548e-07, 1.81159528e-10,
        6.26507964e-05, 2.67827347e-05, 8.57557328e-01, 2.31319954e-10,
        1.97708385e-13, 2.05210694e-08, 1.41015802e-01, 6.37639352e-29,
        2.43847610e-21, 3.92984646e-21],
       [8.41957063e-04, 4.95780413e-04, 1.60207616e-09, 1.14467098e-12,
        1.15082137e-04, 5.70255138e-07, 8.57523336e-01, 2.68281218e-12,
        5.57425598e-14, 5.78350855e-09, 1.41023267e-01, 4.02581440e-31,
        1.00282614e-22, 3.12281324e-22]
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

expected_vmr_10x_solar = np.array([
       [3.42317996e-03, 3.56514112e-20, 4.90888312e-03, 1.44348469e-05,
        3.61428944e-14, 5.77069192e-04, 8.49214837e-01, 5.16852489e-16,
        6.05798535e-08, 1.81967082e-03, 1.40041865e-01, 3.01215945e-25,
        1.00810806e-15, 1.28099983e-10],
       [3.42547269e-03, 5.67445808e-18, 4.91210360e-03, 1.44334420e-05,
        4.56133187e-13, 5.77446504e-04, 8.50423879e-01, 6.52273002e-15,
        1.70730087e-08, 5.13217867e-04, 1.40133430e-01, 3.01640658e-25,
        2.84216325e-16, 1.01676825e-11],
       [3.42611942e-03, 9.00420014e-16, 4.91301208e-03, 1.44330486e-05,
        5.74636097e-12, 5.77552940e-04, 8.50764944e-01, 8.21729658e-14,
        4.81177130e-09, 1.44673450e-04, 1.40159260e-01, 3.01760469e-25,
        8.01104259e-17, 8.07476376e-13],
       [3.42630174e-03, 1.42755190e-13, 4.91326819e-03, 1.44329379e-05,
        7.23565400e-11, 5.77582912e-04, 8.50861094e-01, 1.03469745e-12,
        1.35613695e-09, 4.07768222e-05, 1.40166542e-01, 3.01794245e-25,
        2.25787715e-17, 6.41362932e-14],
       [3.42635317e-03, 2.26273275e-11, 4.91334035e-03, 1.44329068e-05,
        9.10964723e-10, 5.77590945e-04, 8.50888194e-01, 1.30267778e-11,
        3.82210972e-10, 1.14926530e-05, 1.40168595e-01, 3.01803760e-25,
        6.36360668e-18, 5.09444106e-15],
       [3.42637142e-03, 3.58627926e-09, 4.91335706e-03, 1.44329034e-05,
        1.14684912e-08, 5.77587985e-04, 8.50895821e-01, 1.63998885e-10,
        1.07721779e-10, 3.23908422e-06, 1.40169176e-01, 3.01805884e-25,
        1.79350345e-18, 4.04664366e-16],
       [3.42694563e-03, 5.68230158e-07, 4.91280171e-03, 1.44336680e-05,
        1.44371840e-07, 5.77521985e-04, 8.50897154e-01, 2.06393130e-09,
        3.03651865e-11, 9.12898682e-07, 1.40169516e-01, 3.01721680e-25,
        5.05449070e-19, 3.21489696e-17],
       [3.51318192e-03, 8.62938400e-05, 4.82779904e-03, 1.45427684e-05,
        1.81600034e-06, 5.76774957e-04, 8.50785478e-01, 2.48860055e-08,
        8.77400503e-12, 2.57272922e-07, 1.40193832e-01, 2.89185184e-25,
        1.42362740e-19, 2.61828842e-18],
       [6.43832025e-03, 2.98766925e-03, 1.95875912e-03, 1.08614198e-05,
        2.25708171e-05, 5.69739458e-04, 8.47001811e-01, 6.84772736e-08,
        4.54190117e-12, 7.23479474e-08, 1.41010128e-01, 6.37382624e-26,
        3.98778091e-20, 3.82846914e-19],
       [8.42613141e-03, 4.95057490e-03, 2.70603203e-05, 1.97008605e-07,
        2.53224800e-04, 4.56843383e-04, 8.44293207e-01, 8.10964166e-09,
        1.67798801e-12, 2.03577929e-08, 1.41592732e-01, 6.70664663e-28,
        1.00641433e-20, 3.99274751e-20],
       [8.46073466e-03, 4.98197885e-03, 1.72876630e-07, 1.26462225e-09,
        1.06563054e-03, 5.11499774e-05, 8.43724915e-01, 2.17132881e-10,
        4.75023300e-13, 5.73567429e-09, 1.41715410e-01, 4.26418964e-30,
        9.49108503e-22, 3.18672154e-21]
])

expected_vmr_e_abund = np.array([
       [6.68807408e-04, 6.54098765e-21, 1.71516785e-04, 9.77018504e-08,
        1.16072133e-14, 5.80128164e-05, 8.56489896e-01, 2.96841243e-17,
        1.17854787e-08, 1.82744858e-03, 1.40784209e-01, 5.43294391e-26,
        3.19635580e-16, 2.48150838e-11],
       [6.69255130e-04, 1.04109647e-18, 1.71629480e-04, 9.76924521e-08,
        1.46486301e-13, 5.80509088e-05, 8.57708901e-01, 3.74617372e-16,
        3.32145998e-09, 5.15411380e-04, 1.40876651e-01, 5.44060906e-26,
        9.01151174e-17, 1.96964840e-12],
       [6.69381424e-04, 1.65200636e-16, 1.71661271e-04, 9.76898201e-08,
        1.84543319e-12, 5.80616537e-05, 8.58052776e-01, 4.71941010e-15,
        9.36103630e-10, 1.45291780e-04, 1.40902729e-01, 5.44277140e-26,
        2.54002413e-17, 1.56421544e-13],
       [6.69417027e-04, 2.61913860e-14, 1.71670233e-04, 9.76890797e-08,
        2.32371687e-11, 5.80646723e-05, 8.58149719e-01, 5.94254086e-14,
        2.63828979e-10, 4.09511004e-05, 1.40910080e-01, 5.44338098e-26,
        7.15894673e-18, 1.24242621e-14],
       [6.69427067e-04, 4.15145017e-12, 1.71672754e-04, 9.76888691e-08,
        2.92554405e-10, 5.80653912e-05, 8.58177043e-01, 7.48161700e-13,
        7.43570405e-11, 1.15417720e-05, 1.40912152e-01, 5.44355261e-26,
        2.01767782e-18, 9.86877597e-16],
       [6.69430561e-04, 6.57974978e-10, 1.71672805e-04, 9.76885315e-08,
        3.68304504e-09, 5.80639325e-05, 8.58184741e-01, 9.41875980e-12,
        2.09566768e-11, 3.25292791e-06, 1.40912737e-01, 5.44357463e-26,
        5.68651729e-19, 7.83901800e-17],
       [6.69535224e-04, 1.04203657e-07, 1.71569438e-04, 9.76447468e-08,
        4.63584435e-08, 5.80426227e-05, 8.58186751e-01, 1.18463911e-10,
        5.90731053e-12, 9.16800723e-07, 1.40912937e-01, 5.43945926e-26,
        1.60238420e-19, 6.22771227e-18],
       [6.84237023e-04, 1.47781736e-05, 1.56905827e-04, 9.12621232e-08,
        5.82256435e-07, 5.77758216e-05, 8.58168175e-01, 1.33148882e-09,
        1.70148310e-12, 2.58386755e-07, 1.40917195e-01, 4.86757143e-26,
        4.50574081e-20, 5.05558107e-19],
       [8.28653846e-04, 1.58915603e-04, 1.29014520e-05, 9.08975944e-09,
        7.11860710e-06, 5.45248832e-05, 8.57979052e-01, 1.10522624e-09,
        5.80820695e-13, 7.28152570e-08, 1.40958750e-01, 3.30407237e-27,
        1.23364588e-20, 4.86444593e-20],
       [8.41552395e-04, 1.71751724e-04, 8.93635331e-08, 6.39452848e-11,
        6.27333668e-05, 2.67227442e-05, 8.57926922e-01, 6.64306482e-11,
        1.66250625e-13, 2.05215043e-08, 1.40970208e-01, 2.25339274e-29,
        2.43407339e-21, 3.92435041e-21],
       [8.41685507e-04, 1.71849674e-04, 5.64323052e-10, 4.03888725e-13,
        1.15050680e-04, 5.67170201e-07, 8.57893232e-01, 7.69232906e-13,
        4.68641239e-14, 5.78363218e-09, 1.40977609e-01, 1.42271729e-31,
        9.99424159e-23, 3.11783783e-22]
])

expected_vmr_e_scale = np.array([
       [5.28442374e-21, 4.00616316e-03, 8.43339166e-04, 3.81592285e-24,
        7.31487504e-20, 2.34096878e-15, 8.51956952e-01, 1.16413211e-04,
        9.33675760e-26, 1.82260631e-03, 1.41254493e-01, 3.36302202e-08,
        2.03044370e-21, 1.97113740e-28],
       [8.41099548e-19, 4.00882042e-03, 8.43891506e-04, 6.06898270e-22,
        1.16427795e-17, 3.72599062e-13, 8.53169746e-01, 1.16489455e-04,
        4.18539981e-24, 5.14045744e-04, 1.41347007e-01, 2.11729849e-10,
        7.21961469e-21, 2.48856430e-27],
       [1.33465717e-16, 4.00956084e-03, 8.44047303e-04, 9.62818164e-20,
        1.84747495e-15, 5.91239131e-11, 8.53511872e-01, 1.16510843e-04,
        1.87142408e-22, 1.44906821e-04, 1.41373102e-01, 1.33510080e-12,
        2.56315415e-20, 3.13542973e-26],
       [2.11599632e-14, 4.00978819e-03, 8.44091240e-04, 1.52638117e-17,
        2.92856134e-13, 9.37060212e-09, 8.53608309e-01, 1.16498285e-04,
        8.36166696e-21, 4.08425982e-05, 1.41380461e-01, 8.42249076e-15,
        9.09445943e-20, 3.94814587e-25],
       [3.35158142e-12, 4.01266475e-03, 8.44105985e-04, 2.41764507e-15,
        4.52687967e-11, 1.41259618e-06, 8.53633680e-01, 1.13693825e-04,
        3.73268445e-19, 1.15111792e-05, 1.41382931e-01, 5.31772864e-17,
        3.14703921e-19, 4.96724004e-24],
       [5.22299512e-10, 4.08073898e-03, 8.44166175e-04, 3.76800742e-13,
        2.84887503e-09, 3.53038528e-05, 8.53597527e-01, 4.59168883e-05,
        1.63945809e-17, 3.24422240e-06, 1.41393100e-01, 3.41247045e-19,
        4.43408683e-19, 6.14898433e-23],
       [8.19332772e-08, 4.12227588e-03, 8.44120652e-04, 5.91073625e-11,
        4.51383985e-08, 5.59249024e-05, 8.53572869e-01, 4.63746953e-06,
        7.24848591e-16, 9.14332897e-07, 1.41399130e-01, 2.17516510e-21,
        1.57288061e-19, 7.66225565e-22],
       [1.27374462e-05, 4.13932750e-03, 8.31472862e-04, 9.05142099e-09,
        5.77643372e-07, 5.77911467e-05, 8.53554014e-01, 3.76024578e-07,
        3.17595349e-14, 2.57691177e-07, 1.41403436e-01, 1.37817431e-23,
        4.50633835e-20, 9.46211603e-21],
       [5.74956390e-04, 4.70668308e-03, 2.69961800e-04, 1.32768704e-07,
        7.07113296e-06, 5.47817949e-05, 8.52822861e-01, 3.31090818e-08,
        4.04215313e-13, 7.25961286e-08, 1.41563446e-01, 9.90452536e-26,
        1.23654883e-20, 3.39557233e-20],
       [8.43034615e-04, 4.97710237e-03, 2.64459243e-06, 1.90790799e-09,
        6.25568608e-05, 2.70889680e-05, 8.52440473e-01, 1.95694678e-09,
        1.67078532e-13, 2.04557815e-08, 1.41647075e-01, 6.61431433e-28,
        2.45069560e-21, 3.95656459e-21],
       [8.45715538e-04, 4.98002428e-03, 1.67513651e-08, 1.21240301e-11,
        1.15576538e-04, 5.83498301e-07, 8.52402746e-01, 2.28289691e-11,
        4.72399211e-14, 5.76509495e-09, 1.41655331e-01, 4.17616991e-30,
        1.01370815e-22, 3.15294492e-22]
])

expected_vmr_e_ratio = np.array([
       [1.67922984e-04, 1.02093871e-19, 6.71992104e-04, 9.61022255e-08,
        1.16057462e-14, 5.79837490e-05, 8.56560809e-01, 4.63145896e-16,
        2.95895471e-09, 1.82752423e-03, 1.40713669e-01, 8.47851224e-25,
        3.19555493e-16, 6.23001164e-12],
       [1.68035455e-04, 1.62497631e-17, 6.72433438e-04, 9.60929859e-08,
        1.46467787e-13, 5.80218238e-05, 8.57779911e-01, 5.84495560e-15,
        8.33912081e-10, 5.15432715e-04, 1.40806068e-01, 8.49046880e-25,
        9.00925395e-17, 4.94495075e-13],
       [1.68067181e-04, 2.57850332e-15, 6.72557936e-04, 9.60903985e-08,
        1.84519995e-12, 5.80325636e-05, 8.58123815e-01, 7.36344324e-14,
        2.35025623e-10, 1.45297794e-04, 1.40832134e-01, 8.49384174e-25,
        2.53938774e-17, 3.92708115e-14],
       [1.68076126e-04, 4.08803338e-13, 6.72593032e-04, 9.60896711e-08,
        2.32342318e-11, 5.80355804e-05, 8.58220764e-01, 9.27182818e-13,
        6.62390042e-11, 4.09527956e-05, 1.40839482e-01, 8.49479254e-25,
        7.15715310e-18, 3.11920506e-15],
       [1.68078722e-04, 6.47970968e-11, 6.72602850e-04, 9.60894983e-08,
        2.92517416e-10, 5.80362939e-05, 8.58248090e-01, 1.16731594e-11,
        1.86686790e-11, 1.15422498e-05, 1.40841553e-01, 8.49505580e-25,
        2.01717221e-18, 2.47763203e-16],
       [1.68089772e-04, 1.02691335e-08, 6.72595310e-04, 9.60938775e-08,
        3.68257731e-09, 5.80347732e-05, 8.58255777e-01, 1.46945112e-10,
        5.26187097e-12, 3.25306254e-06, 1.40842140e-01, 8.49447822e-25,
        5.68508925e-19, 1.96816483e-17],
       [1.69689619e-04, 1.60837463e-06, 6.70997805e-04, 9.67780627e-08,
        4.63521531e-08, 5.80128538e-05, 8.58255869e-01, 1.82779069e-09,
        1.49711154e-12, 9.16837642e-07, 1.40842761e-01, 8.39440717e-25,
        1.60197324e-19, 1.57824858e-18],
       [2.90284213e-04, 1.22192160e-04, 5.50528591e-04, 1.35856998e-07,
        5.82080774e-07, 5.77544464e-05, 8.58101403e-01, 1.10085581e-08,
        7.21873976e-13, 2.58376702e-07, 1.40876851e-01, 4.02533934e-25,
        4.50490725e-20, 2.14497246e-19],
       [7.93220415e-04, 6.24861806e-04, 4.86493662e-05, 3.28305776e-08,
        7.11373241e-06, 5.45506802e-05, 8.57452117e-01, 4.35082437e-09,
        5.56155477e-13, 7.27928935e-08, 1.41019377e-01, 1.30077028e-26,
        1.23393768e-20, 4.65930264e-20],
       [8.41712081e-04, 6.73299337e-04, 3.51091030e-07, 2.51443436e-10,
        6.27152167e-05, 2.67608336e-05, 8.57354285e-01, 2.60867408e-10,
        1.66337693e-13, 2.05146544e-08, 1.41040855e-01, 8.84553124e-29,
        2.43580748e-21, 3.92771668e-21],
       [8.42106162e-04, 6.73684484e-04, 2.21780419e-09, 1.58914794e-12,
        1.15105611e-04, 5.68851091e-07, 8.57320183e-01, 3.02303777e-12,
        4.69032131e-14, 5.78170020e-09, 1.41048344e-01, 5.58478975e-31,
        1.00090403e-22, 3.12148112e-22]
])

expected_vmr_metal_e_ratio = np.array([
       [1.88961113e-15, 1.09657840e-07, 8.33232438e-03, 1.35237656e-17,
        1.89463063e-14, 1.58529084e-04, 8.49294484e-01, 8.33122780e-04,
        3.34387905e-20, 1.81975615e-03, 1.39561674e-01, 9.26316628e-13,
        5.28381515e-16, 7.07051501e-23],
       [2.39314829e-14, 1.37904777e-06, 8.33778308e-03, 1.71144037e-16,
        2.39585197e-13, 1.59267598e-04, 8.50502826e-01, 8.32399260e-04,
        1.19272138e-19, 5.13241688e-04, 1.39653104e-01, 7.32932816e-14,
        1.49264612e-16, 7.10281928e-23],
       [3.14400788e-13, 1.66593946e-05, 8.33944740e-03, 2.24798491e-15,
        3.08979269e-12, 1.66939424e-04, 8.50834009e-01, 8.17285346e-04,
        4.41538401e-19, 1.44679322e-04, 1.39680980e-01, 5.58220670e-15,
        4.30697744e-17, 7.40927476e-23],
       [5.57875656e-12, 1.48833702e-04, 8.34098299e-03, 3.98953096e-14,
        4.59608181e-11, 2.33055759e-04, 8.50844386e-01, 6.85264604e-04,
        2.20810446e-18, 4.07764218e-05, 1.39706700e-01, 3.14656993e-16,
        1.43424406e-17, 1.04429738e-22],
       [2.36588088e-10, 5.55988408e-04, 8.34450052e-03, 1.69309559e-12,
        7.91712671e-10, 4.36699528e-04, 8.50607237e-01, 2.78461906e-04,
        2.63958476e-17, 1.14907554e-05, 1.39765621e-01, 7.42068454e-18,
        5.53330364e-18, 3.51884969e-22],
       [2.60930056e-08, 7.98749530e-04, 8.34653362e-03, 1.86807830e-10,
        1.12646523e-08, 5.58100999e-04, 8.50457294e-01, 3.59327580e-05,
        8.20550144e-16, 3.23824944e-06, 1.39800113e-01, 6.72886747e-20,
        1.76298875e-18, 3.08324838e-21],
       [3.95083557e-06, 8.35640484e-04, 8.34290414e-03, 2.82737554e-08,
        1.43877092e-07, 5.74510439e-04, 8.50432880e-01, 3.02977706e-06,
        3.50167800e-14, 9.12649596e-07, 1.39806000e-01, 4.44197148e-22,
        5.04129490e-19, 3.70839363e-20],
       [4.01457751e-04, 1.23920162e-03, 7.94669740e-03, 2.73822515e-06,
        1.81116727e-06, 5.75476738e-04, 8.49913445e-01, 3.57516518e-07,
        1.00313590e-12, 2.57141039e-07, 1.39918557e-01, 4.16129929e-24,
        1.42202433e-19, 2.99503611e-19],
       [5.18114588e-03, 6.03882772e-03, 3.22376310e-03, 1.44419168e-05,
        2.24592188e-05, 5.70794458e-04, 8.43687144e-01, 1.39355036e-07,
        3.66220270e-12, 7.22062449e-08, 1.41261212e-01, 1.29845277e-25,
        3.99147134e-20, 3.09300985e-19],
       [8.43733347e-03, 9.28668070e-03, 5.17397366e-05, 3.79422664e-07,
        2.51859400e-04, 4.60024128e-04, 8.39312110e-01, 1.54016726e-08,
        1.68519726e-12, 2.02976514e-08, 1.42199837e-01, 1.27306305e-27,
        1.00991180e-20, 4.02178305e-20],
       [8.49730550e-03, 9.34707174e-03, 3.31634472e-07, 2.45103511e-09,
        1.06832511e-03, 5.23378073e-05, 8.38703970e-01, 4.15788614e-10,
        4.78502441e-13, 5.71858254e-09, 1.42330650e-01, 8.09644851e-30,
        9.60065587e-22, 3.21965582e-21]
])
