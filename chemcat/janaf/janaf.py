# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'is_in',
    'get_filenames',
    'read_file',
    'read_stoich',
    'setup_network',
    'find_species',
]

import more_itertools
import numpy as np
import scipy.interpolate as si
import scipy.constants as sc

from .. import utils as u


def is_in(species):
    r"""
    Element-wise check whether species name exist in CEA database.

    Parameters
    ----------
    species: 1D iterable of strings
        Names of species to search in the database.

    Returns
    -------
    in_database: 1D bool array
        Flag whether each species is in the database.

    Examples
    --------
    >>> import chemcat.janaf as janaf
    >>> species = 'H2O (KOH)2 HO2 CO'.split()
    >>> in_janaf = janaf.is_in(species)
    >>> for spec, is_in in zip(species, in_janaf):
    >>>     print(f'{spec:6s}  {is_in}')
    H2O     True
    (KOH)2  True
    HO2     False
    CO      True
    """
    species = np.atleast_1d(species)
    janaf_names = [
        line.split()[0]
        for line in open(f'{u.ROOT}chemcat/data/janaf_conversion.txt', 'r')
    ]

    in_database = np.isin(species, janaf_names)
    return in_database


def get_filenames(species):
    """
    Convert species names to their respective JANAF file names.

    Parameters
    ----------
    species: String or 1D string iterable
        Species to search.

    Returns
    -------
    janaf_names: 1D string array
        Array of janaf filenames.  If a species is not found,
        return None in its place.

    Examples
    --------
    >>> import chemcat.janaf as janaf
    >>> species = 'H2O CH4 CO CO2 H2 e- H- H+ H2+ Na'.split()
    >>> janaf_species = janaf.get_filenames(species)
    >>> for mol, jname in zip(species, janaf_species):
    >>>     print(f'{mol:5}  {jname}')
    H2O    H-064.txt
    CH4    C-067.txt
    CO     C-093.txt
    CO2    C-095.txt
    H2     H-050.txt
    e-     D-020.txt
    H-     H-003.txt
    H+     H-002.txt
    H2+    H-051.txt
    Na     Na-005.txt
    """
    species = np.atleast_1d(species)

    janaf_dict = {}
    for line in open(f'{u.ROOT}chemcat/data/janaf_conversion.txt', 'r'):
        species_name, janaf_name = line.split()
        janaf_dict[species_name] = janaf_name

    janaf_names = [
        janaf_dict[molec] if molec in janaf_dict else None
        for molec in species
    ]
    return janaf_names


def read_file(janaf_file):
    """
    Read a JANAF file to extract tabulated thermal properties.

    Parameters
    ----------
    janaf_file: 1D string array
        A JANAF filename.

    Returns
    -------
    temps: 1D double array
        Tabulated JANAF temperatures (K).
    heat_capacity: 1D double array
        Tabulated JANAF heat capacity cp/R (unitless).
    gibbs_free_energy: 1D double array
        Tabulated JANAF Gibbs free energy G/RT (unitless).

    Examples
    --------
    >>> import chemcat.janaf as janaf
    >>> janaf_file = 'H-064.txt'  # Water
    >>> temps, heat, gibbs = janaf.read_file(janaf_file)
    >>> for i in range(5):
    >>>     print(f'{temps[i]:6.2f}  {heat[i]:.3f}  {gibbs[i]:.3f}')
    100.00  4.005  -317.133
    200.00  4.011  -168.505
    298.15  4.040  -120.263
    300.00  4.041  -119.662
    400.00  4.121  -95.583

    >>> temps, heat, gibbs = janaf.read_file(janaf_file)
    >>> for i in range(5):
    >>>     print(f'{temps[i]:6.2f}  {heat[i]:.3f}')
    298.15  2.500  -2.523
    300.00  2.500  -2.523
    350.00  2.500  -2.554
    400.00  2.500  -2.621
    450.00  2.500  -2.709
    """
    janaf_data = np.genfromtxt(
        f'{u.ROOT}chemcat/data/janaf/{janaf_file}',
        skip_header=3, usecols=(0,1,3,5), delimiter='\t',
        filling_values=np.nan,
        unpack=True,
    )
    temps = janaf_data[0]
    heat_capacity = janaf_data[1] / sc.R

    # https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-1Vol1-Intro.pdf
    # Page 15
    df_H298 = janaf_data[3][temps==298.15]
    gibbs_free_energy = (-janaf_data[2] + df_H298*1000.0/temps) / sc.R

    idx_valid = np.isfinite(heat_capacity)
    return (
        temps[idx_valid],
        heat_capacity[idx_valid],
        gibbs_free_energy[idx_valid],
    )


def read_stoich(species=None, janaf_file=None, formula=None):
    """
    Get the stoichiometric data from the JANAF data base for the
    requested species.

    Parameters
    ----------
    species: String
        A species name (takes precedence over janaf_file argument).
    janaf_file: String
        A JANAF filename.
    formula: String
        A chemical formula in JANAF format (takes precedence over
        species and janaf_file arguments).

    Returns
    -------
    stoich: Dictionary
        Dictionary containing the stoichiometric values for the
        requested species. The dict's keys are the elements/electron
        names and their values are the respective stoich values.

    Examples
    --------
    >>> import chemcat.janaf as janaf
    >>> # From species name:
    >>> for species in 'C H2O e- H2+'.split():
    >>>     print(f'{species}:  {janaf.read_stoich(species)}')
    C:  {'C': 1.0}
    H2O:  {'H': 2.0, 'O': 1.0}
    e-:  {'e': 1.0}
    H2+:  {'e': -1, 'H': 2.0}

    >>> # From JANAF filename:
    >>> print(janaf.read_stoich(janaf_file='H-064.txt'))
    {'H': 2.0, 'O': 1.0}

    >>> # Or directly from the chemical formula:
    >>> print(janaf.read_stoich(formula='H3O1+'))
    {'e': -1, 'H': 3.0, 'O': 1.0}
    """
    # Get chemical formula (JANAF format):
    if formula is None and species is not None:
        janaf_file = get_filenames(species)[0]
    if formula is None and janaf_file is not None:
        with open(f'{u.ROOT}chemcat/data/janaf/{janaf_file}', 'r') as f:
            header = f.readline()
        formula = header.split('\t')[-1]
        formula = formula[0:formula.index('(')]

    if '-' in formula:
        stoich = {'e': 1}
    elif '+' in formula:
        stoich = {'e': -1}
    else:
        stoich = {}

    previous_type = formula[0].isalpha()
    word = ''
    groups = []
    for letter in formula.replace('-','').replace('+',''):
        if letter.isalpha() != previous_type:
            groups.append(word)
            word = ''
        word += letter
        previous_type = letter.isalpha()
    groups.append(word)
    for e, num in more_itertools.chunked(groups,2):
        stoich[e] = float(num)
    return stoich


def setup_network(input_species):
    r"""
    Extract JANAF thermal data for a requested chemical network.

    Parameters
    ----------
    species: 1D string iterable
        Species to search in the JANAF data base.

    Returns
    -------
    species: 1D string array
        Species found in the JANAF database (might differ from input_species).
    heat_capacity_splines: 1D list of numpy splines
        Splines sampling the species' heat capacity/R.
    gibbs_free_energy: 1D list of callable objects
        Functions that return the species's Gibbs free energy, G/RT.
    stoich_data: List of Dictionaries
        Stoichiometric data (as dictionary of element-value pairs) for
        a list of species.

    Examples
    --------
    >>> import chemcat.janaf as janaf

    >>> molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    >>> species, cp_funcs, gibbs_funcs, stoich_data = \
    >>>     janaf.setup_network(molecules)

    >>> for spec, stoich in zip(species, stoich_data):
    >>>     print(f'{spec:3s}:  {stoich}')
    H2O:  {'H': 2.0, 'O': 1.0}
    CH4:  {'C': 1.0, 'H': 4.0}
    CO :  {'C': 1.0, 'O': 1.0}
    CO2:  {'C': 1.0, 'O': 2.0}
    NH3:  {'H': 3.0, 'N': 1.0}
    N2 :  {'N': 2.0}
    H2 :  {'H': 2.0}
    HCN:  {'C': 1.0, 'H': 1.0, 'N': 1.0}
    OH :  {'H': 1.0, 'O': 1.0}
    H  :  {'H': 1.0}
    He :  {'He': 1.0}
    C  :  {'C': 1.0}
    N  :  {'N': 1.0}
    O  :  {'O': 1.0}
    """
    # Find which species exists in data base:
    janaf_species = get_filenames(input_species)
    nspecies = len(input_species)
    idx_missing = np.array([janaf is None for janaf in janaf_species])
    if np.any(idx_missing):
        missing_species = np.array(input_species)[idx_missing]
        print(f'These input species were not found:\n  {missing_species}')

    species = np.array(input_species)[~idx_missing]
    janaf_files = np.array(janaf_species)[~idx_missing]

    nspecies = len(species)
    heat_capacity = []
    gibbs_free_energy = []
    stoich_data = []
    for i in range(nspecies):
        janaf = janaf_files[i]
        janaf_data = read_file(janaf)
        temp = janaf_data[0]
        heat = janaf_data[1]
        gibbs = janaf_data[2]

        h_interp = si.interp1d(temp, heat, fill_value='extrapolate')
        g_interp = si.interp1d(
            temp, gibbs, fill_value='extrapolate', kind='cubic',
        )
        heat_capacity.append(h_interp)
        gibbs_free_energy.append(g_interp)
        stoich_data.append(read_stoich(janaf_file=janaf))

    return (
        species,
        heat_capacity,
        gibbs_free_energy,
        stoich_data,
    )


def find_species(elements, charge='neutral', num_atoms=None, state='gas'):
    """
    Find all JANAF species that contain the specified properties
    (elements, charge, state).

    Parameters
    ----------
    elements: Dict or 1D string iterable
        Either:
        - A list of elements that must be present in the species, or
        - A dictionary of elements and their stoichiometric values.
    charge: String
        If 'neutral', limit the output only to neutrally charged species.
        If 'ion', limit the output only to charged species.
        Else, do not limit output.
    num_atoms: Integer
        Limit the number of atoms to the requested value.
    state: String
        If 'gas', limit the output to gaseous species.

    Returns
    -------
    species: 1D string array
        List of all species containing the required elements.

    Examples
    --------
    >>> import chemcat.janaf as janaf

    >>> # Get all sodium-bearing species:
    >>> salts = janaf.find_species(['Na'])
    >>> print(salts)
    ['LiONa' 'Na2' 'Na2SO4' 'NaAlF4' 'NaBO2' '(NaBr)2' 'NaBr' '(NaCl)2' 'NaCl'
     '(NaCN)2' 'NaCN' '(NaF)2' 'NaF' 'Na' 'NaH' 'NaO' '(NaOH)2' 'NaOH']

    >>> # Get species containing exactly two Na atoms:
    >>> species = janaf.find_species({'Na':2})
    >>> print(species)
    ['Na2' 'Na2SO4' '(NaBr)2' '(NaCl)2' '(NaCN)2' '(NaF)2' '(NaOH)2']

    >>> # Species containing exactly two Na atoms and any amount of oxygen:
    >>> species = janaf.find_species({'Na':2, 'O':None})
    >>> print(species)
    ['Na2SO4' '(NaOH)2']

    >>> # Get all species containing sodium and oxygen (any amount):
    >>> species = janaf.find_species(['Na', 'O'])
    >>> print(species)
    ['LiONa' 'Na2SO4' 'NaBO2' 'NaO' '(NaOH)2' 'NaOH']

    >>> # Get all hydrogen-ion species:
    >>> H_ions = janaf.find_species(['H'], charge='ion')
    >>> print(H_ions)
    ['AlOH-' 'AlOH+' 'BaOH+' 'BeH+' 'BeOH+' 'CaOH+' 'CH+' 'CsOH+' 'H2-' 'H2+'
     'H3O+' 'HBO-' 'HBO+' 'HBS+' 'HCO+' 'HD-' 'HD+' 'H-' 'H+' 'KOH+' 'LiOH+'
     'MgOH+' 'NaOH+' 'OH-' 'OH+' 'SiH+' 'SrOH+']

    >>> # Only diatomic Na species:
    >>> diatomic = janaf.find_species(['Na'], num_atoms=2, charge='all')
    >>> print(diatomic)
    ['Na2' 'NaBr' 'NaCl' 'NaF' 'NaH' 'NaO' 'NaO-']
    """
    # Turn into dict if needed:
    if not isinstance(elements, dict):
        elements = {e:None for e in elements}

    janaf_dict = {}
    for line in open(f'{u.ROOT}chemcat/data/janaf_conversion.txt', 'r'):
        species_name, janaf_name = line.split()
        janaf_dict[species_name] = janaf_name

    species = []
    for molec, janaf_file in janaf_dict.items():
        if state == 'gas' and '_' in molec:
            continue

        is_charged = '+' in molec or '-' in molec
        if charge == 'neutral' and is_charged:
            continue
        elif charge == 'ion' and not is_charged:
            continue

        stoich = read_stoich(janaf_file=janaf_file)

        if num_atoms is not None:
            n_atoms = sum([val for key,val in stoich.items() if key!='e'])
            if n_atoms != num_atoms:
                continue

        # Request selected stoichiometric values:
        for key, val in elements.items():
            if key not in stoich or (val is not None and stoich[key] != val):
                break
        else:
            species.append(molec)

    return np.array(species)


