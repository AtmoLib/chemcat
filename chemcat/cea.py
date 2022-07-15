# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'is_in',
    'read_thermo_build',
    'heat_func',
    'gibbs_func',
    'setup_network',
    'find_species',
]

import sys
from collections.abc import Iterable

from more_itertools import sliced
import numpy as np

from .utils import ROOT
sys.path.append(f'{ROOT}chemcat/lib')
import _utils as u


def is_in(species, thermo_file=None):
    r"""
    Element-wise check whether species name exist in CEA database.
    Parameters
    ----------
    species: 1D iterable of strings
        Names of species to search in the database.
    thermo_file: String
        Optional ThermoBuild CEA database file path.
    Returns
    -------
    in_database: 1D bool array
        Flag whether each species is in the database.
    Examples
    --------
    >>> import chemcat.cea as cea
    >>> species = 'H2O (KOH)2 HO2 CO'.split()
    >>> in_cea = cea.is_in(species)
    >>> for spec, is_in in zip(species, in_cea):
    >>>     print(f'{spec:6s}  {is_in}')
    H2O     True
    (KOH)2  False
    HO2     True
    CO      True
    """
    if thermo_file is None:
        thermo_file = f'{ROOT}chemcat/data/thermo_build_cea.dat'

    with open(thermo_file, 'r') as f:
        lines = f.readlines()
    nlines = len(lines)

    all_species = []
    i = 0
    while i < nlines:
        line = lines[i]
        all_species.append(line[0:16].strip())
        line = lines[i+1]
        ndata_intervals = int(line[0:2])
        i += 2 + 3*ndata_intervals

    in_database = np.isin(species, all_species)
    return in_database


def read_thermo_build(species, thermo_file=None):
    """
    Read data from NASA's CEA thermoBuild file.
    https://cearun.grc.nasa.gov/ThermoBuild/index_ds.html

    Parameters
    ----------
    species: 1D iterable of string
        List of species names to extract their info.
    thermo_file: String
        Path to a file containing CEA ThermoBuild data.

    Returns
    -------
    thermo_data: Dict
        A dictionary containing the species thermal properties from the
        CEA database (one entry for each species):
        name: String
            Species name.
        stoich: Dict
            Stoichiometric value of the species as element-value pairs.
        a_coeffs: 2D float ndarray
            Polynomial coefficients to reproduce the heat capacity data.
        b_coeffs: 2D float ndarray
            Integration constants to obtain the enthalpy and entropy.
        t_coeffs: 1D float ndarray
            Temperature intervals of validity for each set of coefficients.

    Examples
    --------
    >>> import chemcat.cea as cea

    >>> # A simple HCNO network:
    >>> hcno_species = 'H2O CH4 CO CO2 NH3 HCN N2 H2 H He'.split()
    >>> hcno_thermo_data = cea.read_thermo_build(hcno_species)

    >>> # Network will all species from the database:
    >>> all_thermo_data = cea.read_thermo_build(species=None)
    """
    if thermo_file is None:
        thermo_file = f'{ROOT}chemcat/data/thermo_build_cea.dat'

    with open(thermo_file, 'r') as f:
        lines = f.readlines()
    nlines = len(lines)

    all_species = []
    loc_species = []
    i = 0
    ndata_max = 0
    while i < nlines:
        line = lines[i]
        all_species.append(line[0:16].strip())
        loc_species.append(i)
        line = lines[i+1]
        ndata_intervals = int(line[0:2])
        i += 2 + 3*ndata_intervals
        ndata_max = np.amax([ndata_max, ndata_intervals])

    if species is None:
        species = all_species

    thermo_data = []
    for species_name in species:
        if species_name not in all_species:
            print(f'{species_name} not in Thermo Build database')
            continue
        i = loc_species[all_species.index(species_name)]
        line = lines[i+1]
        ndata_intervals = int(line[0:2])
        # Chemical composition:
        stoich = {}
        for element in sliced(line[10:50], 8):
            if element[0:2].strip() == '':
                break
            element_name = element[0:2].strip().capitalize()
            if element_name == 'E':
                element_name = 'e'
            stoich[element_name] = float(element[2:])
        t_coeffs = np.zeros(ndata_max+1)
        a_coeffs = np.zeros((ndata_max,7))
        b_coeffs = np.zeros((ndata_max,2))
        for j in range(ndata_intervals):
            line = lines[i+3*j+2]
            t_coeffs[j:j+2] = line[0:22].split()
            line = lines[i+3*j+3].replace('D','E')
            a_coeffs[j,0:5] = list(sliced(line[0:80],16))
            line = lines[i+3*j+4].replace('D','E')
            a_coeffs[j,5:7] = list(sliced(line[0:32],16))
            b_coeffs[j] = list(sliced(line[48:80],16))
        thermo_data.append({
            'name': species_name,
            'stoich': stoich,
            'a_coeffs': a_coeffs,
            'b_coeffs': b_coeffs,
            't_coeffs': t_coeffs,
        })

    return thermo_data


def heat_func(a_coeffs, t_coeffs):
    """
    Generate a callable that evaluates the molar heat capacity
    at a given temperature array.

    Parameters
    ----------
    a_coeffs: 2D float ndarray
        Polynomial coefficients to reproduce the heat capacity data.
    t_coeffs: 1D float ndarray
        Temperature intervals of validity for each set of coefficients.

    Returns
    -------
    heat: Callable
        A function heat(temperature) that evaluates the molar heat
        capacity, cp(T)/R, for a given temperature input
        (which can be a single value or a 1D iterable).

    Examples
    --------
    >>> import chemcat.cea as cea

    >>> data = cea.read_thermo_build(['H2O'])[0]
    >>> heat = cea.heat_func(
    >>>     data['a_coeffs'], data['t_coeffs'])

    >>> print(heat(300.0))
    [4.04063805]
    >>> print(heat([300.0, 1000.0, 3000.0]))
    [4.04063805 4.96614188 6.8342561 ]
    """
    def heat(temperature):
        if not isinstance(temperature, Iterable):
            temperature = [temperature]
        temperature = np.array(temperature, np.double)

        heat_capacity = u.heat(
            temperature, a_coeffs, t_coeffs)
        return heat_capacity
    return heat


def gibbs_func(a_coeffs, b_coeffs, t_coeffs):
    """
    Generate a callable that evaluates the Gibbs free energy
    for a given temperature array.

    Parameters
    ----------
    a_coeffs: 2D float ndarray
        Polynomial coefficients to reproduce the heat capacity data.
    b_coeffs: 2D float ndarray
        Integration constants to obtain the enthalpy and entropy.
    t_coeffs: 1D float ndarray
        Temperature intervals of validity for each set of coefficients.

    Returns
    -------
    gibbs: Callable
        A function gibbs(temperature) that evaluates the Gibbs free
        energy, G(T)/RT, for a given temperature input (which can be
        a single value or a 1D iterable).

    Examples
    --------
    >>> import chemcat.cea as cea

    >>> data = cea.read_thermo_build(['H2O'])[0]
    >>> gibbs = cea.gibbs_func(
    >>>     data['a_coeffs'], data['b_coeffs'], data['t_coeffs'])

    >>> print(gibbs(300.0))
    [-119.66025955]
    >>> print(gibbs([300.0, 1000.0, 3000.0]))
    [-119.66025955  -53.94898416  -39.09425268]
    """
    def gibbs(temperature):
        if not isinstance(temperature, Iterable):
            temperature = [temperature]
        temperature = np.array(temperature, np.double)

        free_energy = u.gibbs(
            temperature, a_coeffs, b_coeffs, t_coeffs)
        return free_energy
    return gibbs


def setup_network(input_species):
    r"""
    Extract CEA thermal data for a requested chemical system.

    Parameters
    ----------
    species: 1D string iterable
        Species to search in the CEA data base.

    Returns
    -------
    species: 1D string array
        Species found in the CEA database (might differ from
        input_species if there are species not found on the database).
    heat_capacity: 1D list of callable objects
        Functions that evaluate the species's heat capacity (cp/R)
        at requested temperatures.
    gibbs_free_energy: 1D list of callable objects
        Functions that evaluate the species's Gibbs free energy (G/RT)
        at requested temperatures.
    stoich_data: List of dictionaries
        Stoichiometric data (as dictionary of element-value pairs) for
        a list of species.

    Examples
    --------
    >>> import chemcat.cea as cea

    >>> molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    >>> species, heat_capacity, gibbs, stoich_data = \
    >>>     cea.setup_network(molecules)

    >>> for spec, stoich in zip(species, stoich_data):
    >>>     print(f'{spec:3s}:  {stoich}')
    H2O:  {'H': 2.0, 'O': 1.0}
    CH4:  {'C': 1.0, 'H': 4.0}
    CO :  {'C': 1.0, 'O': 1.0}
    CO2:  {'C': 1.0, 'O': 2.0}
    NH3:  {'N': 1.0, 'H': 3.0}
    N2 :  {'N': 2.0}
    H2 :  {'H': 2.0}
    HCN:  {'H': 1.0, 'C': 1.0, 'N': 1.0}
    OH :  {'O': 1.0, 'H': 1.0}
    H  :  {'H': 1.0}
    He :  {'He': 1.0}
    C  :  {'C': 1.0}
    N  :  {'N': 1.0}
    O  :  {'O': 1.0}
    """
    # Find which species exists in data base:
    thermo_data = read_thermo_build(input_species)

    cea_species = [data['name'] for data in thermo_data]
    idx_missing = np.array([
        spec not in cea_species for spec in input_species])
    if np.any(idx_missing):
        missing_species = np.array(input_species)[idx_missing]
        print(
            'These input species were not found in CEA database:\n'
            f'  {missing_species}')

    species = np.array(input_species)[~idx_missing]

    heat_capacity = []
    gibbs_free_energy = []
    stoich_data = []
    for data in thermo_data:
        heat_capacity.append(
            heat_func(data['a_coeffs'], data['t_coeffs']))
        gibbs_free_energy.append(
            gibbs_func(data['a_coeffs'], data['b_coeffs'], data['t_coeffs']))
        stoich_data.append(data['stoich'])

    return (
        species,
        heat_capacity,
        gibbs_free_energy,
        stoich_data,
    )


def find_species(elements, charge='neutral', num_atoms=None):
    """
    Find all CEA species that contain the specified properties
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

    Returns
    -------
    species: 1D string array
        List of all species containing the required elements.

    Examples
    --------
    >>> import chemcat.cea as cea
    >>> # Get all sodium-bearing species:
    >>> species = cea.find_species(['Na'])
    >>> print(species)
    ['KNa' 'Na' 'NaCN' 'NaH' 'NaNO2' 'NaNO3' 'NaO' 'NaOH' 'Na2' 'Na2O' 'Na2O2'
     'Na2O2H2']

    >>> # Get species containing exactly two Na atoms:
    >>> species = cea.find_species({'Na':2})
    >>> print(species)
    ['Na2' 'Na2O' 'Na2O2' 'Na2O2H2']

    >>> # Species containing exactly two Na atoms and any amount of oxygen:
    >>> species = cea.find_species({'Na':2, 'O':None})
    >>> print(species)
    ['Na2O' 'Na2O2' 'Na2O2H2']

    >>> # Get all species containing sodium and oxygen (any amount):
    >>> species = cea.find_species(['Na', 'O'])
    >>> print(species)
    ['NaNO2' 'NaNO3' 'NaO' 'NaOH' 'Na2O' 'Na2O2' 'Na2O2H2']

    >>> # Get all hydrogen-ion species:
    >>> H_ions= cea.find_species(['H'], charge='ion')
    >>> print(H_ions)
    ['CH+' 'CH2OH+' 'H+' 'H-' 'HCO+' 'HD+' 'HO2-' 'H2+' 'H2-' 'H2O+' 'H3O+'
     'NH+' 'NH4+' 'NaOH+' 'OH+' 'OH-' 'MgOH+' 'SH-' 'SiH+']

    >>> # Only diatomic Na species:
    >>> diatomic = cea.find_species(['Na'], num_atoms=2, charge='all')
    >>> print(diatomic)
    ['KNa' 'NaH' 'NaO' 'Na2']
    """
    # Turn into dict if needed:
    if not isinstance(elements, dict):
        elements = {e:None for e in elements}

    with open(f'{ROOT}chemcat/data/thermo_build_cea.dat', 'r') as f:
        lines = f.readlines()
    nlines = len(lines)

    species = []
    i = 0
    while i < nlines:
        line = lines[i]
        name = line[0:16].strip()
        line = lines[i+1]
        ndata_intervals = int(line[0:2])
        i += 2 + 3*ndata_intervals

        stoich = {}
        for element in sliced(line[10:50], 8):
            if element[0:2].strip() == '':
                break
            element_name = element[0:2].strip().capitalize()
            if element_name == 'E':
                element_name = 'e'
            stoich[element_name] = float(element[2:])

        is_charged = 'e' in stoich
        if charge == 'neutral' and is_charged:
            continue
        elif charge == 'ion' and not is_charged:
            continue

        if num_atoms is not None:
            n_atoms = sum([val for key,val in stoich.items() if key!='e'])
            if n_atoms != num_atoms:
                continue

        # Request selected stoichiometric values:
        for key, val in elements.items():
            if key not in stoich or (val is not None and stoich[key] != val):
                break
        else:
            species.append(name)

    return np.array(species)

