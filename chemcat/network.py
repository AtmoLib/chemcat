# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'ROOT',
    'Network',
    'thermo_eval',
    'read_elemental',
    'set_element_abundance',
]

import os
from pathlib import Path
import warnings

import numpy as np

from . import janaf


ROOT = str(Path(__file__).parents[1]) + os.path.sep


class Network(object):
    r"""
    A chemcat chemical network object.

    Examples
    --------
    >>> import chemcat as cat
    >>> import numpy as np

    >>> nlayers = 81
    >>> temperature = np.tile(1200.0, nlayers)
    >>> pressure = np.logspace(-8, 3, nlayers)
    >>> HCNO_molecules = (
    >>>     'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O').split()
    >>> net = cat.Network(pressure, temperature, HCNO_molecules)

    >>> # Compute heat capacity at current temperature profile:
    >>> cp = net.heat_capacity()
    >>> print(f'Heat capacity (cp/R):\n{cp[0]}')
    Heat capacity (cp/R):
    [5.26408044 9.48143057 4.11030773 6.77638503 7.34238673 4.05594463
     3.72748083 6.3275286  3.79892261 2.49998117 2.49998117 2.50082308
     2.49998117 2.51092596]

    >>> # Compute heat capacity at updated temperature profile:
    >>> temp2 = np.tile(700.0, nlayers)
    >>> cp2 = net.heat_capacity(temp2)
    >>> print(
    >>>     f'Temperature: {net.temperature[0]} K\n'
    >>>     f'Heat capacity (cp/R):\n{cp2[0]}')
    Temperature: 700.0 K
    Heat capacity (cp/R):
    [4.50961195 6.95102049 3.74900958 5.96117901 5.81564946 3.69885601
     3.5409384  5.4895911  3.56763887 2.49998117 2.49998117 2.50106362
     2.49998117 2.53053035]
    """
    def __init__(
        self, pressure, temperature, input_species, source='janaf',
    ):
        self.pressure = pressure
        self.temperature = temperature
        self.input_species = input_species

        if source == 'janaf':
            network_data = janaf.setup_network(input_species)
        self.species = network_data[0]
        self.elements = network_data[1]
        self._heat_capacity = network_data[2]
        self._gibbs_free_energy = network_data[3]
        self.stoich_vals = network_data[4]

        self.element_file = f'{ROOT}chemcat/data/abundances.txt'
        base_data = read_elemental(self.element_file)
        self._base_composition = base_data[0]
        self._base_dex_abundances = base_data[1]

        self.element_rel_abundance = set_element_abundance(
            self.elements,
            self._base_composition,
            self._base_dex_abundances,
        )


    def heat_capacity(self, temperature=None):
        if temperature is None:
            temperature = self.temperature
        return thermo_eval(temperature, self._heat_capacity)


    def gibbs_free_energy(self, temperature=None):
        if temperature is None:
            temperature = self.temperature
        return thermo_eval(temperature, self._gibbs_free_energy)


def thermo_eval(temperature, thermo_funcs):
    """
    Compute the thermochemical property specified by thermo_func at
    at the requested temperature(s).  These can be, e.g., the
    heat_capacity or gibbs_free_energy functions returned by
    setup_network().

    Parameters
    ----------
    temperature: float or 1D float iterable
        Temperature (Kelvin).
    thermo_funcs: 1D iterable of callable functions
        Functions that return the thermochemical property.

    Returns
    -------
    thermo_prop: 1D or 2D float array
        The provided thermochemical property evaluated at the requested
        temperature(s).
        The shape of the output depends on the shape of the
        temperature input.

    Examples
    --------
    >>> # (First, make sure you added the path to the TEA package)
    >>> import chemcat as cat
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    >>> janaf_data = janaf.setup_network(molecules)
    >>> species = janaf_data[0]
    >>> heat_funcs = janaf_data[2]
    >>> gibbs_funcs = janaf_data[3]

    >>> temperature = 1500.0
    >>> temperatures = np.arange(100.0, 4501.0, 10)
    >>> cp1 = cat.thermo_eval(temperature, heat_funcs)
    >>> cp2 = cat.thermo_eval(temperatures, heat_funcs)
    >>> gibbs = cat.thermo_eval(temperatures, gibbs_funcs)

    >>> cols = {
    >>>     'H': 'blue',
    >>>     'H2': 'deepskyblue',
    >>>     'He': 'olive',
    >>>     'H2O': 'navy',
    >>>     'CH4': 'orange',
    >>>     'CO': 'limegreen',
    >>>     'CO2': 'red',
    >>>     'NH3': 'magenta',
    >>>     'HCN': '0.55',
    >>>     'N2': 'gold',
    >>>     'OH': 'steelblue',
    >>>     'C': 'salmon',
    >>>     'N': 'darkviolet',
    >>>     'O': 'greenyellow',
    >>> }

    >>> nspecies = len(species)
    >>> plt.figure('Heat capacity', (6.5, 4.5))
    >>> plt.clf()
    >>> plt.subplot(121)
    >>> for j in range(nspecies):
    >>>     label = species[j]
    >>>     plt.plot(temperatures, cp2[:,j], label=label, c=cols[label])
    >>> plt.xlim(np.amin(temperatures), np.amax(temperatures))
    >>> plt.plot(np.tile(temperature,nspecies), cp1, 'ob', ms=4, zorder=-1)
    >>> plt.xlabel('Temperature (K)')
    >>> plt.ylabel('Heat capacity / R')

    >>> plt.subplot(122)
    >>> for j in range(nspecies):
    >>>     label = species[j]
    >>>     plt.plot(temperatures, gibbs[:,j], label=label, c=cols[label])
    >>> plt.xlim(np.amin(temperatures), np.amax(temperatures))
    >>> plt.legend(loc=(1.01, 0.01), fontsize=8)
    >>> plt.xlabel('Temperature (K)')
    >>> plt.ylabel('Gibbs free energy / RT')
    >>> plt.tight_layout()
    """
    temp = np.atleast_1d(temperature)
    ntemp = np.shape(temp)[0]
    nspecies = len(thermo_funcs)
    thermo_prop= np.zeros((ntemp, nspecies))
    for j in range(nspecies):
        thermo_prop[:,j] = thermo_funcs[j](temp)
    if np.shape(temperature) == ():
        return thermo_prop[0]
    return thermo_prop


def read_elemental(element_file):
    """
    Extract elemental abundances from a file (defaulted to a solar
    elemental abundance file from Asplund 2009).
    Inputs
    ------
    element_file: String
        Path to a file containing a list of elements (second column)
        and their relative abundances in log10 scale relative to H=12.0
        (third column).
    Returns
    -------
    elements: 1D string array
        The list of elements.
    dex_abundances: 1D float array
        The elemental abundances in dex units relative to H=12.0.
    Examples
    --------
    >>> import chemcat as cat
    >>> # Asplund et al. (2009) solar elemental composition:
    >>> element_file = f'{cat.ROOT}chemcat/data/abundances.txt'
    >>> elements, dex = cat.read_elemental(element_file)
    >>> for e in 'H He C N O'.split():
    >>>     print(f'{e:2}:  {dex[elements==e][0]:6.3f}')
    H :  12.000
    He:  10.914
    C :   8.460
    N :   7.830
    O :   8.690
    """
    elements, dex = np.loadtxt(
        element_file, dtype=str, comments='#', usecols=(1,2), unpack=True)
    dex_abundances = np.array(dex, float)
    return elements, dex_abundances



def set_element_abundance(
        elements, base_composition, base_dex_abundances,
        metallicity=0.0, e_abundances={},
    ):
    """
    Set an elemental composition by scaling metals and custom atomic species.

    Parameters
    ----------
    elements: 1D string array
        List of elements to return their abundances.
    base_composition: 1D float array
        List of all possible elements.
    base_dex_abundances: 1D float iterable
        The elemental base abundances in dex units relative to H=12.0.
    metallicity: Float
        Scaling factor for all elemental species except H and He.
        Dex units relative to the sun (e.g., solar is metallicity=0.0,
        10x solar is metallicity=1.0).
    e_abundances: Dictionary of element-abundance pairs
        Custom elemental abundances (dex, relative to H=12.0) for
        specific atoms.  These values (if any) override metallicity.

    Returns
    -------
    elemental_abundances: 1D float array

    Examples
    --------
    >>> import chemcat as cat
    >>> # Asplund et al. (2009) solar elemental composition:
    >>> element_file = f'{cat.ROOT}chemcat/data/abundances.txt'
    >>> sun_elements, sun_dex = cat.read_elemental(element_file)
    >>> elements = 'H He C N O'.split()
    >>> solar = cat.set_element_abundance(
            elements, sun_elements, sun_dex)
    >>> heavy = cat.set_element_abundance(
            elements, sun_elements, sun_dex, metallicity=0.5)
    >>> carbon = cat.set_element_abundance(
            elements, sun_elements, sun_dex, e_abundances={'C': 8.8})
    >>> for i in range(len(elements)):
    >>>     print(
    >>>         f'{elements[i]:2}:  '
    >>>         f'{solar[i]:.1e}  {heavy[i]:.1e}  {carbon[i]:.1e}')
    H :  1.0e+00  1.0e+00  1.0e+00
    He:  8.5e-02  8.5e-02  8.5e-02
    C :  2.7e-04  8.5e-04  6.3e-04
    N :  6.8e-05  2.1e-04  6.8e-05
    O :  4.9e-04  1.5e-03  4.9e-04
    """
    nelements = len(elements)
    elemental_abundances = np.zeros(nelements)
    for j in range(nelements):
        if elements[j] == 'e':
            continue
        idx = list(base_composition).index(elements[j])
        elemental_abundances[j] = base_dex_abundances[idx]

    # Scale the metals' abundances:
    imetals = np.isin(elements, 'H He D'.split(), invert=True)
    elemental_abundances[imetals] += metallicity
    # Set custom elemental abundances:
    for element, abundance in e_abundances.items():
        elemental_abundances[np.array(elements) == element] = abundance

    # Convert elemental log VMR (relative to H=12.0) to VMR (rel. to H=1.0):
    elemental_abundances = 10**(elemental_abundances-12.0)
    elemental_abundances[elements == 'e'] = 0.0
    return elemental_abundances


def thermo_eval(temperature, thermo_func):
    r"""
    Compute the thermochemical property specified by thermo_func at
    at the requested temperature(s).  These can be, e.g., the
    heat_capacity or gibbs_free_energy functions returned by
    setup_network().

    Parameters
    ----------
    temperature: float or 1D float iterable
        Temperature (Kelvin).
    cp_splines: 1D iterable of heat-capacity numpy splines
        Numpy splines containing heat capacity info for species.

    Returns
    -------
    cp: 1D or 2D float ndarray
        The heat capacity (divided by the universal gas constant, R) for
        each species at the requested temperature(s).
        The shape of the output depends on the shape of the temperature input.

    Examples
    --------
    >>> import chemcat as cat
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    >>> janaf_data = janaf.setup_network(molecules)
    >>> species = janaf_data[0]
    >>> heat_funcs = janaf_data[2]

    >>> temperature = 1500.0
    >>> temperatures = np.arange(100.0, 4501.0, 10)
    >>> cp1 = cat.thermo_eval(temperature, heat_funcs)
    >>> cp2 = cat.thermo_eval(temperatures, heat_funcs)
    >>> cols = {
    >>>     'H': 'blue',
    >>>     'H2': 'deepskyblue',
    >>>     'He': 'olive',
    >>>     'H2O': 'navy',
    >>>     'CH4': 'orange',
    >>>     'CO': 'limegreen',
    >>>     'CO2': 'red',
    >>>     'NH3': 'magenta',
    >>>     'HCN': '0.55',
    >>>     'N2': 'gold',
    >>>     'OH': 'steelblue',
    >>>     'C': 'salmon',
    >>>     'N': 'darkviolet',
    >>>     'O': 'greenyellow',
    >>> }

    >>> nspecies = len(species)
    >>> plt.figure('heat capacity')
    >>> plt.clf()
    >>> for j in range(nspecies):
    >>>     label = species[j]
    >>>     plt.plot(temperatures, cp2[:,j], label=label, c=cols[label])
    >>> plt.xlim(np.amin(temperatures), np.amax(temperatures))
    >>> plt.plot(np.tile(temperature,nspecies), cp1, 'ob', ms=4, zorder=-1)
    >>> plt.legend(loc=(1.01, 0.01), fontsize=8)
    >>> plt.xlabel('Temperature (K)')
    >>> plt.ylabel('Heat capacity / R')
    >>> plt.tight_layout()
    """
    temp = np.atleast_1d(temperature)
    ntemp = np.shape(temp)[0]
    nspecies = len(thermo_func)

    thermo_prop = np.zeros((ntemp, nspecies))
    for j in range(nspecies):
        thermo_prop[:,j] = thermo_func[j](temp)
    if np.shape(temperature) == ():
        return thermo_prop[0]
    return thermo_prop
