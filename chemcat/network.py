# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'Network',
    'thermo_eval',
    'read_elemental',
    'set_element_abundance',
    'thermochemical_equilibrium',
    'write_file',
]

import itertools
import sys
import warnings

import numpy as np

from . import janaf
from . import cea
from . import utils as u

sys.path.append(f'{u.ROOT}chemcat/lib')
import _thermo as nr


class Thermo_Prop():
    """
    Descriptor objects that automate setting the elemental abundances.

    To understand this sorcery see:
    https://docs.python.org/3/howto/descriptor.html
    """
    def __set_name__(self, obj, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        value = getattr(obj, self.private_name)
        return value

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)
        has_all_attributes = (
            hasattr(obj, 'metallicity')
            and hasattr(obj, 'e_abundances')
            and hasattr(obj, 'e_scale')
            and hasattr(obj, 'e_ratio')
        )
        if has_all_attributes:
            obj.element_rel_abundance = set_element_abundance(
                obj.elements,
                obj._base_composition,
                obj._base_dex_abundances,
                obj.metallicity, obj.e_abundances,
                obj.e_scale, obj.e_ratio,
            )


class Network(object):
    """A chemcat chemical network object."""
    metallicity = Thermo_Prop()
    e_abundances = Thermo_Prop()
    e_scale = Thermo_Prop()
    e_ratio = Thermo_Prop()

    def __init__(
        self, pressure, temperature, input_species,
        metallicity=0.0,
        e_abundances={},
        e_scale={},
        e_ratio={},
        sources=['janaf', 'cea'],
    ):
        """
        Parameters
        ----------
        pressure: 1D float iterable
            Pressure profile (bar).
        temperature: 1D float iterable
            Temperature profile (Kelvin).
        input_species: 1D string iterable
            List of atomic, molecular, and ionic species to include in the
            chemical network.  Species not found in the database will be
            discarded.
        metallicity: Float
            Scaling factor for all elemental species except H and He.
            Dex units relative to the sun (e.g., solar is metallicity=0.0,
            10x solar is metallicity=1.0).
        e_abundances: Dictionary
            Custom elemental abundances.
            The dict contains the name of the element and their custom
            abundance in dex units relative to H=12.0.
            These values override metallicity.
        e_scale: Dictionary
            Custom elemental abundances scaled relative to solar values.
            The dict contains the name of the element and their custom
            scaling factor in dex units, e.g., for 2x solar carbon set
            e_scale = {'C': np.log10(2.0)}.
            This argument modifies the abundances on top of any custom
            metallicity and e_abundances.
        e_ratio: Dictionary
            Custom elemental abundances scaled relative to another element.
            The dict contains the pair of elements joined by an underscore
            and their ratio in dex units, e.g., for a C/O ratio of 0.8 set
            e_ratio = {'C_O': np.log10(0.8)}.
            These values modify the abundances on top of any custom
            metallicity, e_abundances, and e_scale.
        sources: List of strings
            Name of databases where to get the thermochemical properties
            (in order of priority).  Available options: 'janaf' or 'cea'.

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

        >>> # Compute abundances in thermochemical equilibrium:
        >>> vmr = net.thermochemical_equilibrium()
        >>> species = list(net.species)

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

        >>> plt.figure('Plotty McPlt dot plot', (8,4.5))
        >>> plt.clf()
        >>> plt.subplots_adjust(0.09, 0.12, 0.89, 0.94)
        >>> ax = plt.subplot(111)
        >>> for name in species:
        >>>     plt.loglog(
        >>>         vmr[:,species.index(name)], pressure,
        >>>         label=name, lw=2.0, color=cols[name])
        >>> plt.ylim(np.amax(pressure), np.amin(pressure))
        >>> plt.xlim(1e-30, 2)
        >>> plt.ylabel('Pressure (bar)', fontsize=12)
        >>> plt.legend(loc=(1.01,0.01), fontsize=9.0)
        >>> plt.xlabel('Volume mixing ratio', fontsize=12)
        >>> plt.title(f'{temperature[0]} K', fontsize=12)

        >>> # Compute heat capacity:
        >>> cp = net.heat_capacity()
        >>> print(f'Heat capacity (cp/R):\n{cp[0]}')
        Heat capacity (cp/R):
        [5.26408044 9.48143057 4.11030773 6.77638503 7.34238673 4.05594463
         3.72748083 6.3275286  3.79892261 2.49998117 2.49998117 2.50082308
         2.49998117 2.51092596]
        """
        self.pressure = pressure
        self.temperature = temperature
        self.input_species = input_species

        dealiased_species = u.de_aliasing(input_species, sources)
        source_names = u.resolve_sources(dealiased_species, sources)

        idx_valid = source_names != None
        self.provenance = source_names[idx_valid]
        self.species = np.array(input_species)[idx_valid]

        nspecies = len(self.species)
        self._heat_capacity = np.zeros(nspecies, object)
        self._gibbs_free_energy = np.zeros(nspecies, object)
        stoich_data = np.tile(None, nspecies)

        for source in np.unique(self.provenance):
            if source == 'janaf':
                setup_network = janaf.setup_network
            if source == 'cea':
                setup_network = cea.setup_network
            species = np.array(dealiased_species)[source_names==source]
            network_data = setup_network(species)

            idx_db = self.provenance == source
            self._heat_capacity[idx_db] = network_data[1]
            self._gibbs_free_energy[idx_db] = network_data[2]
            stoich_data[idx_db] = network_data[3]

        self.elements, self.stoich_vals = u.stoich_matrix(stoich_data)

        self.element_file = \
            f'{u.ROOT}chemcat/data/asplund_2021_solar_abundances.dat'
        base_data = read_elemental(self.element_file)
        self._base_composition = base_data[0]
        self._base_dex_abundances = base_data[1]

        self.metallicity = metallicity
        self.e_abundances = e_abundances
        self.e_scale = e_scale
        self.e_ratio = e_ratio

        self.element_rel_abundance = set_element_abundance(
            self.elements,
            self._base_composition,
            self._base_dex_abundances,
            self.metallicity,
            self.e_abundances,
            self.e_scale,
            self.e_ratio,
        )


    def heat_capacity(self, temperature=None):
        if temperature is None:
            temperature = self.temperature
        return thermo_eval(temperature, self._heat_capacity)


    def gibbs_free_energy(self, temperature=None):
        if temperature is None:
            temperature = self.temperature
        return thermo_eval(temperature, self._gibbs_free_energy)


    def thermochemical_equilibrium(
        self, temperature=None,
        metallicity=None,
        e_abundances=None,
        e_scale=None,
        e_ratio=None,
        savefile=None,
    ):
        """
        Compute thermochemical-equilibrium abundances, updating the
        atmospheric properties if requested.
        """
        if temperature is not None:
            if np.shape(temperature) != np.shape(self.pressure):
                raise ValueError(
                    'Temperature profile does not match size of pressure '
                    f'profile ({len(self.pressure)} layers)'
                )
            self.temperature = temperature
        if metallicity is not None:
            self.metallicity = metallicity
        if e_abundances is not None:
            self.e_abundances = e_abundances
        if e_scale is not None:
            self.e_scale = e_scale
        if e_ratio is not None:
            self.e_ratio = e_ratio

        self.vmr = thermochemical_equilibrium(
            self.pressure,
            self.temperature,
            self.element_rel_abundance,
            self.stoich_vals,
            self._gibbs_free_energy,
        )

        if savefile is not None:
            write_file(
                savefile,
                self.species, self.pressure, self.temperature,
                self.vmr,
            )

        return self.vmr


def thermo_eval(temperature, thermo_funcs):
    r"""
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
    >>> import chemcat as cat
    >>> import chemcat.janaf as janaf
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
    >>> plt.figure('Heat capacity, Gibbs free energy', (8.5, 4.5))
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
    >>> plt.legend(loc='upper right', fontsize=8)
    >>> plt.xlabel('Temperature (K)')
    >>> plt.ylabel('Gibbs free energy / RT')
    >>> plt.tight_layout()
    """
    temp = np.atleast_1d(temperature)
    ntemp = np.shape(temp)[0]
    nspecies = len(thermo_funcs)
    thermo_prop = np.zeros((ntemp, nspecies))
    for j in range(nspecies):
        thermo_prop[:,j] = thermo_funcs[j](temp)
    if np.shape(temperature) == ():
        return np.squeeze(thermo_prop)
    return thermo_prop


def read_elemental(element_file):
    """
    Extract elemental abundances from a file (defaulted to a solar
    elemental abundance file from Asplund et al. 2021).
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
    >>> import chemcat.utils as u

    >>> element_file = f'{u.ROOT}chemcat/data/asplund_2021_solar_abundances.dat'
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
        metallicity=0.0, e_abundances={}, e_scale={}, e_ratio={},
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
        Set custom elemental abundances.
        The dict contains the name of the element and their custom
        abundance in dex units relative to H=12.0.
        These values (if any) override metallicity.
    e_scale: Dictionary of element-scaling pairs
        Set custom elemental abundances by scaling from its solar value.
        The dict contains the name of the element and their custom
        scaling factor in dex units, e.g., for 2x solar carbon set
        e_scale = {'C': np.log10(2.0)}.
        This argument modifies the abundances on top of any custom
        metallicity and e_abundances.
    e_ratio: Dictionary of element-ratio pairs
        Set custom elemental abundances by scaling relative to another
        element.
        The dict contains the pair of elements joined by an underscore
        and their ratio in dex units, e.g., for a C/O ratio of 0.8 set
        e_ratio = {'C_O': np.log10(0.8)}.
        These values scale on top of any custom metallicity,
        e_abundances, and e_scale.

    Returns
    -------
    elemental_abundances: 1D float array
        Elemental volume mixing ratios relative to H=1.0.

    Examples
    --------
    >>> import chemcat as cat
    >>> import chemcat.utils as u
    >>> element_file = f'{u.ROOT}chemcat/data/asplund_2021_solar_abundances.dat'
    >>> sun_elements, sun_dex = cat.read_elemental(element_file)
    >>> elements = 'H He C N O'.split()

    >>> solar = cat.set_element_abundance(
    >>>     elements, sun_elements, sun_dex,
    >>> )

    >>> # Set custom metallicity to [M/H] = 0.5:
    >>> abund = cat.set_element_abundance(
    >>>     elements, sun_elements, sun_dex, metallicity=0.5,
    >>> )
    >>> print([f'{e}: {q:.1e}' for e,q in zip(elements, abund)])
    ['H: 1.0e+00', 'He: 8.2e-02', 'C: 9.1e-04', 'N: 2.1e-04', 'O: 1.5e-03']

    >>> # Custom carbon abundance by direct value (dex):
    >>> abund = cat.set_element_abundance(
    >>>     elements, sun_elements, sun_dex, e_abundances={'C': 8.8},
    >>> )
    >>> print([f'{e}: {q:.1e}' for e,q in zip(elements, abund)])
    ['H: 1.0e+00', 'He: 8.2e-02', 'C: 6.3e-04', 'N: 6.8e-05', 'O: 4.9e-04']

    >>> # Custom carbon abundance by scaling to 2x its solar value:
    >>> abund = cat.set_element_abundance(
    >>>     elements, sun_elements, sun_dex, e_scale={'C': np.log10(2)},
    >>> )
    >>> print([f'{e}: {q:.1e}' for e,q in zip(elements, abund)])
    ['H: 1.0e+00', 'He: 8.2e-02', 'C: 5.8e-04', 'N: 6.8e-05', 'O: 4.9e-04']

    >>> # Custom carbon abundance by scaling to C/O = 0.8:
    >>> abund = cat.set_element_abundance(
    >>>     elements, sun_elements, sun_dex, e_ratio={'C_O': np.log10(0.8)},
    >>> )
    >>> print([f'{e}: {q:.1e}' for e,q in zip(elements, abund)])
    ['H: 1.0e+00', 'He: 8.2e-02', 'C: 3.9e-04', 'N: 6.8e-05', 'O: 4.9e-04']
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

    # Scale custom elemental abundances (additive to metallicity):
    for element, fscale in e_scale.items():
        elemental_abundances[np.array(elements) == element] += fscale

    # Set custom elemental ratios:
    for element, log_ratio in e_ratio.items():
        element1, element2 = element.split('_')
        idx1 = np.array(elements) == element1
        idx2 = np.array(elements) == element2
        elemental_abundances[idx1] = elemental_abundances[idx2] + log_ratio

    # Convert elemental log VMR (relative to H=12.0) to VMR (rel. to H=1.0):
    elemental_abundances = 10**(elemental_abundances-12.0)
    elemental_abundances[np.array(elements) == 'e'] = 0.0
    return elemental_abundances


def thermochemical_equilibrium(
        pressure, temperature, element_rel_abundance, stoich_vals,
        gibbs_funcs,
    ):
    """
    Compute thermochemical equilibrium for the given chemical network
    at the specified temperature--pressure profile.

    Parameters
    ----------
    pressure: 1D float array
        Pressure profile (bar).
    temperature: 1D float array
        Temperature profile (Kelvin).
    element_rel_abundance: 1D float array
        Elemental abundances (relative to H=1.0).
    stoich_vals: 2D float array
        Species stoichiometric values for CHON.
    gibbs_funcs: 1D iterable of callable functions
        Functions that return the Gibbs free energy (divided by RT)
        for each species in the network.

    Returns
    -------
    vmr: 2D float array
        Species volume mixing ratios in thermochemical equilibrium
        of shape [nlayers, nspecies].
    """
    nlayers = len(pressure)
    nspecies, nelements = np.shape(stoich_vals)

    # Target elemental fractions (and charge) for conservation:
    b0 = element_rel_abundance / np.sum(element_rel_abundance)
    # Total elemental abundance:
    total_abundance = np.sum(b0)

    is_atom = element_rel_abundance > 0.0
    # Maximum abundance reachable by each species
    # (i.e., species takes all available atoms)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        max_abundances = np.array([
            np.nanmin((b0/is_atom)[stoich>0] / stoich[stoich>0])
            for stoich in stoich_vals])

    total_natoms = np.sum(stoich_vals*is_atom, axis=1)
    electron_index = total_natoms == 0
    max_abundances[electron_index] = total_abundance

    # Initial guess for abundances of species, gets modified in-place
    # with best-fit values by nr.gibbs_energy_minimizer()
    abundances = np.copy(max_abundances)

    # Number of conservation equations to solve with Newton-Raphson
    nequations = nelements + 1
    pilag = np.zeros(nelements)  # pi lagrange multiplier
    x = np.zeros(nequations)

    vmr = np.zeros((nlayers, nspecies))
    mu = np.zeros(nspecies)  # chemical potential/RT
    h_ts = thermo_eval(temperature, gibbs_funcs).T + np.log(pressure)
    dlnns = np.zeros(nspecies)
    tolx = tolf = 2.22e-16

    # Compute thermochemical equilibrium abundances at each layer:
    # (Go down the layers and then sweep back in case the first run
    # didn't get the global minimum)
    for i in itertools.chain(range(nlayers), reversed(range(nlayers))):
        abundances[abundances <= 0] = 1e-300
        exit_status = nr.gibbs_energy_minimizer(
            nspecies, nequations, stoich_vals, b0,
            temperature[i], h_ts[:,i], pilag,
            abundances, max_abundances, total_abundance,
            mu, x, dlnns, tolx, tolf)

        if exit_status == 1:
            print(f"Gibbs minimization failed at layer {i}")
        vmr[i] = abundances / np.sum(abundances[~electron_index])

    return vmr


def write_file(file, species, pressure, temperature, vmr):
    """
    Write results to file.

    Parameters
    ----------
    file: String
        Output file name.
    species: 1D string iterable
        Names of atmospheric species.
    pressure: 1D float iterable
        Atmospheric pressure profile (bar).
    temperature: 1D float iterable
        Atmospheric temperature profile (kelvin).
    vmr: 2D float iterable
        Atmospheric volume mixing ratios (of shape [nlayers,nspecies]).

    Examples
    --------
    >>> import chemcat as cat
    >>> import numpy as np

    >>> nlayers = 81
    >>> temperature = np.tile(1200.0, nlayers)
    >>> pressure = np.logspace(-8, 3, nlayers)
    >>> molecules = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH H He C N O'.split()
    >>> net = cat.Network(pressure, temperature, molecules)
    >>> vmr = net.thermochemical_equilibrium()
    >>> cat.write_file(
    >>>     'output_file.dat', net.species, pressure, temperature, vmr,
    >>> )
    """
    fout = open(file, 'w+')

    # Header info:
    fout.write(
        '# TEA output file with abundances calculated in '
        'thermochemical equilibrium.\n'
        '# Units: pressure (bar), temperature (K), abundance '
        '(volume mixing ratio).\n\n')

    # List of species:
    fout.write('#SPECIES\n')
    fout.write(' '.join(spec for spec in species))
    fout.write('\n\n')

    # Atmospheric data:
    fout.write('#TEADATA\n')
    fout.write('#Pressure   Temp     ')
    fout.write(''.join(f'{spec:<11}' for spec in species) + '\n')

    nlayers = len(pressure)
    for i in range(nlayers):
        fout.write(f'{pressure[i]:.4e}  {temperature[i]:7.2f} ')
        for abund in vmr[i]:
            fout.write(f'{abund:11.4e}')
        fout.write('\n')
    fout.close()

