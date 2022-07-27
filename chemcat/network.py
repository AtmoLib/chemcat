# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'Thermo_Prop',
    'Network',
    'thermo_eval',
    'thermochemical_equilibrium',
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
            obj.element_rel_abundance = u.set_element_abundance(
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
        e_source='asplund_2021',
        sources=['janaf', 'cea'],
    ):
        r"""
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
            and their ratio, e.g., for a C/O ratio of 0.8 set
            e_ratio = {'C_O': 0.8}.
            These values modify the abundances on top of any custom
            metallicity, e_abundances, and e_scale.
        e_source: String
            Source of elemental composition used as base composition.
            If value is 'asplund_2021' or 'asplund_2009', adopt the
            solar elemental composition values from Asplund+2009 or
            Asplund+2021, respectively.
            Else, assume e_source is a path to a custom elemental
            composition file.
        sources: List of strings
            Name of databases where to get the thermochemical properties
            (in order of priority).  Available options: 'janaf' or 'cea'.

        Examples
        --------
        >>> import chemcat as cat
        >>> import chemcat.utils as u
        >>> import numpy as np

        >>> nlayers = 81
        >>> temperature = np.tile(1200.0, nlayers)
        >>> pressure = np.logspace(-8, 3, nlayers)
        >>> molecules = (
        >>>     'H2O CH4 CO CO2 NH3 N2 H2 HCN OH C2H2 C2H4 H He C N O').split()
        >>> net = cat.Network(pressure, temperature, molecules)

        >>> # Compute abundances in thermochemical equilibrium:
        >>> vmr = net.thermochemical_equilibrium()

        >>> # See results:
        >>> ax = u.plot_vmr(pressure, vmr, net.species, vmr_range=(1e-30,2))

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

        if None in source_names:
            missing_species = np.array(input_species)[~idx_valid]
            print(
                'These input species were not found in any database:'
                f'\n  {missing_species}'
            )

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

        # Setup input elemental composition:
        if e_source == 'asplund_2009' or e_source == 'asplund_2021':
            e_source = f'{u.ROOT}chemcat/data/{e_source}_solar_abundances.dat'

        self.element_file = e_source
        base_data = u.read_elemental(self.element_file)
        self._base_composition = base_data[0]
        self._base_dex_abundances = base_data[1]

        self.metallicity = metallicity
        self.e_abundances = e_abundances
        self.e_scale = e_scale
        self.e_ratio = e_ratio

        self.element_rel_abundance = u.set_element_abundance(
            self.elements,
            self._base_composition,
            self._base_dex_abundances,
            self.metallicity,
            self.e_abundances,
            self.e_scale,
            self.e_ratio,
        )


    def heat_capacity(self, temperature=None):
        """
        Evaluate the heat capacity of each species in the network
        at the given temperature (default to self.temperature if needed).
        """
        if temperature is None:
            temperature = self.temperature
        return thermo_eval(temperature, self._heat_capacity)


    def gibbs_free_energy(self, temperature=None):
        """
        Evaluate the Gibbs free energy of each species in the network
        at the given temperature (default to self.temperature if needed).
        """
        if temperature is None:
            temperature = self.temperature
        return thermo_eval(temperature, self._gibbs_free_energy)


    def thermochemical_equilibrium(
        self,
        temperature=None,
        metallicity=None,
        e_abundances=None,
        e_scale=None,
        e_ratio=None,
        savefile=None,
    ):
        """
        Compute thermochemical-equilibrium abundances, update the
        atmospheric properties according to any non-None argument.

        Parameters
        ----------
        temperature: 1D float iterable
            Temperature profile (Kelvin).
            Must have same number of layers as self.pressure.
        metallicity: Float
            Scaling factor for all elemental species except H and He.
            Dex units relative to the sun (e.g., solar is metallicity=0.0,
            10x solar is metallicity=1.0).
        e_abundances: Dictionary
            Elemental abundances for custom species set as
            {element: abundance} pairs in dex units relative to H=12.0.
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
            and their ratio, e.g., for a C/O ratio of 0.8 set
            e_ratio = {'C_O': 0.8}.
            These values modify the abundances on top of any custom
            metallicity, e_abundances, and e_scale.
        savefile: String
            If not None, store vmr outputs to given file path.

        Returns
        -------
        vmr: 2D float array
            Species volume mixing ratios in thermochemical equilibrium
            of shape [nlayers, nspecies].
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
            u.write_file(
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
    >>> import chemcat.utils as u
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> molecules = (
    >>>     'H2O CH4 CO CO2 NH3 N2 H2 HCN OH C2H2 C2H4 H He C N O'.split()
    >>> janaf_data = janaf.setup_network(molecules)
    >>> species = janaf_data[0]
    >>> heat_funcs = janaf_data[1]
    >>> gibbs_funcs = janaf_data[2]

    >>> temperature = 1500.0
    >>> temperatures = np.arange(100.0, 4501.0, 10)
    >>> cp1 = cat.thermo_eval(temperature, heat_funcs)
    >>> cp2 = cat.thermo_eval(temperatures, heat_funcs)
    >>> gibbs = cat.thermo_eval(temperatures, gibbs_funcs)

    >>> nspecies = len(species)
    >>> plt.figure('Heat capacity, Gibbs free energy', (8.5, 4.5))
    >>> plt.clf()
    >>> plt.subplot(121)
    >>> for j in range(nspecies):
    >>>     label = species[j]
    >>>     plt.plot(
    >>>         temperatures, cp2[:,j], label=label, c=u.COLOR_DICT[label],
    >>>     )
    >>> plt.xlim(np.amin(temperatures), np.amax(temperatures))
    >>> plt.plot(np.tile(temperature,nspecies), cp1, 'ob', ms=4, zorder=-1)
    >>> plt.xlabel('Temperature (K)')
    >>> plt.ylabel('Heat capacity / R')

    >>> plt.subplot(122)
    >>> for j in range(nspecies):
    >>>     label = species[j]
    >>>     plt.plot(
    >>>         temperatures, gibbs[:,j], label=label, c=u.COLOR_DICT[label],
    >>>     )
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
    # Total elemental abundance as first guess for total species abundance:
    total_abundance = np.sum(b0)

    is_atom = element_rel_abundance > 0.0
    # Maximum abundance reachable by each species
    # (i.e., species takes all available atoms)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        max_abundances = np.array([
            np.nanmin((b0/is_atom)[stoich>0] / stoich[stoich>0])
            for stoich in stoich_vals
        ])

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
    delta_ln_vmr = np.zeros(nspecies)
    tolx = tolf = 2.22e-16

    # Compute thermochemical equilibrium abundances at each layer:
    # (Go down the layers and then sweep back up in case the first run
    # didn't get the global minimum)
    for i in itertools.chain(range(nlayers), reversed(range(nlayers))):
        abundances[abundances <= 0] = 1e-300
        exit_status = nr.gibbs_energy_minimizer(
            nspecies, nequations, stoich_vals, b0,
            temperature[i], h_ts[:,i], pilag,
            abundances, max_abundances, total_abundance,
            mu, x, delta_ln_vmr, tolx, tolf,
        )

        if exit_status == 1:
            print(f"Gibbs minimization failed at layer {i}")
        vmr[i] = abundances / np.sum(abundances[~electron_index])

    return vmr

