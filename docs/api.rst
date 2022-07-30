API
===


chemcat
_______


.. py:module:: chemcat

.. py:class:: Thermo_Prop()

    .. code-block:: pycon

        Descriptor objects that automate setting the elemental abundances.

        To understand this sorcery see:
        https://docs.python.org/3/howto/descriptor.html


        Initialize self.  See help(type(self)) for accurate signature.

.. py:class:: Network(pressure, temperature, input_species, metallicity=0.0, e_abundances={}, e_scale={}, e_ratio={}, e_source='asplund_2021', sources=['janaf', 'cea'])

    .. code-block:: pycon

        A chemcat chemical network object.


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

    .. py:method:: gibbs_free_energy(temperature=None)
    .. code-block:: pycon

        Evaluate the Gibbs free energy of each species in the network
        at the given temperature (default to self.temperature if needed).

    .. py:method:: heat_capacity(temperature=None)
    .. code-block:: pycon

        Evaluate the heat capacity of each species in the network
        at the given temperature (default to self.temperature if needed).

    .. py:method:: thermochemical_equilibrium(temperature=None, metallicity=None, e_abundances=None, e_scale=None, e_ratio=None, savefile=None)
    .. code-block:: pycon

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


chemcat.cea
___________


.. py:module:: chemcat.cea

.. py:function:: is_in(species, thermo_file=None)
.. code-block:: pycon

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

.. py:function:: read_thermo_build(species, thermo_file=None)
.. code-block:: pycon

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

.. py:function:: heat_func(a_coeffs, t_coeffs)
.. code-block:: pycon

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

.. py:function:: gibbs_func(a_coeffs, b_coeffs, t_coeffs)
.. code-block:: pycon

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

.. py:function:: setup_network(input_species)
.. code-block:: pycon

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

.. py:function:: find_species(elements, charge='neutral', num_atoms=None)
.. code-block:: pycon

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


chemcat.janaf
_____________


.. py:module:: chemcat.janaf

.. py:function:: is_in(species)
.. code-block:: pycon

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

.. py:function:: get_filenames(species)
.. code-block:: pycon

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

.. py:function:: read_file(janaf_file)
.. code-block:: pycon

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

    >>> temps, heat = janaf.read_file(janaf_file)
    >>> for i in range(5):
    >>>     print(f'{temps[i]:6.2f}  {heat[i]:.3f}')
    298.15  2.500  -2.523
    300.00  2.500  -2.523
    350.00  2.500  -2.554
    400.00  2.500  -2.621
    450.00  2.500  -2.709

.. py:function:: read_stoich(species=None, janaf_file=None, formula=None)
.. code-block:: pycon

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

.. py:function:: setup_network(input_species)
.. code-block:: pycon

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

.. py:function:: find_species(elements, charge='neutral', num_atoms=None, state='gas')
.. code-block:: pycon

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


chemcat.utils
_____________


.. py:module:: chemcat.utils

.. py:data:: ROOT
.. code-block:: pycon

  '/Users/pato/Dropbox/IWF/projects/2022_chemcat/chemcat/'

.. py:data:: COLORS
.. code-block:: pycon

  ['royalblue', 'darkorange', 'red', 'darkgreen', 'magenta', 'blue', 'limegreen', 'gold', 'dimgray', 'navy', 'deepskyblue', 'silver', 'black', 'olive', 'chocolate', 'skyblue', 'darkviolet', 'greenyellow', 'pink', 'coral', 'darkcyan', 'rosybrown', 'cornflowerblue', 'mediumvioletred', 'maroon', 'darkgoldenrod', 'darkkhaki', 'hotpink', 'darkslateblue', 'lightgreen', 'yellowgreen', 'seagreen', 'yellow', 'slateblue', 'sienna', 'peachpuff', 'orangered', 'goldenrod', 'brown', 'khaki', 'saddlebrown', 'mediumseagreen', 'darksalmon', 'cadetblue', 'mediumaquamarine', 'darkslategray', 'lightsteelblue', 'indigo', 'lightcoral', 'lightslategray', 'lawngreen', 'lightblue', 'darkseagreen', 'sandybrown', 'tan', 'slategray', 'steelblue', 'wheat', 'mediumslateblue', 'mediumorchid', 'cyan', 'springgreen', 'lime', 'dodgerblue', 'deeppink', 'mediumblue', 'green', 'tomato', 'crimson', 'palegoldenrod', 'lightsalmon', 'forestgreen', 'orchid', 'turquoise', 'darkolivegreen', 'lightseagreen', 'violet', 'salmon', 'indianred', 'rebeccapurple', 'peru', 'darkturquoise', 'lightskyblue', 'plum', 'aquamarine', 'mediumspringgreen', 'orange', 'purple', 'midnightblue', 'darkgray', 'darkorchid', 'blueviolet', 'teal', 'darkmagenta', 'palevioletred', 'firebrick', 'mediumpurple', 'gainsboro']

.. py:data:: COLOR_DICT
.. code-block:: pycon

  {'H': 'blue', 'H2': 'deepskyblue', 'He': 'olive', 'C': 'coral', 'CH4': 'darkorange', 'CO': 'limegreen', 'CO2': 'red', 'HCN': 'dimgray', 'C2H2': 'pink', 'C2H4': 'deeppink', 'N': 'darkviolet', 'NH3': 'magenta', 'N2': 'gold', 'O': 'greenyellow', 'H2O': 'navy', 'OH': 'darkkhaki', 'Si': 'lightslategray', 'SiO': 'darkturquoise', 'SiH4': 'mediumvioletred', 'Na': 'silver', '(NaCl)2': 'maroon', '(NaOH)2': 'hotpink', 'NaCl': 'rosybrown', 'K': 'black', '(KCl)2': 'chocolate', '(KOH)2': 'darkslateblue', 'KOH': 'lightgreen', 'KCl': 'darksalmon', 'S': 'cornflowerblue', 'H2S': 'darkgoldenrod', 'HS': 'yellowgreen', 'SO': 'mediumseagreen', 'SO2': 'skyblue', 'Al': 'khaki', 'AlOH': 'steelblue', 'Al2O': 'seagreen', 'OAlOH': 'tomato', 'Ca': 'orange', 'Ca(OH)2': 'indigo', 'e': 'darkgreen', 'Ti': 'crimson', 'TiO': 'brown', 'TiO2': 'indianred', 'VO': 'aquamarine', 'VO2': 'mediumaquamarine', 'V': 'darkcyan', 'Mg': 'sandybrown', 'MgH': 'lawngreen', 'Mg(OH)2': 'orangered', 'Fe': 'royalblue', 'FeH': 'wheat', 'Fe(OH)2': 'tan', 'F': 'yellow', 'OAlF2': 'sienna', 'TiF3': 'saddlebrown', 'AlF': 'orange', 'HF': 'lightblue', 'MnH': 'lime', 'Mn': 'rebeccapurple', 'PN': 'palegoldenrod', 'P': 'peachpuff', '(P2O3)2': 'cadetblue'}

.. py:function:: thermochemical_equilibrium(pressure, temperature, element_rel_abundance, stoich_vals, gibbs_funcs, tolx=2.22e-16, tolf=2.22e-16)
.. code-block:: pycon

    Low-level function to compute thermochemical equilibrium for the
    given chemical network at the specified temperature--pressure
    profile.

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
    tolx: float
        Relative error desired for convergence in the sum of squares.
    tolf: float
        Relative error desired for convergence in the approximate solution.

    Returns
    -------
    vmr: 2D float array
        Species volume mixing ratios in thermochemical equilibrium
        of shape [nlayers, nspecies].

.. py:function:: thermo_eval(temperature, thermo_funcs)
.. code-block:: pycon

    Low-level function to compute the thermochemical property
    specified by thermo_func at at the requested temperature(s).
    These can be, e.g., the heat_capacity or gibbs_free_energy
    functions returned by setup_network().

    Normally you want to use this function via the heat_capacity()
    and gibbs_free_energy() methods of the chemcat.Network() object.

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

.. py:function:: stoich_matrix(stoich_data)
.. code-block:: pycon

    Compute matrix of stoichiometric values for the given stoichiometric
    data for a network of species.

    Parameters
    ----------
    stoich_data: List of dictionaries
        Stoichiometric data (as dictionary of element-value pairs) for
        a list of species.

    Returns
    -------
    elements: 1D string array
        Elements for this chemical network.
    stoich_vals: 2D integer array
        Array containing the stoichiometric values for the
        requested species sorted according to the species and elements
        arrays.

    Examples
    --------
    >>> import chemcat.utils as u
    >>> stoich_data = [
    >>>     {'H': 2.0, 'O': 1.0},
    >>>     {'C': 1.0, 'H': 4.0},
    >>>     {'C': 1.0, 'O': 2.0},
    >>>     {'H': 2.0},
    >>>     {'H': 1.0},
    >>>     {'He': 1.0},
    >>> ]
    >>> elements, stoich_matrix = u.stoich_matrix(stoich_data)
    >>> print(elements, stoich_matrix, sep='\n')
    ['C' 'H' 'He' 'O']
    [[0 2 0 1]
     [1 4 0 0]
     [1 0 0 2]
     [0 2 0 0]
     [0 1 0 0]
     [0 0 1 0]]

.. py:function:: read_elemental(element_file)
.. code-block:: pycon

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
    >>> import chemcat.utils as u

    >>> element_file = f'{u.ROOT}chemcat/data/asplund_2021_solar_abundances.dat'
    >>> elements, dex = u.read_elemental(element_file)
    >>> for e in 'H He C N O'.split():
    >>>     print(f'{e:2}:  {dex[elements==e][0]:6.3f}')
    H :  12.000
    He:  10.914
    C :   8.460
    N :   7.830
    O :   8.690

.. py:function:: set_element_abundance(elements, base_composition, base_dex_abundances, metallicity=0.0, e_abundances={}, e_scale={}, e_ratio={})
.. code-block:: pycon

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
        and their ratio, e.g., for a C/O ratio of 0.8 set
        e_ratio = {'C_O': 0.8}.
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

    >>> solar = u.set_element_abundance(
    >>>     elements, sun_elements, sun_dex,
    >>> )

    >>> # Set custom metallicity to [M/H] = 0.5:
    >>> abund = u.set_element_abundance(
    >>>     elements, sun_elements, sun_dex, metallicity=0.5,
    >>> )
    >>> print([f'{e}: {q:.1e}' for e,q in zip(elements, abund)])
    ['H: 1.0e+00', 'He: 8.2e-02', 'C: 9.1e-04', 'N: 2.1e-04', 'O: 1.5e-03']

    >>> # Custom carbon abundance by direct value (dex):
    >>> abund = u.set_element_abundance(
    >>>     elements, sun_elements, sun_dex, e_abundances={'C': 8.8},
    >>> )
    >>> print([f'{e}: {q:.1e}' for e,q in zip(elements, abund)])
    ['H: 1.0e+00', 'He: 8.2e-02', 'C: 6.3e-04', 'N: 6.8e-05', 'O: 4.9e-04']

    >>> # Custom carbon abundance by scaling to 2x its solar value:
    >>> abund = u.set_element_abundance(
    >>>     elements, sun_elements, sun_dex, e_scale={'C': np.log10(2)},
    >>> )
    >>> print([f'{e}: {q:.1e}' for e,q in zip(elements, abund)])
    ['H: 1.0e+00', 'He: 8.2e-02', 'C: 5.8e-04', 'N: 6.8e-05', 'O: 4.9e-04']

    >>> # Custom carbon abundance by scaling to C/O = 0.8:
    >>> abund = u.set_element_abundance(
    >>>     elements, sun_elements, sun_dex, e_ratio={'C_O': 0.8},
    >>> )
    >>> print([f'{e}: {q:.1e}' for e,q in zip(elements, abund)])
    ['H: 1.0e+00', 'He: 8.2e-02', 'C: 3.9e-04', 'N: 6.8e-05', 'O: 4.9e-04']

.. py:function:: de_aliasing(input_species, sources)
.. code-block:: pycon

    Get the right species names as given in the selected database.

    Parameters
    ----------
    input_species: List of strings
        List of species names.
    sources: String or 1D iterable of strings
        The desired database sources.

    Returns
    -------
    output_species: List of strings
        Species names with aliases replaced with the names
        as given in source database.

    Examples
    --------
    >>> import chemcat.utils as u
    >>> input_species = ['H2O', 'C2H2', 'HO2', 'CO']
    >>> sources = 'janaf'
    >>> output_species = u.de_aliasing(input_species, sources)
    >>> print(output_species)
    ['H2O', 'C2H2', 'HOO', 'CO']

    >>> sources = 'cea'
    >>> output_species = u.de_aliasing(input_species, sources)
    >>> print(output_species)
    ['H2O', 'C2H2,acetylene', 'HO2', 'CO']

.. py:function:: resolve_sources(species, sources)
.. code-block:: pycon

    For each species in input, assign the right database proritizing
    by the order in sources.
    Parameters
    ----------
    species: 1D interable of strings
        Species to assign a database source.
    sources: 1D iterable of strings
        List of database sources in order of priority.
    Returns
    -------
    source_names: 1D array of strings
        Array with the assigned database to each species.
        If none found, leave value as None.
    Examples
    --------
    >>> import chemcat.utils as u
    >>> species = 'H2O CO (KOH)2 HO2'.split()
    >>> # Prioritize JANAF:
    >>> sources1 = u.resolve_sources(species, sources=['janaf', 'cea'])
    >>> # Prioritize CEA:
    >>> sources2 = u.resolve_sources(species, sources=['cea', 'janaf'])
    >>> # CEA exclusively:
    >>> sources3 = u.resolve_sources(species, sources=['cea'])
    >>> print(sources1, sources2, sources3, sep='\n')
    ['janaf' 'janaf' 'janaf' 'cea']
    ['cea' 'cea' 'janaf' 'cea']
    ['cea' 'cea' None 'cea']

.. py:function:: resolve_colors(species, color_dict=None, color_list=None)
.. code-block:: pycon

    Assign a color for each input neutral species (ions will have
    the same color of a neutral form of the same species).

    Parameters
    ----------
    species: 1D string iterable
        The species that need to be assigned a color.
    color_dict: dict
        A dict with predefined colors for species.
        It does not need to contain a value for all input species.
        Defaulted to u.COLOR_DICT.
    color_list: 1D string iterable
        A list of color names to assing to the species that have not
        been assigned via color_dict.
        If there are more species than colors, then cycle over this
        list.
        Defaulted to u.COLORS.

    Returns
    -------
    colors: Dict
        Dict assigning a color to each of the (neutral) species.

    Examples
    --------
    >>> import chemcat.utils as u
    >>> species = 'H He C H2 CH4 CO CO2 e- H+ H- H3+'.split()
    >>> colors = u.resolve_colors(species)

    >>> print(colors)
    {'H': 'blue',
     'He': 'olive',
     'C': 'salmon',
     'H2': 'deepskyblue',
     'CH4': 'darkorange',
     'CO': 'limegreen',
     'CO2': 'red',
     'e': 'darkgreen',
     'H3': 'royalblue'}

.. py:function:: plot_vmr(pressure, vmr, species, colors=None, vmr_range=None, fignum=320, title=None, fontsize=14, linewidth=2.0, rect=None, axis=None, savefig=None)
.. code-block:: pycon

    Plot VMRs vs pressure.

    Parameters
    ----------
    pressure: 1D float iterable
        pressure array in bars.
    vmr: 2D float array
        Volume mixing ratios of shape [nlayers, nspecies].
    species: 1D string iterable
        Names of the species in vmr.
    colors: 1D iterable of strings
        Color names to assign (sequentially) to the species.
        If None, default to chemcat.utils.COLOR_DICT values.
        Note that different ionic variations of a same species
        (e.g., H, H+, H-) are assigned a same color, but differ
        in line style.
    vmr_range: 1D float iterable
        The plotting boundaries along the vmr axis.
    fignum: integer or string
        The identifier of the figure.
    title: Syting
        Title for the figure.
    fontsize: Float
        Font size for labels texts. Legend texts will be fontsize-5.
    linewidth: Float
        Width of VMR lines.
    rect: 4-element float iterable
        Axis position (left, bottom, right, top).
        Note that legend will be placed to the right of this rect.
    axis: AxesSubplot instance
        Axis where to draw the VMRs. If not None, overrides fignum.
    savefig: String
        If not None, file name where to save the figure.

    Returns
    -------
    ax: AxesSubplot instance
        The matplotlib Axes of the figure.

    Examples
    --------
    >>> import chemcat as cat
    >>> import chemcat.utils as u
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> nlayers = 81
    >>> temperature = np.tile(1500.0, nlayers)
    >>> pressure = np.logspace(-8, 3, nlayers)
    >>> molecs = (
    >>>     'H2O CH4 CO CO2 NH3 N2 H2 HCN C2H2 C2H4 OH H He C N O '
    >>>     'e- H- H+ H2+ He+ '
    >>>     'Na Na- Na+ K K- K+ '
    >>>     'Si S SiO SiH4 H2S HS SO SO2 SiS'
    >>> ).split()

    >>> net = cat.Network(pressure, temperature, molecs)
    >>> vmr = net.thermochemical_equilibrium()
    >>> ax = u.plot_vmr(pressure, vmr, net.species, vmr_range=(1e-20,3))

.. py:function:: write_file(file, species, pressure, temperature, vmr)
.. code-block:: pycon

    Write pressure, temperature, and vmr values to file.

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
        Atmospheric volume mixing ratios, of shape [nspecies, nlayers].

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

    >>> # Save results to file:
    >>> cat.utils.write_file(
    >>>     'chemcat_chemistry.dat', net.species, pressure, temperature, vmr,
    >>> )
    >>> # Read from file:
    >>> d = cat.utils.read_file('chemcat_chemistry.dat')

.. py:function:: read_file(file)
.. code-block:: pycon

    Read a chemcat file.

    Parameters
    ----------
    file: String
        Path to file to read.

    Returns
    -------
    species: 1D string list
        Names of atmospheric species.
    pressure: 1D float array
        Atmospheric pressure profile (bar).
    temperature: 1D float array
        Atmospheric temperature profile (kelvin).
    vmr: 2D float array
        Atmospheric volume mixing ratios, of shape [nspecies, nlayers].

    Examples
    --------
    >>> import chemcat.utils as u

    >>> # Continuing from example in u.write_file(),
    >>> # Read from file:
    >>> species, pressure, temperature, vmr = u.read_file(
    >>>     'chemcat_chemistry.dat')

