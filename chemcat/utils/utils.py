# Copyright (c) 2022 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

__all__ = [
    'ROOT',
    'COLORS',
    'COLOR_DICT',
    'stoich_matrix',
    'read_elemental',
    'set_element_abundance',
    'de_aliasing',
    'resolve_sources',
    'resolve_colors',
    'plot_vmr',
    'write_file',
    'read_file',
]

import itertools
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = str(Path(__file__).parents[2]) + os.path.sep


# A long list of colors:
COLORS = [
    'royalblue', 'darkorange', 'red', 'darkgreen', 'magenta',
    'blue', 'limegreen', 'gold', 'dimgray', 'navy',
    'deepskyblue', 'silver', 'black', 'olive', 'chocolate',
    'skyblue', 'darkviolet', 'greenyellow', 'pink', 'coral',
    'darkcyan', 'rosybrown', 'cornflowerblue', 'mediumvioletred', 'maroon',
    'darkgoldenrod', 'darkkhaki', 'hotpink', 'darkslateblue', 'lightgreen',
    'yellowgreen', 'seagreen', 'yellow', 'slateblue', 'sienna',
    'peachpuff', 'orangered', 'goldenrod', 'brown', 'khaki',
    # Unique line
    'saddlebrown', 'mediumseagreen', 'darksalmon', 'cadetblue',
    'mediumaquamarine', 'darkslategray', 'lightsteelblue', 'indigo',
    'lightcoral', 'lightslategray', 'lawngreen', 'lightblue',
    'darkseagreen', 'sandybrown', 'tan', 'slategray',
    'steelblue', 'wheat', 'mediumslateblue', 'mediumorchid',
    'cyan', 'springgreen', 'lime', 'dodgerblue', 'deeppink',
    'mediumblue', 'green', 'tomato', 'crimson', 'palegoldenrod',
    'lightsalmon', 'forestgreen', 'orchid', 'turquoise', 'darkolivegreen',
    'lightseagreen', 'violet', 'salmon', 'indianred', 'rebeccapurple',
    'peru', 'darkturquoise', 'lightskyblue', 'plum', 'aquamarine',
    'mediumspringgreen', 'orange', 'purple', 'midnightblue', 'darkgray',
    'darkorchid', 'blueviolet', 'teal', 'darkmagenta', 'palevioletred',
    'firebrick', 'mediumpurple', 'gainsboro',
]

COLOR_DICT = {
    'H': 'blue',
    'H2': 'deepskyblue',
    'He': 'olive',
    # Carbons:
    'C': 'salmon',
    'CH4': 'darkorange',
    'CO': 'limegreen',
    'CO2': 'red',
    'HCN': 'dimgray',
    'C2H2': 'pink',
    'C2H4': 'deeppink',
    # Nitrogens:
    'N': 'darkviolet',
    'NH3': 'magenta',
    'N2': 'gold',
    # Oxygens:
    'O': 'greenyellow',
    'H2O': 'navy',
    'OH': 'darkkhaki',
    # Silicons:
    'Si': 'coral',
    'SiO': 'darkturquoise',
    'SiH4': 'mediumvioletred',
    # Alkali:
    'Na': 'silver',
    '(NaCl)2': 'maroon',
    '(NaOH)2': 'hotpink',
    'NaCl': 'rosybrown',
    'K': 'black',
    '(KCl)2': 'chocolate',
    '(KOH)2': 'darkslateblue',
    'KOH': 'lightgreen',
    'KCl': 'darksalmon',
    # Sulfurs:
    'S': 'cornflowerblue',
    'H2S': 'darkgoldenrod',
    'HS': 'yellowgreen',
    'e': 'darkgreen',
    # Aluminum:
    'Al': 'khaki',
    'AlOH': 'steelblue',
    'Al2O': 'seagreen',
    'OAlOH': 'tomato',
    'Ca': 'orange',
    'Ca(OH)2': 'indigo',
    # Heavy metals:
    'Ti': 'crimson',
    'TiO': 'brown',
    'TiO2': 'indianred',
    'VO': 'aquamarine',
    'VO2': 'skyblue',
    'V': 'darkcyan',
    'Mg': 'sandybrown',
    'MgH': 'lawngreen',
    'Mg(OH)2': 'orangered',
    'Fe': 'royalblue',
    'FeH': 'wheat',
    'Fe(OH)2': 'tan',
    'F': 'yellow',
    'OAlF2': 'sienna',
    'TiF3': 'saddlebrown',
    'AlF': 'mediumseagreen',
    'HF': 'lightblue',
    'MnH': 'lime',
    'Mn': 'rebeccapurple',
    'PN': 'palegoldenrod',
    'P': 'peachpuff',
    '(P2O3)2': 'cadetblue',
}


def stoich_matrix(stoich_data):
    r"""
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
    """
    elements = []
    for s in stoich_data:
        elements += list(s.keys())
    elements = sorted(set(elements))

    nspecies = len(stoich_data)
    nelements = len(elements)
    stoich_vals = np.zeros((nspecies, nelements), int)
    for i in range(nspecies):
        for key,val in stoich_data[i].items():
            j = elements.index(key)
            stoich_vals[i,j] = val
    elements = np.array(elements)
    return elements, stoich_vals


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
    for element, ratio in e_ratio.items():
        element1, element2 = element.split('_')
        idx1 = np.array(elements) == element1
        idx2 = np.array(elements) == element2
        log_ratio = np.log10(ratio)
        elemental_abundances[idx1] = elemental_abundances[idx2] + log_ratio

    # Convert elemental log VMR (relative to H=12.0) to VMR (rel. to H=1.0):
    elemental_abundances = 10**(elemental_abundances-12.0)
    elemental_abundances[np.array(elements) == 'e'] = 0.0
    return elemental_abundances


def de_aliasing(input_species, sources):
    """
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
    """
    if isinstance(sources, str):
        sources = [sources]

    # Get lists of species names aliases:
    whites = f'{ROOT}chemcat/data/white_pages.txt'
    aliases = []
    for line in open(whites, 'r'):
        if line.startswith('#'):
            continue
        aliases.append(line.split())
    all_aliases = np.concatenate(aliases)

    source_index = {
        'janaf': 0,
        'cea': 1,
    }

    output_species = []
    for species in input_species:
        if species not in all_aliases:
            output_species.append(species)
            continue
        # Find set of aliases:
        for alias in aliases:
            if species in alias:
                break
        # Search source in priority order:
        for source in sources:
            alias_name = alias[source_index[source]]
            if alias_name != 'None':
                output_species.append(alias_name)
                break
        else:
            output_species.append(species)
    return output_species


def resolve_sources(species, sources):
    r"""
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
    """
    # Need to import here to avoid circular import error in Python 3.6:
    from .. import janaf
    from .. import cea

    if isinstance(sources, str):
        sources = [sources]

    nspecies = len(species)
    source_names = np.tile(None, nspecies)

    for source in sources:
        if source == 'janaf':
            in_janaf = janaf.is_in(species)
            is_missing = [name is None for name in source_names]
            source_names[is_missing & in_janaf] = 'janaf'

        elif source == 'cea':
            in_cea = cea.is_in(species)
            is_missing = [name is None for name in source_names]
            source_names[is_missing & in_cea] = 'cea'

    return source_names


def resolve_colors(species, color_dict=None, color_list=None):
    """
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
    """
    if color_list is not None and color_dict is None:
        color_dict = {}

    if color_dict is None:
        color_dict = COLOR_DICT
    if color_list is None:
        color_list = COLORS

    # Get all neutral species (plus electron):
    neutral_species = []
    for spec in species:
        neutral = spec.replace('-','').replace('+','')
        if neutral not in neutral_species:
            neutral_species.append(neutral)

    dicted_colors = [
        color_dict[spec]
        for spec in neutral_species
        if spec in color_dict
    ]
    remaining_colors = [
        color for color in color_list
        if color not in dicted_colors
    ]
    # Put dict_colors at the end, create an infinite cycle:
    colors_cycle = itertools.cycle(remaining_colors+dicted_colors)

    # Assign colors to each neutral species:
    # Species listed in dict:
    colors = {
        spec: color_dict[spec]
        for spec in neutral_species
        if spec in color_dict
    }
    # Remaining species:
    other_colors = {
        spec: next(colors_cycle)
        for spec in neutral_species
        if spec not in colors
    }
    colors.update(other_colors)
    return colors


def plot_vmr(
    pressure, vmr, species, colors=None, vmr_range=None, fignum=0, title=None,
):
    """
    Plot VMRs vs pressure.

    Parameters
    ----------
    TBD

    Examples
    --------
    >>> import chemcat as cat
    >>> import chemcat.utils as u
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> nlayers = 81
    >>> temperature = np.tile(1200.0, nlayers)
    >>> pressure = np.logspace(-8, 3, nlayers)
    >>> molecs = 'H2O CH4 CO CO2 NH3 N2 H2 HCN OH C2H2 C2H4 H He C N O'.split()

    >>> molecs = (
    >>>     'H2O CH4 CO CO2 NH3 N2 H2 HCN C2H2 C2H4 OH H He C N O '
    >>>     'e- H- H+ H2+ He+ '
    >>>     'Na Na- Na+ K K- K+ Mg Mg+ Fe Fe+ '
    >>>     'Ti TiO TiO2 Ti+ TiO+ V VO VO2 V+ '
    >>>     'Si SiO SiO2 NaCl Cl KCl Al AlO AlO2').split()

    >>> net = cat.Network(pressure, temperature, molecs)
    >>> vmr = net.thermochemical_equilibrium()
    >>> ax = u.plot_vmr(pressure, vmr, net.species)
    """
    species = list(species)

    # neutralized names:
    neutral_species = []
    for spec in species:
        neutral = spec.replace('-','').replace('+','')
        if neutral not in neutral_species:
            neutral_species.append(neutral)

    if colors is None:
        colors = resolve_colors(species)
    else:
        colors = {
            spec: color
            for spec,color in zip(neutral_species, colors)
        }

    fs = 14
    lw = 2.0
    dashes = {
        'neutral': (),
        'cation': (5.0, 1.5),
        'ion': (2.0, 0.75),
    }
    labels = ['neutral', 'cation', 'ion']
    ion_handles = [
        Line2D([], [], color='k', lw=1.5, dashes=dashes[label], label=label)
        for label in labels
    ]
    has_ions = np.any(['+' in spec or '-' in spec for spec in species])

    fig = plt.figure(fignum, (8.5,5.0))
    plt.clf()
    plt.subplots_adjust(0.09, 0.11, 0.88, 0.94)
    ax = plt.subplot(111)
    ion_legend = ax.legend(
        handles=ion_handles,
        loc='lower left', labelspacing=0.1, framealpha=0.6,
    )
    for name in species:
        charge = int('+' in name) - int('-' in name)
        charge_label = labels[charge]
        spec = name.replace('+','').replace('-','')
        if charge == 0 or spec not in species:
            label = name
        else:
            label = None
        ax.loglog(
            vmr[:,species.index(name)], pressure,
            label=label, lw=lw, color=colors[spec],
            dashes=dashes[charge_label],
        )
    ax.tick_params(
        which='both', right=True, top=True, direction='in', labelsize=fs-2,
    )
    ax.set_ylim(np.amax(pressure), np.amin(pressure))
    if vmr_range is not None:
        ax.set_xlim(vmr_range)
    ax.set_ylabel('Pressure (bar)', fontsize=fs)
    ax.set_xlabel('Volume mixing ratio', fontsize=fs)
    if has_ions:
        ax.add_artist(ion_legend)
    if title is not  None:
        ax.set_title(title, fontsize=fs)

    leg_args = {
        'loc': (1.01, 0.0),
        'fontsize': fs-5,
        'ncol': 1,
        'columnspacing': 1.0,
        'labelspacing': 0.0,
    }
    ax.legend(**leg_args)

    # Matplotlib black magic:
    def on_draw(event):
        """This will be called once the figure is drawn"""
        ax = event.canvas.figure.axes[0]
        legend = ax.get_legend()
        height_ratio = (
            legend.get_window_extent().height / ax.get_window_extent().height
        )
        ncols = int(height_ratio) + 1
        if ncols > 1:
            plt.subplots_adjust(0.09, 0.11, 0.79, 0.94)
            leg_args['ncol'] = ncols
            ax.legend(**leg_args)
            fig.canvas.draw()
    fig.canvas.mpl_connect('draw_event', on_draw)

    return ax


def write_file(file, species, pressure, temperature, vmr):
    """
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
    """
    str_species = ' '.join(f'{spec:<15}' for spec in species)

    with open(file, 'w') as f:
        # Header:
        f.write(
            '# chemcat chemistry composition calculation '
            '(bar, K, volume mixing ratios)\n'
            f'# pressure   temperature  {str_species.rstrip()}\n\n'
        )
        # Per-layer data:
        for press, temp, vmr_layer in zip(pressure, temperature, vmr):
            str_vmr = ' '.join(f'{q:<15.8e}' for q in vmr_layer)
            f.write(f'{press:.8e}  {temp:8.2f}  {str_vmr}\n')


def read_file(file):
    r"""
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
    """
    with open(file, 'r') as f:
        _= f.readline()
        header = f.readline()
    species = header.split()[3:]

    data = np.loadtxt(file, unpack=True)
    pressure = data[0]
    temperature = data[1]
    vmr = data[2:].T

    return species, pressure, temperature, vmr
