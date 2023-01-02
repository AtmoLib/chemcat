# Copyright (c) 2022-2023 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

import os
import re
import setuptools
from setuptools import setup, Extension

from numpy import get_include


def get_version(package):
    """Return package version as listed in __version__ in version.py"""
    path = os.path.join(os.path.dirname(__file__), package, 'version.py')
    with open(path, 'rb') as f:
        init_py = f.read().decode('utf-8')
    return re.search("__version__ = ['\"]([^'\"]+)['\"]", init_py).group(1)


srcdir = 'src_c/'  # C-code source folder
incdir = 'src_c/include/'  # Include filder with header files

cfiles = os.listdir(srcdir)
cfiles = list(filter(lambda x: re.search('.+[.]c$', x), cfiles))
cfiles = list(filter(lambda x: not re.search('[.#].+[.]c$', x), cfiles))

inc = [get_include(), incdir]
eca = ['-ffast-math']
ela = []

extensions = [
    Extension(
        'chemcat.lib.' + cfile.rstrip('.c'),
        sources=[f'{srcdir}{cfile}'],
        include_dirs=inc,
        extra_compile_args=eca,
        extra_link_args=ela,
    )
    for cfile in cfiles
]

package_data = {
    'chemcat': [
        'data/*',
        'data/janaf/*',
    ]
}

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name = 'chemcat',
    version = get_version('chemcat'),
    author = 'Jasmina Blecic and Patricio Cubillos',
    author_email = 'patricio.cubillos@oeaw.ac.at',
    url = 'https://github.com/atmolib/chemcat',
    packages = setuptools.find_packages(),
    package_data = package_data,
    install_requires = [
        'numpy>=1.19.1',
        'scipy>=1.5.2',
        'matplotlib>=3.3.4',
        'more-itertools>=8.4.0',
        ],
    tests_require = [
        'pytest>=3.9',
        ],
    license = 'GPLv2',
    description = 'Chemistry Calculator for Atmospheres',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    include_dirs = inc,
    ext_modules = extensions,
)

