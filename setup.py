# Copyright (c) 2022-2024 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

import os
import re
import setuptools
from setuptools import setup, Extension

from numpy import get_include


srcdir = 'src_c/'  # C-code source folder
incdir = 'src_c/include/'  # Include filder with header files

cfiles = os.listdir(srcdir)
cfiles = list(filter(lambda x: re.search('.+[.]c$', x), cfiles))
cfiles = list(filter(lambda x: not re.search('[.#].+[.]c$', x), cfiles))

inc = [get_include(), incdir]
eca = ['-lm', '-O3', '-ffast-math']
ela = ['-lm']

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

setup(
    include_dirs = inc,
    ext_modules = extensions,
)

