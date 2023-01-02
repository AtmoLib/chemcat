# Copyright (c) 2022-2023 Blecic and Cubillos
# chemcat is open-source software under the GPL-2.0 license (see LICENSE)

from .janaf import *

__all__ = janaf.__all__


# Clean up top-level namespace--delete everything that isn't in __all__
# or is a magic attribute, and that isn't a submodule of this package
for varname in dir():
    if not ((varname.startswith('__') and varname.endswith('__')) or
            varname in __all__):
        del locals()[varname]
del(varname)
