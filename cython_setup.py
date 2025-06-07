from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the extension
board_extension = Extension(
    "reversi.cython.board",
    ["src/reversi/cython/board.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

# Setup configuration
setup(
    name="reversi-cython",
    ext_modules=cythonize(
        [board_extension],
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'nonecheck': False,
            'cdivision': True,
        },
    ),
)
