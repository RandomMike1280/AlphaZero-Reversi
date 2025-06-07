from setuptools import setup, find_packages

setup(
    name="reversi",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[],  # Will be populated by cythonize
    install_requires=[
        'numpy>=1.19.0',
        'cython>=0.29.0',
    ],
    python_requires='>=3.7',
)
