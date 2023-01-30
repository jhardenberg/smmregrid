
from distutils.core import setup

setup(name='smmregrid',
      version='0.0.1',
      description='Regridding based on sparse matrix multiplication',
      author=' Jost von Hardenberg, Scott Wales',
      author_email='jost.hardenberg@polito.it',
      url='https://github.com/jhardenberg/smmregrid',
      python_requires='>3.7, <3.11',
      packages=['smmregrid'],
      install_requires=[
          'numpy',
          'xarray',
          'netcdf4',
          'dask',
          'sparse',
          'cfunits'
      ]
      )
