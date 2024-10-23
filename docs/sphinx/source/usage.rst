Basic Usage
===========

`smmregrid` can be used in two ways. By feeding to the tool a previously generated weights computed
with cdo or by creating the weights starting from the source and the target grid.

For further details on the different usages, please consult the `demo notebook <https://github.com/jhardenberg/smmregrid/blob/main/demo.ipynb>`_

Starting from weights
---------------------

You can simply generate the weights with `CdoGenerate()` class and its method `weights()` (or load precomputed weights on your own).
This will create the weight with the `gen<method>` command from CDO. 
Then activate the sparse smmregrid `Regridder()` class and run the `regrid()` method on the data you want to regrid.


.. code-block:: python 

    from smmregrid import Regridder, CdoGenerate
    weights = CdoGenerate(filein, target_grid).weights(method='con')
    interpolator = Regridder(weights=wfield)
    myfile = interpolator.regrid(xfield[var])

Starting from data
------------------

You can simply initiate the class with the source file (both a file or a xarray object) and then
simply operate with the `.regrid()` method

.. code-block:: python

    from smmregrid import Regridder
    interpolator = Regridder(filein, target_grid)
    myfile = interpolator.regrid(xfield[var])


Area generation
---------------

Similarly, it is possible to produce the area for both source and target grid making use of 
the `CdoGenerate()` class and the method `areas()`. This build again on CDO, specifically on
the `gridarea` command.

For source area:

.. code-block:: python 

    from smmregrid import CdoGenerate
    areas = CdoGenerate(filein, target_grid).areas()

For target area:

.. code-block:: python 

    from smmregrid import CdoGenerate
    areas = CdoGenerate(filein, target_grid).areas(target=True)