Basic Usage
===========

`smmregrid` can be used in two ways. By feeding to the tool a previously generated weights computed
with cdo or by creating the weights starting from the source and the target grid.

For further details on the different usages, please consult the `demo notebook <https://github.com/jhardenberg/smmregrid/blob/main/demo.ipynb>`_

Starting from weights
---------------------

You can simply generate the weights with `cdo_generate_weights` function (or load precomputed weights on your own)
and then activate the `Regridder()` class and run the `.regrid()` method on the data you want to regrid

.. code-block:: python 

    from smmregrid import Regridder, cdo_generate_weights
    weights = cdo_generate_weights(filein, target_grid)
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