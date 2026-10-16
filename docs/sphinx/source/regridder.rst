Regridder Class
===============

The :class:`Regridder` class is the main interface for performing regridding operations in smmregrid.
It provides a high-level API for interpolating data between different grids using either pre-computed
weights or by generating weights on-the-fly through CDO integration.

.. autoclass:: smmregrid.Regridder
   :members:
   :undoc-members:

Basic Usage
-----------

The Regridder can be initialized in two ways:

**From Grids (generates weights automatically):**

.. code-block:: python

   import xarray as xr
   from smmregrid import Regridder

   # Initialize with source and target grids
   regridder = Regridder('source.nc', 'r360x180')

   # Apply regridding to data
   data = xr.open_dataset('data.nc')
   regridded = regridder.regrid(data)

**From Pre-computed Weights:**

.. code-block:: python

   # Initialize with pre-computed weights, as an xarray.Dataset or as a file path
   weights = xr.open_dataset('weights.nc')
   regridder = Regridder(weights=weights)
   regridder = Regridder(weights='weights.nc')
   regridded = regridder.regrid(data)

**Philosophy:**

``regrid()`` accepts both a ``Dataset`` and a ``DataArray``. The output is lazy when the input is
Dask-backed (e.g. opened with ``chunks={}`` or ``open_mfdataset``): call ``.compute()`` or ``.to_netcdf()``
to get the values.

For a one-off operation, the :func:`smmregrid.regrid` function builds a :class:`Regridder` and applies it
in a single call. As for the class, it can be initialized in two ways:

* **With a target grid**: the weights are generated from the data at each call. This is convenient for a single
  dataset, but repeating it for several datasets on the same grid regenerates the same weights every time.
* **With pre-computed weights**: no weights are generated, so this is efficient also when called repeatedly.

.. note::

   3D weights (``mask_dim``, or ``check_nan`` when it finds a mask-changing dimension) are generated
   with one process per level, using ``multiprocessing``. With the ``spawn`` and ``forkserver`` start
   methods (the defaults on macOS, and on Linux from Python 3.14) a script must protect its main code,
   otherwise it fails with ``An attempt has been made to start a new process before the current process
   has finished its bootstrapping phase``:

   .. code-block:: python

      def main():
          regridder = Regridder('ocean_data.nc', 'r180x90', mask_dim='lev')
          ...

      if __name__ == '__main__':
          main()

   This is not needed in Jupyter notebooks or interactive sessions.

Constructor Parameters
----------------------

**Grid-based Initialization:**

* **source_grid** (*str* or *xarray.Dataset/DataArray*): Source grid specification. Can be:

  - File path to NetCDF file
  - xarray Dataset or DataArray
  - Must be provided together with target_grid

* **target_grid** (*str* or *xarray.Dataset/DataArray*): Target grid specification. Can be:

  - File path to NetCDF file
  - xarray Dataset or DataArray
  - CDO grid description (e.g., 'r360x180', 'F128')

**Weights-based Initialization:**

* **weights** (*xarray.Dataset* or *str*): Pre-computed interpolation weights from CDO or CdoGenerate,
  as a Dataset or as a file path. If provided, source_grid and target_grid are not needed.
  The interpolation method is read from the weights, so ``method`` is ignored.

**Regridding Options:**

* **method** (*str*, default: 'con'): Interpolation method for weight generation. Options:

  - 'bic': Bicubic interpolation
  - 'bil': Bilinear interpolation
  - 'con': First-order conservative (default)
  - 'con2': Second-order conservative
  - 'dis': Distance-weighted average
  - 'laf': Largest area fraction
  - 'nn': Nearest neighbour
  - 'ycon': YAC first-order conservative

* **remap_area_min** (*float*, default: 0.5): Minimum fractional area for conservative remapping.
  Grid cells with area fraction below this threshold are masked as NaN. Range: 0.0-1.0.
  It has no effect on non-conservative methods.

* **transpose** (*bool*, default: True): If True, transposes the output of 3D regridding (with ``mask_dim``)
  so that the mask dimension appears just before the spatial coordinates, e.g. ``(time, lev, lat, lon)``
  instead of ``(lev, time, lat, lon)``.

**Dimension Control:**

* **mask_dim** (*str*, optional): Name of coordinate with varying mask, that will be used for 3D regridding.
  Forces recognition of this dimension as vertical even if not in standard list.

* **horizontal_dims** (*list*, optional): List of spatial dimensions to interpolate.
  Forces recognition of these dimensions as horizontal spatial coordinates.

**NaN Handling:**

* **check_nan** (*bool*, default: False): If True, analyzes NaN patterns in source data
  to automatically identify vertical-like dimensions based on missing data variation and set
  up the mask_dim accordingly. Useful for datasets with unpredicted vertical structures.

* **skipna** (*bool*, default: False): If True, allows regridding of data with time-varying masks or vertical masks in 3D data without declaring mask_dim.
  It is based on renormalization of weights similarly to what done by ESMF.
  If ``na_thes``(see below) is set to "auto", it tries to replicate CDO behaviour for conservative and nearest neighbour, but it is not guaranteed to be exactly the same for other methods.
  Can be faster than using ``mask_dim``, but it is not guaranteed to be exactly the same as CDO for all methods.
  When initializing from grids, ``mask_dim`` and ``check_nan`` are ignored: the same weights apply to all levels.

* **na_thres** (*float* or *str*, default: "auto"): Maximum fraction of missing weights before a target cell
  is set to NaN (with ``skipna``) or masked (without ``skipna``, in the destination mask computed from the weights).
  Range: 0.0-1.0. "auto" gives 0.5 without ``skipna``; with ``skipna`` it gives 1e-6 for 'bil' and 'bic'
  (NaN as soon as a neighbour is missing, as CDO) and 1.0 for the other methods.

**CDO Configuration:**

* **cdo** (*str*, optional): Path to CDO executable. Default: the ``CDO`` environment variable, or ``cdo`` on the PATH.
  CDO >= 2.6.2 is required.

* **cdo_extra** (*list*, optional): Additional CDO commands for weight generation.
  Example: ``['-selname,temperature']`` to select specific variables.

* **cdo_options** (*list*, optional): CDO options for weight generation.
  Example: ``['-f', 'nc4', '-P', '4']`` for NetCDF4 output with 4-thread parallelization.

**Logging:**

* **loglevel** (*str*, default: 'WARNING'): Logging verbosity level.


2D vs 3D Regridding
-------------------

**2D/3D Regridding (default):**

Applies to data without masked structure. This can be used for atmospheric 2D or 3D data.

.. code-block:: python

   # 2D surface data
   regridder = Regridder('surface_temp.nc', 'r180x90')
   regridded = regridder.regrid(data)

   # 3D atmospheric data, with the same mask on all levels
   regridder = Regridder('3d_wind.nc', 'r180x90')
   regridded = regridder.regrid(data)

**3D masked Regridding:**

For data with mask-changing levels, specify the dimension with varying mask using `mask_dim`.
In this way smmregrid will compute specific weights for each level, without the risk of mixing masked and unmasked points.

.. code-block:: python

   # 3D oceanic data
   regridder = Regridder(
       'ocean_data.nc', 'target.nc',
       mask_dim='depth'
   )

.. warning::

   When using `mask_dim`, ensure that the specified dimension is correctly identified as the one with varying mask.
   Incorrect specification may lead to unexpected regridding results.

.. warning::

   If you select levels along the `mask_dim` before regridding, keep the dimension by using a list or a slice,
   e.g. ``data.isel(depth=[5])`` or ``data.sel(depth=slice(0, 100))``. Selecting a single level with an integer
   (``data.isel(depth=5)``) drops the dimension, and the variable is silently left out of the output.

**Automatic 3D Detection with NaN Analysis:**

This is feature under test which allows smmregrid to automatically detect mask-changing dimension

.. code-block:: python

   # Let smmregrid detect vertical dimension from NaN patterns
   regridder = Regridder(
       'ocean_data.nc', 'target.nc',
       check_nan=True
   )

Choosing how to handle missing values
-------------------------------------

* Mask fixed in time and along the vertical (e.g. land-sea mask of a 2D field): no option needed.
  The mask of the first time step is used for the weights.
* Mask changing along a vertical dimension (e.g. ocean levels, or pressure levels below the surface):
  use ``mask_dim``, which builds one set of weights per level and matches CDO exactly.
  ``check_nan=True`` tries to detect that dimension for you.
* Mask changing in time (e.g. sea ice), or a large number of levels: use ``skipna=True``. A single set of weights
  is used and the missing values are handled at each call, which gives a much smaller Dask graph.
  It matches CDO for conservative and nearest neighbour.

Without ``skipna``, NaNs that are not in the first time step propagate to the target cells they contribute to.

Conservative Remapping Options
------------------------------

For conservative methods ('con', 'con2', 'ycon'), additional masking options are available:

**Area Fraction Masking:**

.. code-block:: python

   # Strict masking - only cells with >80% coverage
   regridder = Regridder(
       'source.nc', 'target.nc',
       method='con',
       remap_area_min=0.8
   )


Performance Optimization
-------------------------


**Pre-computed Weights:**

.. code-block:: python

   # Generate weights once, reuse for multiple datasets
   from smmregrid import CdoGenerate

   generator = CdoGenerate('source.nc', 'target.nc')
   weights = generator.weights(method='con')

   # Reuse weights for multiple regridding operations
   regridder = Regridder(weights=weights)
   result1 = regridder.regrid(dataset1)
   result2 = regridder.regrid(dataset2)

   # Save them to disk to skip the generation in later sessions
   weights.to_netcdf('weights.nc')
   regridder = Regridder(weights='weights.nc')

For 3D weights, ``CdoGenerate(...).weights(mask_dim='lev', nproc=4)`` generates several levels in parallel.
To use ``skipna`` with pre-computed weights, pass ``skipna=True`` to both :class:`CdoGenerate` and :class:`Regridder`.


Examples
--------

**Basic Example:**

.. code-block:: python

   from smmregrid import Regridder
   import xarray as xr

   # Load data and regrid to regular 1-degree grid
   data = xr.open_dataset('model_output.nc')
   regridder = Regridder('model_output.nc', 'r360x180')
   regridded = regridder.regrid(data)

**3D Atmospheric Data with Missing Values Below the Surface:**

.. code-block:: python

   # Pressure-level data with NaNs below the orography (e.g. CMOR output): the mask changes
   # along plev, so weights are built per level. Without such NaNs, mask_dim is not needed.
   regridder = Regridder(
       'cmor_ua.nc', 'r180x90',
       method='bil',
       mask_dim='plev',
   )
   regridded = regridder.regrid(atmospheric_data)

**Ocean Data with NaN Detection:**

.. code-block:: python

   # Let smmregrid automatically detect depth dimension from NaN patterns
   regridder = Regridder(
       'ocean_model.nc', 'r180x90',
       method='con',
       check_nan=True,
   )
   regridded = regridder.regrid(xr.open_dataset('ocean_model.nc'))

**Ocean Data with skipna:**

.. code-block:: python

   # Same data with a single set of weights: missing values are skipped at each call
   regridder = Regridder('ocean_model.nc', 'r180x90', skipna=True)
   regridded = regridder.regrid(xr.open_dataset('ocean_model.nc'))
