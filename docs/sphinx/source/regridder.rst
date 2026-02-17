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

   from smmregrid import Regridder
   
   # Initialize with source and target grids
   regridder = Regridder('source.nc', 'r360x180')
   
   # Apply regridding to data
   import xarray as xr
   data = xr.open_dataset('data.nc')
   regridded = regridder.regrid(data)

**From Pre-computed Weights:**

.. code-block:: python

   # Initialize with pre-computed weights
   weights = xr.open_dataset('weights.nc')
   regridder = Regridder(weights=weights)
   regridded = regridder.regrid(data)

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
  - CDO grid description (e.g., 'r360x180', 'n128')

**Weights-based Initialization:**

* **weights** (*xarray.Dataset*): Pre-computed interpolation weights from CDO or CdoGenerate.
  If provided, source_grid and target_grid are not needed.

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

* **transpose** (*bool*, default: True): If True, transposes output so vertical coordinate 
  appears just before spatial coordinates. Affects dimension ordering in output.

**Dimension Control:**

* **mask_dim** (*str*, optional): Name of coordinate with varying mask, that will be used for 3D regridding.
  Forces recognition of this dimension as vertical even if not in standard list.

* **horizontal_dims** (*list*, optional): List of spatial dimensions to interpolate.
  Forces recognition of these dimensions as horizontal spatial coordinates.

**NaN Handling:**

* **check_nan** (*bool*, default: False): If True, analyzes NaN patterns in source data 
  to automatically identify vertical-like dimensions based on missing data variation and set
  up the mask_dim accordingly. Useful for datasets with unpredicted vertical structures.
  Planned to be extended in a more automatic way in the future.

**CDO Configuration:**

* **cdo** (*str*, default: 'cdo'): Path to CDO executable.

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

   # 3D surface data
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

**Automatic 3D Detection with NaN Analysis:**

This is feature under test which allows smmregrid to automatically detect mask-changing dimension

.. code-block:: python

   # Let smmregrid detect vertical dimension from NaN patterns
   regridder = Regridder(
       'ocean_data.nc', 'target.nc',
       check_nan=True
   )

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

**3D Atmospheric Data:**

.. code-block:: python

   # Regrid 3D atmospheric data with pressure levels
   regridder = Regridder(
       'ecmwf_data.nc', 'regular_grid.nc',
       method='bil',
       mask_dim='plev',
   )
   regridded = regridder.regrid(atmospheric_data)

**Ocean Data with NaN Detection:**

.. code-block:: python

   # Let smmregrid automatically detect depth dimension from NaN patterns
   regridder = Regridder(
       'ocean_model.nc', 'obs_locations.nc', 
       method='con',
       check_nan=True,
   )
