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
  
  - Larger values (0.7-0.9): More aggressive masking, avoids partial coverage artifacts
  - Smaller values (0.1-0.3): Less masking, retains more data but may include edge effects

* **transpose** (*bool*, default: True): If True, transposes output so vertical coordinate 
  appears just before spatial coordinates. Affects dimension ordering in output.

**Dimension Control:**

* **vertical_dim** (*str*, optional): Name of vertical coordinate for 3D regridding.
  Forces recognition of this dimension as vertical even if not in standard list.

* **horizontal_dims** (*list*, optional): List of spatial dimensions to interpolate.
  Forces recognition of these dimensions as horizontal spatial coordinates.

**NaN Handling:**

* **check_nan** (*bool*, default: False): If True, analyzes NaN patterns in source data 
  to automatically identify vertical-like dimensions based on missing data variation and set
  up the vertical_dim accordingly. Useful for datasets with unpredicted vertical structures.
  Planned to be extended in a more automatic way in the future.

**CDO Configuration:**

* **cdo** (*str*, default: 'cdo'): Path to CDO executable.

* **cdo_extra** (*list*, optional): Additional CDO commands for weight generation.
  Example: ``['-selname,temperature']`` to select specific variables.

* **cdo_options** (*list*, optional): CDO options for weight generation.  
  Example: ``['-f', 'nc4', '-P', '4']`` for NetCDF4 output with 4-thread parallelization.

**Logging:**

* **loglevel** (*str*, default: 'WARNING'): Logging verbosity level.
  Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.

Methods
-------

regrid()
~~~~~~~~

Apply regridding to input data.

.. code-block:: python

   regridded_data = regridder.regrid(source_data)

**Parameters:**

* **source_data** (*xarray.Dataset* or *xarray.DataArray*): Data to be regridded.
  Can handle both single variables (DataArray) and multi-variable datasets (Dataset).

**Returns:** Regridded data with same structure as input but on target grid.

**Features:**

* **Multi-variable support**: Automatically processes all variables in Dataset
* **Dimension preservation**: Maintains non-spatial dimensions (time, ensemble, etc.)
* **Coordinate handling**: Properly transfers and updates coordinate information
* **Chunking preservation**: Maintains Dask chunking for large datasets
* **Attribute preservation**: Retains metadata and attributes from source data

Grid Type Handling
-------------------

The Regridder automatically detects and handles different grid types:

**Single Grid Type:**

.. code-block:: python

   # Standard case - all variables share same spatial grid
   regridder = Regridder('model.nc', 'obs_grid.nc')

**Multiple Grid Types:**

.. code-block:: python

   # Mixed grids (e.g., atmospheric + ocean data)
   regridder = Regridder('coupled_model.nc', 'target.nc', loglevel='INFO')
   # Will detect multiple grids and handle appropriately

**Forced Grid Recognition:**

.. code-block:: python

   # Force specific dimension recognition
   regridder = Regridder(
       'data.nc', 'target.nc',
       vertical_dim='model_level',    # Force vertical recognition
       horizontal_dims=['xi', 'eta']  # Force horizontal recognition
   )

2D vs 3D Regridding
-------------------

**2D Regridding (default):**

Applies to data without vertical structure:

.. code-block:: python

   # 2D surface data
   regridder = Regridder('surface_temp.nc', 'r180x90')
   regridded = regridder.regrid(data)

**3D Regridding:**

For data with vertical levels, specify vertical dimension:

.. code-block:: python

   # 3D atmospheric data
   regridder = Regridder(
       'atmos_data.nc', 'target.nc',
       vertical_dim='plev'
   )

**Automatic 3D Detection with NaN Analysis:**

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
   
   # Lenient masking - cells with >10% coverage  
   regridder = Regridder(
       'source.nc', 'target.nc',
       method='con',
       remap_area_min=0.1
   )

Output Formatting
-----------------

**Dimension Transposition:**

.. code-block:: python

   # Default: vertical dim moved before spatial dims
   # Output order: (time, lev, lat, lon)
   regridder = Regridder('data.nc', 'target.nc', transpose=True)
   
   # Preserve original dimension order
   regridder = Regridder('data.nc', 'target.nc', transpose=False)

**Coordinate Handling:**

The regridder automatically:

* Updates latitude/longitude coordinates with proper metadata
* Preserves non-spatial coordinates (time, ensemble members, etc.)
* Handles coordinate bounds for conservative methods
* Converts coordinate units and adds standard attributes

Performance Optimization
-------------------------

**Memory Management:**

.. code-block:: python

   # For large datasets, use chunking-aware processing
   import dask
   with dask.config.set(scheduler='threads', num_workers=4):
       regridded = regridder.regrid(large_dataset)

**CDO Parallelization:**

.. code-block:: python

   # Use CDO's built-in parallelization
   regridder = Regridder(
       'source.nc', 'target.nc',
       cdo_options=['-P', '8']  # 8 parallel threads
   )

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
       vertical_dim='plev',
       transpose=True
   )
   regridded = regridder.regrid(atmospheric_data)

**Ocean Data with NaN Detection:**

.. code-block:: python

   # Let smmregrid automatically detect depth dimension from NaN patterns
   regridder = Regridder(
       'ocean_model.nc', 'obs_locations.nc', 
       method='con',
       check_nan=True,
       remap_area_min=0.3  # Lenient masking for sparse ocean data
   )

**High-Performance Regridding:**

.. code-block:: python

   # Pre-compute weights for multiple uses
   from smmregrid import CdoGenerate
   
   generator = CdoGenerate('source.nc', 'target.nc')
   weights = generator.weights(
       method='con2',  # Second-order conservative
       nproc=8         # Parallel weight generation
   )
   
   # Fast regridding with pre-computed weights
   regridder = Regridder(weights=weights, transpose=False)
   
   # Process multiple files efficiently
   for file in file_list:
       data = xr.open_dataset(file)
       regridded = regridder.regrid(data)
       regridded.to_netcdf(f'regridded_{file}')

**Custom Grid Recognition:**

.. code-block:: python

   # Force recognition of non-standard dimension names
   regridder = Regridder(
       'custom_model.nc', 'standard_grid.nc',
       horizontal_dims=['xi_rho', 'eta_rho'],  # ROMS ocean model
       vertical_dim='s_rho',                   # Sigma coordinates
       loglevel='DEBUG'
   )
