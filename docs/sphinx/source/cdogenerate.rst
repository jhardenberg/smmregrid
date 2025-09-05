CdoGenerate Class
=================

The :class:`CdoGenerate` class provides a powerful interface to Climate Data Operators (CDO) 
for generating regridding weights and grid areas. This class handles both 2D and 3D grids and 
supports multiple interpolation methods through CDO's regridding capabilities.

.. autoclass:: smmregrid.CdoGenerate
   :members:
   :undoc-members:

Basic Usage
-----------

To use CdoGenerate, you need to provide a source grid and target grid:

.. code-block:: python

   from smmregrid import CdoGenerate
   
   # Initialize with file paths
   generator = CdoGenerate('source.nc', 'target.nc')
   
   # Or with xarray objects
   import xarray as xr
   source_data = xr.open_dataset('source.nc')
   target_grid = 'r360x180'  # CDO grid description
   generator = CdoGenerate(source_data, target_grid)

Constructor Parameters
----------------------

**Required Parameters:**

* **source_grid** (*str* or *xarray.Dataset*): The source grid from which to generate weights. 
  Can be a file path, xarray dataset, or xarray DataArray.

* **target_grid** (*str* or *xarray.Dataset*): The target grid to which the source grid will be regridded. 
  Can be a file path, xarray dataset, xarray DataArray, or CDO grid description (e.g., 'r360x180').

**Optional Parameters:**

* **cdo** (*str*, default: 'cdo'): Path to the CDO executable. Useful if CDO is installed in a non-standard location.

* **loglevel** (*str*, default: 'warning'): Logging level for output messages. 

* **cdo_extra** (*list*, optional): Additional CDO commands to pass to be used before the usual weights generation commands. 
  For example: ``-selname,va`` to select the variable 'va'.

* **cdo_options** (*list*, optional): CDO options to be applied to specific commands.
  For example: ``['--force']`` to get to force HEALPix computation.

* **cdo_download_path** (*str*, optional): Path where CDO should download grid files if needed.
  Sets the ``CDO_DOWNLOAD_PATH`` environment variable.

* **cdo_icon_grids** (*str*, optional): Path to ICON grid files if working with ICON model data.
  Sets the ``CDO_ICON_GRIDS`` environment variable.

Methods
-------

weights()
~~~~~~~~~

Generate interpolation weights for regridding.

.. code-block:: python

   weights = generator.weights(method='con', extrapolate=True)

**Parameters:**

* **method** (*str*, default: 'con'): Interpolation method. Available options:

  - 'bic': SCRIP Bicubic interpolation
  - 'bil': SCRIP Bilinear interpolation  
  - 'con': SCRIP First-order conservative (default)
  - 'con2': SCRIP Second-order conservative
  - 'dis': SCRIP Distance-weighted average
  - 'laf': YAC Largest area fraction
  - 'nn': Nearest neighbour
  - 'ycon': YAC First-order conservative

* **extrapolate** (*bool*, default: True): Allow extrapolation beyond grid boundaries.

* **remap_norm** (*str*, default: 'fracarea'): Normalization method for conservative remapping:

  - 'fracarea': Normalize by fractional area (default)
  - 'destarea': Normalize by destination area

* **vertical_dim** (*str*, optional): Name of the vertical dimension for 3D weight generation.
  If specified, generates 3D weights for each vertical level. To be used for vertical changing masks.

* **nproc** (*int*, default: 1): Number of processes for parallel 3D weight generation.

**Returns:** *xarray.Dataset* containing the regridding weights and mask information.

areas()
~~~~~~~

Generate grid cell areas for source or target grids.

.. code-block:: python

   # Generate source grid areas
   source_areas = generator.areas(target=False)
   
   # Generate target grid areas  
   target_areas = generator.areas(target=True)

**Parameters:**

* **target** (*bool*, default: False): If False, generates areas for source grid. 
  If True, generates areas for target grid.

**Returns:** *xarray.Dataset* containing grid cell areas with proper units and metadata.

Grid Input Types
----------------

CdoGenerate accepts several types of grid specifications:

**File Paths:**

.. code-block:: python

   generator = CdoGenerate('input.nc', 'target.nc')

**CDO Grid Descriptions:**

.. code-block:: python

   # Regular lon-lat grids
   generator = CdoGenerate('input.nc', 'r360x180')  # 1° resolution
   generator = CdoGenerate('input.nc', 'r720x360')  # 0.5° resolution
   
   # Gaussian grids
   generator = CdoGenerate('input.nc', 'n128')      # T255 equivalent

**Xarray Objects:**

.. code-block:: python

   import xarray as xr
   source = xr.open_dataset('source.nc')
   target = xr.open_dataset('target.nc')
   generator = CdoGenerate(source, target)

3D Weight Generation
--------------------

For 3D grids with vertical levels, specify the vertical dimension:

.. code-block:: python

   # Generate 3D weights with 4 parallel processes
   weights_3d = generator.weights(
       method='con',
       vertical_dim='lev',
       nproc=4
   )

The 3D weight generation:

* Processes each vertical level separately
* Uses multiprocessing for efficiency
* Returns weights with vertical dimension preserved
* Includes level-specific masking information


Examples
--------

**Basic 2D Regridding:**

.. code-block:: python

   from smmregrid import CdoGenerate
   
   # Generate conservative weights
   generator = CdoGenerate('era5.nc', 'r180x90')
   weights = generator.weights(method='con')

**3D Regridding with Vertical Levels:**

.. code-block:: python

   # Generate 3D weights for atmospheric data
   generator = CdoGenerate('model_data.nc', 'obs_grid.nc')
   weights_3d = generator.weights(
       method='bil',
       vertical_dim='plev',
       nproc=6
   )

**Grid Area Calculation:**

.. code-block:: python

   # Calculate source and target areas
   generator = CdoGenerate('source.nc', 'target.nc')
   source_areas = generator.areas(target=False)
   target_areas = generator.areas(target=True)

**Custom CDO Configuration:**

.. code-block:: python

   generator = CdoGenerate(
       'healpix.nc',
       'n256',  # Gaussian grid
       cdo='/opt/cdo/bin/cdo',
       cdo_options=['-f', 'nc4'],
       cdo_extra=['--force'],
       loglevel='debug'
   )
