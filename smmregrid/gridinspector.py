"""GridInspector class module"""

import os
import xarray as xr
import numpy as np
from smmregrid.log import setup_logger
from .gridtype import GridType
from .util import find_coord

# Define coordinate names for latitude and longitude
LAT_COORDS = ["lat", "latitude", "nav_lat"]
LON_COORDS = ["lon", "longitude", "nav_lon"]


class GridInspector():
    """Class to investigate data and detect its GridType() object"""

    def __init__(self, data, cdo_weights=False, extra_dims=None,
                 clean=True, loglevel='warning'):
        """
        GridInspector class to detect information on the data, based on GridType class

        Parameters:
            data (xr.Datase or xr.DataArray): The input dataset.
            clean (bool): apply the cleaning of grids which are assumed to be not relevant 
                          for regridding purposes (e.g. bounds)
            cdo_weights (bool): if the data provided are cdo weights instead of data to be regridded
            extra_dims(dict): Extra dimensions to be added and passed to GridType
            loglevel: The loglevel that you want you use
        """
        self.loggy = setup_logger(name='smmregrid.GridInspect', level=loglevel)
        self.loglevel = loglevel
        self.extra_dims = extra_dims
        self.data = self._open_data(data)
        self.cdo_weights = cdo_weights
        self.clean = clean
        self.grids = []  # List to hold all grids info
        self.loggy.debug('Extra_dims are %s', extra_dims)
        self.loggy.debug('Clean flag is %s and cdo_weights init is %s', clean, cdo_weights)

    def _open_data(self, data):
        """
        Open the data from a file path if it is a string and the file exists
        """
        if isinstance(data, (xr.Dataset, xr.DataArray)):
            self.loggy.info('Data is already an xarray Dataset or DataArray')
            return data
        if isinstance(data, str):
            if os.path.exists(data):
                self.loggy.info('Data is a file path, opening xr.dataset')
                return xr.open_dataset(data)
            raise FileNotFoundError(f'File {data} not found')
        raise TypeError('Data supplied is neither xarray Dataset or DataArray')

    def _inspect_grids(self):
        """
        Inspects the dataset and identifies different grids.
        """

        if isinstance(self.data, xr.Dataset):
            self.data.map(self._inspect_dataarray_grid, keep_attrs=False)
        if isinstance(self.data, xr.DataArray):
            self._inspect_dataarray_grid(self.data)


        # get variables associated to the grid
        for gridtype in self.grids:
            self.identify_variables(gridtype)
        
        # get grid format
        self._identify_grid_format(self.data)

    def _inspect_dataarray_grid(self, data_array):
        """
        Helper method to inspect a single DataArray and identify its grid type.
        If the data_array is a bounds variable, it is not added to the grids list.
        """
        grid_key = tuple(data_array.dims)
        gridtype = GridType(dims=grid_key, extra_dims=self.extra_dims)
        if gridtype not in self.grids and not self._is_bounds(data_array.name):
            self.grids.append(gridtype)

    def _inspect_weights(self):
        """
        Return basic information about CDO weights
        """

        gridtype = GridType(dims=[], weights=self.data)

        # get vertical info from the weights coords if available
        if self.data.coords:
            vertical_dim = list(self.data.coords)[0]
            self.loggy.debug('Vertical dimension read from weights: %s', vertical_dim)
            gridtype.vertical_dim = vertical_dim

        self.grids.append(gridtype)

    def get_gridtype(self):
        """
        Returns detailed information about all the grids in the dataset.
        """

        if self.cdo_weights:
            self.loggy.info('CDO weights are used to define the grid')
            self._inspect_weights()
        else:
            self._inspect_grids()

        if self.clean:
            self._clean_grids()

        # Log details about identified grids
        for gridtype in self.grids:
            self._log_grid_details(gridtype)

        return self.grids

    def _log_grid_details(self, gridtype):
        """Log detailed information about a grid."""
        self.loggy.debug('Grid details: %s', gridtype.dims)
        if gridtype.horizontal_dims:
            self.loggy.debug('  Horizontal dims: %s', gridtype.horizontal_dims)
        if gridtype.vertical_dim:
            self.loggy.debug('  Vertical dim: %s', gridtype.vertical_dim)
        if gridtype.time_dims:
            self.loggy.debug('  Time dims: %s', gridtype.time_dims)
        if gridtype.other_dims:
            self.loggy.debug('  Other dims: %s', gridtype.other_dims)
        self.loggy.debug('  Variables: %s', list(gridtype.variables.keys()))
        self.loggy.debug('  Bounds: %s', gridtype.bounds)

    def _clean_grids(self):
        """
        Remove degenerate grids which are used by not relevant variables
        """
        removed = []

        # Iterate through grids
        for gridtype in self.grids:
            # Check if any variable in the grid contains 'bnds'
            if not gridtype.dims:
                removed.append(gridtype)
                self.loggy.info('Removing the grid defined by %s with with no spatial dimensions',
                                gridtype.dims)
            #elif all('bnds' in variable for variable in gridtype.variables):
            #    removed.append(gridtype)  # Add to removed list
            #    self.loggy.info('Removing the grid defined by %s with variables containing "bnds"',
            #                     gridtype.dims)
            #elif all('bounds' in variable for variable in gridtype.variables):
            #    removed.append(gridtype)  # Add to removed list
            #    self.loggy.info('Removing the grid defined by %s with variables containing "bounds"',
            #                    gridtype.dims)

        for remove in removed:
            self.grids.remove(remove)


    def _identify_variable(self, gridtype, var_data, var_name=None):
        """Helper function to process individual variables.

        Args:
            gridtype (GridType): The Gridtype object of originally inspected from the data
            var_data (xr.DataArray): An xarray DataArray containing variable data.
            var_name (str, optional): The name of the variable. If None, uses the name from var_data.

        Updates:
            self.variables: Updates the variables dictionary with the variable's coordinates.
        """
        datagridtype = GridType(var_data.dims, extra_dims=self.extra_dims)

        if datagridtype == gridtype:
            gridtype.variables[var_name or var_data.name] = {
                'coords': list(var_data.coords),
            }

    def identify_variables(self, gridtype):
        """
        Identify variables in the provided data that match the defined dimensions.

        Args:
            gridtype (GridType): The Gridtype object of originally inspected from the data

        Raises:
            TypeError: If the input data is neither an xarray Dataset nor DataArray.

        Updates:
            self.variables: Updates the variables dictionary with identified variables and their coordinates.
            self.bounds: Updates the bounds list with identified bounds variables from the dataset.
        """

        if isinstance(self.data, xr.Dataset):
            for var in self.data.data_vars:
                self._identify_variable(gridtype, self.data[var], var)
            gridtype.bounds = self._identify_spatial_bounds(self.data)
            gridtype.variables = {
                key: value
                for key, value in gridtype.variables.items()
                if key not in set(gridtype.bounds)
            }

        elif isinstance(self.data, xr.DataArray):
            self._identify_variable(gridtype, self.data)

    def _is_bounds(self, var):
        """
        Check if a variable is a bounds variable.
        """
        return var.endswith('_bnds') or var.endswith('_bounds') or var == 'vertices' and 'time' not in var

    def _identify_spatial_bounds(self, data):
        """
        Identify bounds variables in the dataset by checking variable names.

        Args:
            data (xr.Dataset): An xarray dataset containing data variables.

        Returns:
            list: A list of bounds variable names identified in the dataset.
        """
        bounds_variables = []

        for var in data.data_vars:
            if self._is_bounds(var):
                bounds_variables.append(var)

        return bounds_variables
    
    def _identify_grid_format(self, data):
        """
        Identify the grid format based on the provided data.

        Args:
            data (xr.Dataset or xr.DataArray): The input dataset.

        Returns:
            str: The identified grid format.
        """
        for gridtype in self.grids:
            variables = list(gridtype.variables.keys())
            if variables:
                if isinstance(data, xr.Dataset):
                    gridtype.kind = self.detect_grid(data[variables])
                elif isinstance(data, xr.DataArray):
                    gridtype.kind = self.detect_grid(data)

    @staticmethod
    def get_gridtype_attr(gridtypes, attr):
        """Helper compact tool to extra gridtypes information"""
        out = []
        for gridtype in gridtypes:
            value = getattr(gridtype, attr, None)
            if isinstance(value, (list, tuple)):
                out.extend(value)
            elif isinstance(value, dict):
                out.extend(value.keys())
            elif isinstance(value, str):
                out.append(value)

        return list(dict.fromkeys(out))

    def detect_grid(self, data, lat='lat', lon='lon'):
        """
        Detect the grid type based on the structure of the data.

        Args:
            data (xr.Dataset or xr.DataArray): The input dataset.
            lat (str): The name of the latitude coordinate.
            lon (str): The name of the longitude coordinate.

        Returns:
            str: The identified grid type.
        """

        lon = find_coord(data, set(LON_COORDS + [lon]))
        lat = find_coord(data, set(LAT_COORDS + [lat]))

        if self.is_healpix_from_attribute(data):
            return "HEALPix"

        if not lat or not lon:
            return "Unknown"

        # 2D coord-dim dependency
        if data[lat].ndim == 2 and data[lon].ndim == 2:
            return "Curvilinear"

        # 1D coord-dim dependency
        if data[lat].ndim == 1 and data[lon].ndim == 1:

            # Regular: latitude and longitude depend on different coordinates
            if data[lat].dims != data[lon].dims:

                lat_diff = np.diff(data[lat].values)
                lon_diff = np.diff(data[lon].values)

                # Regular: latitude and longitude equidistant
                if np.allclose(lat_diff, lat_diff[0]) and np.allclose(lon_diff, lon_diff[0]):
                    return "Regular"

                # Gaussian: longitude equidistant, latitude not
                if not np.allclose(lat_diff, lat_diff[0]) and np.allclose(lon_diff, lon_diff[0]):
                    return "GaussianRegular"
                
                return "UndefinedRegular"
            
            pix = data[lat].size
            if pix % 12 == 0 and (pix // 12).bit_length() - 1 == np.log2(pix // 12):
                return "HEALPix"
            
            # Guess gaussian reduced: increasing number of latitudes from -90 to 0
            lat_values = data[lat].where(data[lat]<0).values
            lat_values=lat_values[~np.isnan(lat_values)]
            _, counts = np.unique(lat_values, return_counts=True)
            gaussian_reduced = np.all(np.diff(counts)>0)
            if gaussian_reduced:
                return "GaussianReduced"

            # None of the above cases
            return "Unstructured"

        return "Unknown"

    @staticmethod
    def is_healpix_from_attribute(data):
        """
        Determine if the given xarray Dataset or DataArray uses a HEALPix grid.

        Returns:
            bool: True if HEALPix grid detected, False otherwise.
        """

        # Attribute-based checks
        if isinstance(data, xr.Dataset):
            if "healpix" in data.variables:
                return True
            for var in data.data_vars:
                if data[var].attrs.get('grid_mapping') == 'healpix':
                    return True
        elif isinstance(data, xr.DataArray):
            if data.attrs.get('grid_mapping') == 'healpix':
                return True

        return False
    