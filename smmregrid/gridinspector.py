"""GridInspector class module"""

import warnings
import xarray as xr
from smmregrid.log import setup_logger
from .gridtype import GridType
from .util import detect_grid


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
        self.data = data
        self.cdo_weights = cdo_weights
        self.clean = clean
        self.grids = []  # List to hold all grids info
        self.loggy.debug('Extra_dims are %s', extra_dims)
        self.loggy.debug('Clean flag is %s and cdo_weights init is %s', clean, cdo_weights)

    def _inspect_grids(self):
        """
        Inspects the dataset and identifies different grids.
        """

        if isinstance(self.data, xr.Dataset):
            self.data.map(self._inspect_dataarray_grid, keep_attrs=False)
        elif isinstance(self.data, xr.DataArray):
            self._inspect_dataarray_grid(self.data)
        else:
            raise TypeError('Data supplied is neither xarray Dataset or DataArray')

        # get variables associated to the grid
        for gridtype in self.grids:
            self.identify_variables(gridtype)
        
        # get grid format
        self._identify_grid_format(self.data)

    def _inspect_dataarray_grid(self, data_array):
        """
        Helper method to inspect a single DataArray and identify its grid type.
        """
        grid_key = tuple(data_array.dims)
        gridtype = GridType(dims=grid_key, extra_dims=self.extra_dims)
        if gridtype not in self.grids:
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

    def get_grid_info(self):
        """
        Returns the gridtype object
        """
        warnings.warn(
            "get_grid_info() is deprecated and will be removed in future versions. "
            "Please use get_gridtype() instead.",
            DeprecationWarning
        )

        return self.get_gridtype()

    def get_gridtype(self):
        """
        Returns detailed information about all the grids in the dataset.
        """

        if isinstance(self.data, str):
            self.data = xr.open_dataset(self.data)

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
            elif all('bnds' in variable for variable in gridtype.variables):
                removed.append(gridtype)  # Add to removed list
                self.loggy.info('Removing the grid defined by %s with variables containing "bnds"',
                                gridtype.dims)
            elif all('bounds' in variable for variable in gridtype.variables):
                removed.append(gridtype)  # Add to removed list
                self.loggy.info('Removing the grid defined by %s with variables containing "bounds"',
                                gridtype.dims)

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

        if not isinstance(self.data, (xr.Dataset, xr.DataArray)):
            raise TypeError("Unsupported data type. Must be an xarray Dataset or DataArray.")

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
            if (var.endswith('_bnds') or var.endswith('_bounds') or var == 'vertices') and 'time' not in var:
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
                    gridtype.kind = detect_grid(data[variables])
                elif isinstance(data, xr.DataArray):
                    gridtype.kind = detect_grid(data)

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
