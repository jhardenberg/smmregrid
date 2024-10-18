# GridType class to gather all information about grids with shared dimensions
import xarray as xr

# default spatial dimensions and vertical coordinates
DEFAULT_DIMS = {
    'horizontal': ['i', 'j', 'x', 'y', 'lon', 'lat', 'longitude', 'latitude',
                   'cell', 'cells', 'ncells', 'values', 'value', 'nod2', 'pix', 'elem',
                   'nav_lon', 'nav_lat'],
    'vertical': ['lev', 'nz1', 'nz', 'depth', 'depth_full', 'depth_half'],
    'time': ['time']
}


class GridType:
    def __init__(self, dims, extra_dims=None, weights=None):
        """
        Initializes a GridType object carrying grid-specific information required by smmregrid.

        Args:
            dims (list): A list of default dimensions for the grid (e.g., ['time', 'lat', 'lon']).
            extra_dims (dict, optional): A dictionary including keys 'vertical', 'time', and 'horizontal'
                                          that can be used to extend the default dimensions. Defaults to None.
            weights (any, optional): Weights used in regridding. The format and purpose depend on the
                                     regridding method. Defaults to None.

        Attributes:
            dims (list): The dimensions defined for the grid.
            horizontal_dims (list): The identified horizontal dimensions from the input.
            vertical_dim (str or None): The identified vertical dimension, if applicable.
            time_dims (list): The identified time dimensions from the input.
            variables (dict): A dictionary holding identified variables and their coordinates.
            bounds (list): A list of bounds variables identified in the dataset.
            masked (any): Placeholder for masked data (to be defined later).
            weights (any): The weights associated with the grid, if provided.
            cdo_weights (bool): A flag indicating if weights are provided.
            weights_matrix (any): Placeholder for a weights matrix (to be defined later).
            level_index (str): A string used to identify the index of the levels.

        Raises:
            ValueError: If multiple vertical dimensions are identified during initialization.
        """

        # key definitions
        self.dims = dims
        default_dims = self._handle_default_dimensions(extra_dims)
        self.horizontal_dims = self._identify_dims('horizontal', default_dims)
        self.vertical_dim = self._identify_dims('vertical', default_dims)
        self.time_dims = self._identify_dims('time', default_dims)
        self.variables = {}
        self.bounds = []

        # used by Regrid class
        self.masked = None
        self.weights = weights
        self.cdo_weights = bool(weights)
        self.weights_matrix = None
        self.level_index = "idx_"

    def _handle_default_dimensions(self, extra_dims):
        """
        Extend the default dimensions based on the provided extra dimensions.

        Args:
            extra_dims (dict): A dictionary that can include 'vertical', 'time', and
                               'horizontal' keys for extending dimensions.

        Returns:
            dict: An updated dictionary of default dimensions that includes any extensions specified
                  in extra_dims.

        Notes:
            If extra_dims is None, the default dimensions remain unchanged.
        """

        if extra_dims is None:
            return DEFAULT_DIMS

        update_dims = DEFAULT_DIMS
        for dim in extra_dims.keys():
            if extra_dims[dim]:
                update_dims[dim] = update_dims[dim] + extra_dims[dim]
        return update_dims

    def __eq__(self, other):
        """
        Check for equality between two GridType instances.

        Args:
            other (GridType): Another GridType instance to compare against.

        Returns:
            bool: True if the dimensions of both instances are equal, False otherwise.
        """
        if isinstance(other, GridType):
            return self.dims == other.dims
        return False

    def __hash__(self):
        """
        Generate a hash value for the GridType instance based on its dimensions.

        Returns:
            int: A hash value representing the dimensions of the GridType instance.
        """
        return hash(self.dims)

    def _identify_dims(self, axis, default_dims):
        """
        Identify dimensions along a specified axis.

        Args:
            axis (str): The axis to check ('horizontal', 'vertical', or 'time').
            default_dims (dict): The dictionary of default dimensions to check against.

        Returns:
            list or str: A list of identified dimensions or a single identified vertical dimension.
                          Returns None if no dimensions are identified.

        Raises:
            ValueError: If more than one vertical dimension is identified.
        """
        identified_dims = list(set(self.dims).intersection(default_dims[axis]))
        if axis == 'vertical':
            if len(identified_dims) > 1:
                raise ValueError(f'Only one vertical dimension can be processed at the time: check {identified_dims}')
            if len(identified_dims) == 1:
                identified_dims = identified_dims[0]  # unlist the single vertical dimension
        return identified_dims if identified_dims else None

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
            if (var.endswith('_bnds') or var.endswith('_bounds')) and 'time' not in var:
                bounds_variables.append(var)

        return bounds_variables

    # def identify_sizes(self, data):
    #    """
    #    Identify the sizes of the dataset based on the horizontal dimensions.
    #    """
    #
    #    if self.horizontal_dims:
    #        self.horizontal_sizes  = [data.sizes[x] for x in self.horizontal_dims]

    def _identify_variable(self, var_data, var_name=None):
        """Helper function to process individual variables.

        Args:
            var_data (xr.DataArray): An xarray DataArray containing variable data.
            var_name (str, optional): The name of the variable. If None, uses the name from var_data.

        Updates:
            self.variables: Updates the variables dictionary with the variable's coordinates.
        """
        if set(var_data.dims) == set(self.dims):
            self.variables[var_name or var_data.name] = {
                'coords': list(var_data.coords),
            }

    def identify_variables(self, data):
        """
        Identify variables in the provided data that match the defined dimensions.

        Args:
            data (xr.Dataset or xr.DataArray): The input data from which to identify variables.

        Raises:
            TypeError: If the input data is neither an xarray Dataset nor DataArray.

        Updates:
            self.variables: Updates the variables dictionary with identified variables and their coordinates.
            self.bounds: Updates the bounds list with identified bounds variables from the dataset.
        """

        if not isinstance(data, (xr.Dataset, xr.DataArray)):
            raise TypeError("Unsupported data type. Must be an xarray Dataset or DataArray.")

        if isinstance(data, xr.Dataset):
            for var in data.data_vars:
                self._identify_variable(data[var], var)
            self.bounds = self._identify_spatial_bounds(data)

        elif isinstance(data, xr.DataArray):
            self._identify_variable(data)

    # def _identify_grid_type(self, grid_key):
    #     """
    #     Determines the grid type (e.g., structured, unstructured, curvilinear).
    #     This could be expanded based on more detailed metadata inspection.
    #     """
    #     horizontal_dims = self._identify_horizontal_dims(grid_key)
    #     if 'mesh' in self.dataset.attrs.get('grid_type', '').lower():
    #         return 'unstructured'
    #     elif any('lat' in coord and 'lon' in coord for coord in horizontal_dims):
    #         return 'regular'
    #     elif 'curvilinear' in self.dataset.attrs.get('grid_type', '').lower():
    #         return 'curvilinear'
    #     else:
    #         return 'unknown'
