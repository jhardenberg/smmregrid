# GridType class to gather all information about grids with shared dimensions
import xarray as xr

# default spatial dimensions and vertical coordinates
DEFAULT_DIMS = {
    'horizontal':  ['i', 'j', 'x', 'y', 'lon', 'lat', 'longitude', 'latitude',
                      'cell', 'cells', 'ncells', 'values', 'value', 'nod2', 'pix', 'elem',
                      'nav_lon', 'nav_lat'],
    'vertical': ['lev', 'nz1', 'nz', 'depth', 'depth_full', 'depth_half'],
    'time': ['time']
}

class GridType:
    def __init__(self, dims, extra_dims=None, weights=None):
        """
        GridType object carrying the grid-specific information required by smmregrid
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
        Extend the default dimensions according some vertical or horizontal

        Args:
            extra_dims (dict): Including 'vertical', 'time', 'horizontal' keys
        """

        if extra_dims is None:
            return DEFAULT_DIMS

        update_dims = DEFAULT_DIMS
        for dim in extra_dims.keys():
            update_dims[dim] = update_dims[dim] + extra_dims[dim]
        return update_dims
                     
    def __eq__(self, other):
        # so far equality based on dims only
        if isinstance(other, GridType):
            return self.dims == other.dims
        return False

    def __hash__(self):
        return hash(self.dims)

    def _identify_dims(self, axis, default_dims):
        """
        Generic dimension identifier method.
        Takes a list of default dimensions to check against.
        """
        identified_dims = list(set(self.dims).intersection(default_dims[axis]))
        if axis == 'vertical':
            if len(identified_dims)>1:
                raise ValueError(f'Only one vertical dimension can be processed at the time: check {identified_dims}')
            if len(identified_dims)==1:
                identified_dims=identified_dims[0] #unlist the single vertical dimension
        return identified_dims if identified_dims else None
    
    def _identify_spatial_bounds(self, data):
        """
        Find all bounds variables in the dataset by looking for variables
        with '_bnds' or '_bounds' suffix.
        """
        bounds_variables = []

        for var in data.data_vars:
            if (var.endswith('_bnds') or var.endswith('_bounds')) and 'time' not in var:
                # store all the bounds fro each grid. not fancy, but effective
                #boundvar = var.split('_')[0]
                #if boundvar in self.dims:
                bounds_variables.append(var)

        return bounds_variables
    
    #def identify_sizes(self, data):
    #    """
    #    Idenfity the sizes of the dataset
    #    """
    #
    #    if self.horizontal_dims:
    #        self.horizontal_sizes  = [data.sizes[x] for x in self.horizontal_dims]

    def _identify_variable(self, var_data, var_name=None):
        """Helper function to process individual variables."""
        if set(var_data.dims) == set(self.dims):
            self.variables[var_name or var_data.name] = {
                'coords': list(var_data.coords),
            }

    def identify_variables(self, data):
        """
        Identify the variables in the data that match the given dimensions and
        collect their coordinate information.
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