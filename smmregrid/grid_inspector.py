import xarray as xr
from collections import defaultdict
from smmregrid.log import setup_logger

# default spatial dimensions and vertical coordinates
default_horizontal_dims = ['i', 'j', 'x', 'y', 'lon', 'lat', 'longitude', 'latitude',
                      'cell', 'cells', 'ncells', 'values', 'value', 'nod2', 'pix', 'elem',
                      'nav_lon', 'nav_lat']
default_vert_dims = ['lev', 'nz1', 'nz', 'depth', 'depth_full', 'depth_half']

class GridInspector():


    def __init__(self, data, clean=True, loglevel='warning'):
        """
        Initializes the GridInspector with.
        
        Parameters:
        data (xr.Datase or xr.DataArray): The input dataset.
        clean (bool): apply the cleaning of grids which are assumed to be not relevant
        loglevel: The loglevel that you want you use
        """
        self.loggy = setup_logger(name='GridInspect', level=loglevel)
        self.loglevel = loglevel
        self.data = data
        self.clean = clean
        self.grids = defaultdict(dict)  # Dictionary to hold all grids info
        #self._inspect_grids()  # Analyze the grids
    
    def _inspect_grids(self, data_array, var_name):
        """
        Inspects the dataset and identifies different grids.
        """

        grid_key = tuple(data_array.dims)

        # If grid_key (set of dimensions) is new, create a new entry
        if grid_key not in self.grids:
            self.grids[grid_key] = {
                'variables': [],
                'dims': grid_key,
                'horizontal_dims': self._identify_horizontal_dims(grid_key),
                'vertical_dims': self._identify_vertical_dims(grid_key),
                'time_dims': self._identify_time_dims(grid_key)
                #'grid_type': self._identify_grid_type(grid_key)
            }

        # Add the variable to the corresponding grid
        self.grids[grid_key]['variables'].append(var_name)


    def _identify_horizontal_dims(self, grid_key):
        """
        Identifies horizontal coordinates (like latitude and longitude) from the grid_key.
        """
        horizontal_dims = list(set(grid_key).intersection(default_horizontal_dims))
        if not horizontal_dims:
            horizontal_dims = None
        return horizontal_dims

    def _identify_vertical_dims(self, grid_key):
        """
        Identifies vertical coordinates (like depth or altitude) from the grid_key.
        """
        vertical_dims = list(set(grid_key).intersection(default_vert_dims))
        if not vertical_dims:
            vertical_dims = None
        return vertical_dims

    def _identify_time_dims(self, grid_key):
        """
        Identifies time coordinates from the grid_key.
        """
        time_dims = []
        for dim in grid_key:
            if 'time' in dim.lower():
                time_dims.append(dim)
        return time_dims

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
    
    def get_grid_info(self):
        """
        Returns detailed information about all the grids in the dataset.
        """

        if isinstance(self.data, str):
            self.data = xr.open_dataset(self.data)
    
        if isinstance(self.data, xr.Dataset):

            # Loop over all variables to analyze their grids
            for var_name, data_array in self.data.data_vars.items():
                # Get the dimensions that define the grid for this variable
                self._inspect_grids(data_array, var_name)
                
        elif isinstance(self.data, xr.DataArray):
            self._inspect_grids(self.data, self.data.name)

        if self.clean:
            self._clean_grids()

        self.loggy.info('Grids that have been identifed are: %s', self.grids)
        return dict(self.grids)  # Converting defaultdict to dict for a cleaner output
    
    def _clean_grids(self):
        """
        Remove degenerate grids which are used by not relevant variables
        """

        # Initialize list to store grids to be removed
        removed = []

        # Iterate through grids
        for grid, grid_data in self.grids.items():
            # Check if any variable in the grid contains 'bnds'
            if any('bnds' in variable for variable in grid_data['variables']):
                removed.append(grid)  # Add to removed list
                self.loggy.info('Removing the grid defined by %s with variables containing "bnds"', grid)


        self.loggy.debug("Grids that will be removed are: %s", removed)

        for remove in removed:
            del self.grids[remove]

    def get_variable_grids(self):
        """
        Return a dictionary with the variable - grids pairs
        """
        all_grids = {}
        for grid_key, grid_details in self.grids.items():
            for variable in grid_details['variables']:
                all_grids[variable] = grid_key
        return all_grids