import xarray as xr
from smmregrid.log import setup_logger
from .gridtype import GridType

class GridInspector():

    def __init__(self, data, cdo_weights=False, clean=True, loglevel='warning'):
        """
        GridInspector class to detect information on the data, based on GridType class
        
        Parameters:
        data (xr.Datase or xr.DataArray): The input dataset.
        clean (bool): apply the cleaning of grids which are assumed to be not relevant
        loglevel: The loglevel that you want you use
        """
        self.loggy = setup_logger(name='smmregrid.GridInspect', level=loglevel)
        self.loglevel = loglevel
        self.data = data
        self.cdo_weights = cdo_weights
        self.clean = clean
        self.grids = []  # List to hold all grids info
    
    def _inspect_grids(self):
        """
        Inspects the dataset and identifies different grids.
        """

        if isinstance(self.data, xr.Dataset):
            for variable in self.data.data_vars:
                grid_key = tuple(self.data[variable].dims)
                gridtype = GridType(dims=grid_key)
                if gridtype not in self.grids:
                    self.grids.append(gridtype)

        elif isinstance(self.data, xr.DataArray):
            grid_key = tuple(self.data.dims)
            self.grids.append(GridType(grid_key))

        for gridtype in self.grids:
            gridtype.identify_variables(self.data)
            #gridtype.identify_sizes(self.data)

    def _inspect_weights(self):
        """
        Return basic information about CDO weights
        """
        
        self.grids.append(GridType(dims=[], weights=self.data))

    def get_grid_info(self):
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

        #self.loggy.info('Grids that have been identifed are: %s', self.grids.)
        for gridtype in self.grids:
            self.loggy.debug('More details on gridtype %s:', gridtype.dims)
            if gridtype.horizontal_dims:
                self.loggy.debug('    Space dims are: %s', gridtype.horizontal_dims)
            if gridtype.vertical_dim:
                self.loggy.debug('    Vertical dims are: %s', gridtype.vertical_dim)
            self.loggy.debug('    Variables are: %s', gridtype.variables)
        return self.grids
    
    def _clean_grids(self):
        """
        Remove degenerate grids which are used by not relevant variables
        """

        # Initialize list to store grids to be removed
        removed = []

        # Iterate through grids
        for gridtype in self.grids:
            # Check if any variable in the grid contains 'bnds'
            if any('bnds' in variable for variable in gridtype.variables):
                removed.append(gridtype)  # Add to removed list
                self.loggy.info('Removing the grid defined by %s with variables containing "bnds"', gridtype.dims)

        #self.loggy.debug("Grids that will be removed are: %s", removed)

        for remove in removed:
            self.grids.remove(remove)

    #def get_variable_grids(self): #TO BE FIXED
    #    """
    #    Return a dictionary with the variable - grids pairs
    #    """
    #    all_grids = {}
    #    for gridtype in self.grids:
    #        for variable in gridtype.variables:
    #            all_grids[variable] = grid_dims
    #    return all_grids