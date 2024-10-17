"""GridInspector class module"""

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
        cdo_weights (bool): if the data provided are cdo weights instead of data to be regridded
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
                self._inspect_dataarray_grid(self.data[variable])
        elif isinstance(self.data, xr.DataArray):
            self._inspect_dataarray_grid(self.data)
        else:
            raise ValueError('Data supplied is neither xarray Dataset or DataArray')

        for gridtype in self.grids:
            gridtype.identify_variables(self.data)
            #gridtype.identify_sizes(self.data)

    def _inspect_dataarray_grid(self, data_array):
        """
        Helper method to inspect a single DataArray and identify its grid type.
        """
        grid_key = tuple(data_array.dims)
        gridtype = GridType(dims=grid_key)
        if gridtype not in self.grids:
            self.grids.append(gridtype)

    def _inspect_weights(self):
        """
        Return basic information about CDO weights
        """
        
        gridtype = GridType(dims=[], weights=self.data)

        # get vertical info from the weights coords if available
        if self.data.coords:
            gridtype.vertical_dim = list(self.data.coords)[0]

        self.grids.append(gridtype)

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
        removed = []

        # Iterate through grids
        for gridtype in self.grids:
            # Check if any variable in the grid contains 'bnds'
            if any('bnds' in variable for variable in gridtype.variables):
                removed.append(gridtype)  # Add to removed list
                self.loggy.info('Removing the grid defined by %s with variables containing "bnds"', gridtype.dims)
            if any('bounds' in variable for variable in gridtype.variables):
                removed.append(gridtype)  # Add to removed list
                self.loggy.info('Removing the grid defined by %s with variables containing "bounds"', gridtype.dims)
    
        for remove in removed:
            self.grids.remove(remove)

    #def get_variable_grids(self):
    #    """
    #    Return a dictionary with the variable - grids pairs
    #    """
    #    all_grids = {}
    #    for gridtype in self.grids:
    #        for variable in gridtype.variables:
    #            all_grids[variable] = grid_dims
    #    return all_grids