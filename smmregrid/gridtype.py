# GridType class to gather all information about grids with shared dimensions

# default spatial dimensions and vertical coordinates
DEFAULT_DIMS = {
    'horizontal': ['i', 'j', 'x', 'y', 'lon', 'lat', 'longitude', 'latitude',
                   'cell', 'cells', 'ncells', 'values', 'value', 'nod2', 'pix', 'elem',
                   'nav_lon', 'nav_lat', 'rgrid'],
    'vertical': ['lev', 'nz1', 'nz', 'depth', 'depth_full', 'depth_half'],
    'time': ['time', 'time_counter'],
}


class GridType:
    """Fundamental GridType object"""

    def __init__(self, dims, extra_dims=None, override=False, weights=None):
        """
        Initializes a GridType object carrying grid-specific information required by smmregrid.

        Args:
            dims (list): A list of default dimensions for the grid (e.g., ['time', 'lat', 'lon']).
            extra_dims (dict, optional): A dictionary including keys 'vertical', 'time', and 'horizontal'
                                          that can be used to extend the default dimensions. Defaults to None.
            override (bool, optional): If True, it will override the default dimensions with the provided ones.
                                         Defaults to False.
            weights (any, optional): CDO weights used in regridding. It will initiate the object in a different
                                         way assuming single-gridtype objects. Defaults to None.

        Attributes:
            horizontal_dims (list): The identified horizontal dimensions from the input.
            vertical_dim (str or None): The identified vertical dimension, if applicable.
            dims (list): The dimensions defined for the grid. A combination of horizontal and vertical. 
            time_dims (list): The identified time dimensions from the input.
            other_dims (list): The dimensions which are there but are not identified automatically.
            variables (dict): A dictionary holding identified variables and their coordinates.
            bounds (list): A list of bounds variables identified in the dataset.
            masked (any): Placeholder for masked data (to be defined later).
            weights (any): The weights associated with the grid, if provided.
            weights_matrix (any): Placeholder for a weights matrix (to be defined later).
            level_index (str): A string used to identify the index of the levels.

        """

        # safety checks
        #if not isinstance(dims, list):
        #    raise TypeError("dims must be a list of dimension names.")
        if extra_dims is not None and not isinstance(extra_dims, dict):
            raise TypeError("extra_dims must be a dictionary or None.")

        default_dims = self._handle_default_dimensions(extra_dims, override=override)
        self.horizontal_dims = self._identify_dims('horizontal', dims, default_dims)
        self.vertical_dim = self._identify_dims('vertical', dims, default_dims)
        self.dims = (self.horizontal_dims or []) + ([self.vertical_dim] if self.vertical_dim else [])
        self.time_dims = self._identify_dims('time', dims, default_dims)
        self.other_dims = self._identify_other_dims(dims)

        # used by GridInspector
        self.variables = {} # dictionary of variables and their coordinates
        self.bounds = [] # list of bounds variables
        self.kind = None  # which kind of grid, regular, guassian, curvilinear, etc.

        # used by Regrid class
        self.masked = None
        self.weights = weights
        self.weights_matrix = None
        self.level_index = "idx_"

    def _handle_default_dimensions(self, extra_dims, override=False):
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
        
        if override:
            return extra_dims

        update_dims = DEFAULT_DIMS
        for dim in extra_dims.keys():
            if extra_dims[dim]:
                update_dims[dim] = list(set(update_dims[dim] + extra_dims[dim]))
        return update_dims
    
    def __repr__(self):
        """
        Return a string representation of the GridType instance. 
        Only includes attributes with values.
        """

        attributes = ', '.join(f"{key}={value}" for key, value in self.__dict__.items() if value is not None)
        return f"{self.__class__.__name__}({attributes})"

    def __eq__(self, other):
        """
        Check for equality between two GridType instances.

        Args:
            other (GridType): Another GridType instance to compare against.

        Returns:
            bool: True if both self.dims are equal for both instances, False otherwise.
        """
        if isinstance(other, GridType):
            return set(self.dims) == set(other.dims)
        return False

    def __hash__(self):
        """
        Generate a hash value for the GridType instance based on its dimensions.

        Returns:
            int: A hash value representing the dimensions of the GridType instance.
        """
        return hash(tuple(self.dims))

    def _identify_dims(self, axis, dims, default_dims):
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

        # Check if the axis is valid
        if axis not in ['horizontal', 'vertical', 'time']:
            raise ValueError(f"Invalid axis '{axis}'. Must be one of 'horizontal', 'vertical', or 'time'.")
        
        # Check if the axis is in the default dimensions
        if axis not in default_dims:
            return None
        
        # Identify dimensions based on the provided axis
        identified_dims = list(set(dims).intersection(default_dims[axis]))
        if axis == 'vertical':
            if len(identified_dims) > 1:
                raise ValueError(f'Only one vertical dimension can be processed at the time: check {identified_dims}')
            if len(identified_dims) == 1:
                identified_dims = identified_dims[0]  # unlist the single vertical dimension
        return identified_dims if identified_dims else None

    def _identify_other_dims(self, dims):
        """
        Calculate and return the dimensions that are not part of horizontal_dims, vertical_dim, or time_dims.

        Returns:
            set: A set of unused dimensions that are not part of horizontal, vertical, or time dimensions.
        """
        # Safely convert identified dimensions to sets, handling None values
        horizontal_dims = set(self.horizontal_dims) if self.horizontal_dims else set()
        vertical_dim = {self.vertical_dim} if self.vertical_dim else set()  # Use {} for single value
        time_dims = set(self.time_dims) if self.time_dims else set()

        # Return the unused dimensions by subtracting used dimensions from all dimensions
        return list(set(dims) - (horizontal_dims | vertical_dim | time_dims))
    


