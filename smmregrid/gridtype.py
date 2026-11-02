"""Fundamental GridType class to gather all information about grids"""

import copy

# horizontal dimensions grouped by grid, from the most to the least reliable: a variable is on
# the first family whose dimensions it has all; a partial match (e.g. a zonal mean) is not a grid
HORIZONTAL_FAMILIES = (
    ('lon', 'lat'), ('longitude', 'latitude'), #standard CF, CMIP6, ERA5, etc.
    ('nav_lon', 'nav_lat'),  # NEMO
    ('rlon', 'rlat'),  # rotated pole (e.g. CORDEX)
    ('nlon', 'nlat'), ('ni', 'nj'),  # CESM/POP, CICE
    ('xh', 'yh'), ('xq', 'yh'), ('xh', 'yq'), ('xq', 'yq'),  # GFDL/MOM6 C-grid: tracer, u, v and corner points
    ('i', 'j'), ('x', 'y'),  # generic curvilinear
    ('ncells',), ('cell',), ('cells',), ('ncol',), ('nod2',), ('elem',), ('pix',),  # unstructured, one dimension
    ('rgrid',), ('values',), ('value',),
    ('i',), ('x',),  # a lone i or x is an unstructured index (e.g. ICON in CMIP6, HEALPix)
)

# default spatial dimensions and vertical coordinates
DEFAULT_DIMS = {
    'horizontal': [dim for family in HORIZONTAL_FAMILIES for dim in family],
    'mask': ['lev', 'nz1', 'nz', 'depth', 'depthu', 'depthv', 'deptht', 'depth_full', 'depth_half'],
    'time': ['time', 'time_counter', 'valid_time', 'forecast_time'],
}


class GridType:
    """Fundamental GridType object"""

    def __init__(self, dims, extra_dims=None, override=False, weights=None):
        """
        Initializes a GridType object carrying grid-specific information required by smmregrid.

        Args:
            dims (list): A list of default dimensions for the grid (e.g., ['time', 'lat', 'lon']).
            extra_dims (dict, optional): A dictionary including keys 'mask', 'time', and 'horizontal'
                                          that can be used to extend the default dimensions. Defaults to None.
            override (bool, optional): If True, it will override the default dimensions with the provided ones.
                                         Defaults to False.
            weights (any, optional): CDO weights used in regridding. It will initiate the object in a different
                                         way assuming single-gridtype objects. Defaults to None.

        Attributes:
            horizontal_dims (list): The identified horizontal dimensions from the input.
            mask_dim (str or None): The identified masked dimension, if applicable.
            dims (list): The dimensions defined for the grid. A combination of horizontal and masked dimensions.
            time_dims (list): The identified time dimensions from the input.
            other_dims (list): The dimensions which are there but are not identified automatically.
            variables (dict): A dictionary holding identified variables and their coordinates.
            bounds (list): A list of bounds variables identified in the dataset.
            masked (any): Placeholder for masked data (to be defined later).
            weights (any): The weights associated with the grid, if provided.
            weights_matrix (any): Placeholder for a weights matrix (to be defined later).
            total_weights (any): Placeholder for the sum of weights per target cell (to be defined later).
            target_coords (dict): Placeholder for the target grid shape, dims and lat/lon (to be defined later).
        """

        # safety checks
        # if not isinstance(dims, list):
        #    raise TypeError("dims must be a list of dimension names.")
        if extra_dims is not None and not isinstance(extra_dims, dict):
            raise TypeError("extra_dims must be a dictionary or None.")

        default_dims = self._handle_default_dimensions(extra_dims, override=override)
        self.horizontal_dims = self._select_horizontal_family(
            self._identify_dims('horizontal', dims, default_dims), extra_dims, override=override)
        self.mask_dim = self._identify_dims('mask', dims, default_dims)
        self.dims = (self.horizontal_dims or []) + ([self.mask_dim] if self.mask_dim else [])
        self.time_dims = self._identify_dims('time', dims, default_dims)
        self.other_dims = self._identify_other_dims(dims)

        # used by GridInspector
        self.variables = {}  # dictionary of variables and their coordinates
        self.bounds = []  # list of bounds variables
        self.kind = None  # which kind of grid, regular, guassian, curvilinear, etc.

        # used by Regrid class
        self.masked = None
        self.weights = weights
        self.weights_matrix = None
        self.total_weights = None
        self.target_coords = None

    def _handle_default_dimensions(self, extra_dims, override=False):
        """
        Extend the default dimensions based on the provided extra dimensions.

        Args:
            extra_dims (dict): A dictionary that can include 'mask', 'time', and
                               'horizontal' keys for extending dimensions.

        Returns:
            dict: An updated dictionary of default dimensions that includes any extensions specified
                  in extra_dims.

        Notes:
            If extra_dims is None, the default dimensions remain unchanged.
            An axis set to False is disabled, e.g. {'mask': False} never identifies a mask dimension.
        """

        if extra_dims is None:
            return DEFAULT_DIMS

        if override:
            return extra_dims

        # Create a deep copy to avoid modifying the global DEFAULT_DIMS
        update_dims = copy.deepcopy(DEFAULT_DIMS)
        # update_dims = DEFAULT_DIMS

        for dim in extra_dims.keys():
            if extra_dims[dim] is False:
                update_dims[dim] = []
            elif extra_dims[dim]:
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
        return hash(frozenset(self.dims))

    def _identify_dims(self, axis, dims, default_dims):
        """
        Identify dimensions along a specified axis.

        Args:
            axis (str): The axis to check ('horizontal', 'mask', or 'time').
            default_dims (dict): The dictionary of default dimensions to check against.

        Returns:
            list or str: A list of identified dimensions or a single identified masked dimension.
                          Returns None if no dimensions are identified.

        Raises:
            ValueError: If more than one masked dimension is identified.
        """

        # Check if the axis is valid
        if axis not in ['horizontal', 'mask', 'time']:
            raise ValueError(f"Invalid axis '{axis}'. Must be one of 'horizontal', 'mask', or 'time'.")

        # Check if the axis is in the default dimensions
        if axis not in default_dims:
            return None

        # Identify dimensions based on the provided axis, in the order of the data
        identified_dims = [dim for dim in dims if dim in default_dims[axis]]
        if axis == 'mask':
            if len(identified_dims) > 1:
                raise ValueError(f'Only one masked dimension can be processed at the time: check {identified_dims}')
            if len(identified_dims) == 1:
                identified_dims = identified_dims[0]  # unlist the single masked dimension
        return identified_dims if identified_dims else None

    def _select_horizontal_family(self, horizontal_dims, extra_dims, override=False):
        """
        Keep only the horizontal dimensions of a single grid, following HORIZONTAL_FAMILIES.

        With override, only the declared dimensions are looked for, so they are used as found.
        Otherwise, dimensions declared in extra_dims come first and are used as found, then the
        first family of HORIZONTAL_FAMILIES whose dimensions are all in the data is used.

        Args:
            horizontal_dims (list): The horizontal dimensions identified in the data.
            extra_dims (dict): The extra dimensions provided by the user.
            override (bool): If the extra dimensions override the default ones.

        Returns:
            list: The horizontal dimensions of the selected family, None if there are none.
        """
        if not horizontal_dims or override:
            return horizontal_dims

        declared = [dim for dim in horizontal_dims if dim in ((extra_dims or {}).get('horizontal') or [])]
        if declared:
            return declared

        for family in HORIZONTAL_FAMILIES:
            if set(family) <= set(horizontal_dims):
                return [dim for dim in horizontal_dims if dim in family]
        return None

    def _identify_other_dims(self, dims):
        """
        Calculate and return the dimensions that are not part of horizontal_dims, mask_dim, or time_dims.

        Returns:
            set: A set of unused dimensions that are not part of horizontal, mask, or time dimensions.
        """
        # Safely convert identified dimensions to sets, handling None values
        horizontal_dims = set(self.horizontal_dims) if self.horizontal_dims else set()
        mask_dim = {self.mask_dim} if self.mask_dim else set()  # Use {} for single value
        time_dims = set(self.time_dims) if self.time_dims else set()

        # Return the unused dimensions by subtracting used dimensions from all dimensions
        return list(set(dims) - (horizontal_dims | mask_dim | time_dims))
