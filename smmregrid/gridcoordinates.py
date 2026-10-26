"""GridCoordinates class module: identify latitude and longitude from xarray coordinates"""

from .log import setup_logger

# Fallback coordinate names for latitude and longitude
LAT_COORDS = ["lat", "latitude", "nav_lat"]
LON_COORDS = ["lon", "longitude", "nav_lon"]

# CF units for latitude and longitude
LAT_UNITS = ["degrees_north", "degree_north", "degree_N", "degrees_N", "degreeN", "degreesN"]
LON_UNITS = ["degrees_east", "degree_east", "degree_E", "degrees_E", "degreeE", "degreesE"]

AXES = {
    'lat': {'standard_name': 'latitude', 'units': LAT_UNITS, 'axis_type': 'Lat', 'names': LAT_COORDS},
    'lon': {'standard_name': 'longitude', 'units': LON_UNITS, 'axis_type': 'Lon', 'names': LON_COORDS},
}

# Evidence used to identify a coordinate, from the strongest to the weakest
EVIDENCE = ['user', 'standard_name', 'units', 'axis_type', 'name']


class GridCoordinates:
    """
    Identify latitude and longitude, and the horizontal dimensions they span, from xarray coordinates.

    Only names, attributes and dimensions of the coordinates are read: coordinate values are never
    loaded, so the inspection is cheap and stays lazy for Dask-backed coordinates.
    """

    def __init__(self, coords, lat=None, lon=None, loglevel='warning'):
        """
        Parameters:
            coords (xr.Coordinates, xr.Dataset or xr.DataArray): The coordinates to inspect.
                For a Dataset or DataArray only its coordinates are used.
            lat (str, optional): Name of the latitude coordinate, preferred over any other evidence.
            lon (str, optional): Name of the longitude coordinate, preferred over any other evidence.
            loglevel: The loglevel that you want you use

        Attributes:
            lat (str): The name of the latitude coordinate, None if not found.
            lon (str): The name of the longitude coordinate, None if not found.
            horizontal_dims (list): The dimensions spanned by latitude and longitude, None if not found.
            source (dict): The evidence used to identify 'lat' and 'lon' (see EVIDENCE).
            pairs (list): All the compatible (lat, lon) pairs, from the most to the least reliable.
        """
        self.loggy = setup_logger(level=loglevel, name='smmregrid.GridCoordinates')
        self.coords = getattr(coords, 'coords', coords)

        lat_candidates = self._find_candidates('lat', lat)
        lon_candidates = self._find_candidates('lon', lon)
        self.loggy.debug('Latitude candidates: %s', lat_candidates)
        self.loggy.debug('Longitude candidates: %s', lon_candidates)

        self.pairs = [
            (latname, lonname)
            for latname, _ in lat_candidates
            for lonname, _ in lon_candidates
            if latname != lonname and self._is_pair(latname, lonname)
        ]
        if len(self.pairs) > 1:
            self.loggy.info('Multiple latitude/longitude pairs found %s, using %s', self.pairs, self.pairs[0])

        self.lat, self.lon = self.pairs[0] if self.pairs else (None, None)
        self.source = {
            'lat': dict(lat_candidates).get(self.lat),
            'lon': dict(lon_candidates).get(self.lon),
        }
        self.horizontal_dims = self._horizontal_dims()
        self.loggy.debug('Latitude %s and longitude %s identified from %s, horizontal dims are %s',
                         self.lat, self.lon, self.source, self.horizontal_dims)

    def __repr__(self):
        return (f"GridCoordinates(lat={self.lat}, lon={self.lon}, "
                f"horizontal_dims={self.horizontal_dims}, source={self.source})")

    def _find_candidates(self, axis, user=None):
        """
        Find the coordinates which can describe the axis ('lat' or 'lon'),
        sorted by the strength of the evidence and then by coordinate order.

        Returns:
            list: (name, evidence) tuples. Scalar coordinates are excluded.
        """
        rules = AXES[axis]
        candidates = []
        for name in self.coords:
            if self.coords[name].ndim == 0:
                continue
            evidence = self._evidence(name, rules, user)
            if evidence:
                candidates.append((name, evidence))
        return sorted(candidates, key=lambda candidate: EVIDENCE.index(candidate[1]))

    def _evidence(self, name, rules, user=None):
        """Return the strongest evidence that the coordinate describes the axis, None if there is none"""
        attrs = self.coords[name].attrs
        if user is not None and name == user:
            return 'user'
        if attrs.get('standard_name') == rules['standard_name']:
            return 'standard_name'
        if attrs.get('units') in rules['units']:
            return 'units'
        if attrs.get('_CoordinateAxisType') == rules['axis_type']:
            return 'axis_type'
        if name in rules['names']:
            return 'name'
        return None

    def _is_pair(self, lat, lon):
        """
        Check if latitude and longitude describe the same grid: both 1D on different
        dimensions (regular), or both on the same dimensions (curvilinear or unstructured).
        """
        lat_dims = self.coords[lat].dims
        lon_dims = self.coords[lon].dims
        if lat_dims == lon_dims:
            return True
        return len(lat_dims) == 1 and len(lon_dims) == 1

    def _horizontal_dims(self):
        """Return the ordered union of the latitude and longitude dimensions"""
        if self.lat is None:
            return None
        return list(dict.fromkeys(self.coords[self.lat].dims + self.coords[self.lon].dims))
