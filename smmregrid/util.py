"""Utility module with various support function"""

import os
import warnings
import xarray
from .cdogrid import CdoGrid
from .default import CONSERVATIVE_METHODS
from .log import setup_logger


def check_gridfile(filename):
    """Check if a grid is a file, a cdo string or xarray object"""

    if filename is None:
        return None
    if isinstance(filename, (xarray.Dataset, xarray.DataArray)):
        return "xarray"
    if isinstance(filename, str):
        if CdoGrid(filename).grid_kind:
            return "grid"
        if os.path.exists(filename):
            return "file"
        raise FileNotFoundError(f"Cannot find {filename} on disk")

    raise TypeError(f'Unsuported format for {filename}')


def deprecated_argument(old, new, oldname='var1', newname='var2'):
    """Utility to provide warning in case of deprecated argument"""

    # Check for deprecated 'old' argument
    if old is not None:
        warnings.warn(
            f"{oldname} is deprecated and will be removed in future versions. "
            f"Please use {newname} instead.",
            DeprecationWarning
        )
        # If new is not provided, use the value from old
        if new is None:
            new = old
    return new


def find_coord(ds, possible_names):
    """Find the first matching coordinate in the dataset."""
    return next((name for name in possible_names if name in ds.coords), None)


def tolist(value):
    """Convert value to list format. Returns [] for None, [value] for single values, unchanged for lists."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def detect_nan_variation_dims(field, time_dim, check_dims):
    """
    Check if NaN values vary along specific dimension(s) in an xarray object
    after removing the time dimension.

    Args:
        field: Input xarray DataArray with dask backend.
        time_dim: Name of the time dimension to remove.
        check_dims:  list[str]. Dimension name(s) to check for NaN variation after time removal.
    
    Returns:
        list[str]: List of dimension names from check_dims that have NaN variations.
    """

    if time_dim[0] in field.dims:
        # If time_dim is present, reduce it to a single value (e.g., first time step)
        field = field.isel({time_dim[0]: 0}, drop=True)
    
    nan_mask = field.isnull()
    dims_with_variation = []
    for dim in check_dims:
        # Variation along this dim
        variation_mask = nan_mask.astype("int8").diff(dim=dim).astype(bool).any(dim=dim)                 
        
        count = variation_mask.sum().compute()
        # print(f"Dimension '{dim}' has {count} variations with NaN values.")
        if count > 0:
            dims_with_variation.append(dim)

    return dims_with_variation

def resolve_na_thres(na_thres, method, loglevel='warning'):
    """Resolve the na_thres value to use for the masking of the target cells.

    na_thres is the maximum fraction of missing weights of a target cell. If it is a string,
    pick the default of the method:

    * 'auto' is the smmregrid default: 0.5 for the conservative methods (the target cells
      covered by less than 50% of valid source points are masked), 1e-6 for bilinear and
      bicubic (NaN as soon as a neighbour is missing, as CDO) and 1.0 otherwise.
    * 'cdo' replicates CDO, where remap_area_min is 0 by default: as 'auto', but 1.0
      for the conservative methods, i.e. only the cells without any valid source point are masked.

    Otherwise, validate that the provided value is within [0.0, 1.0].

    Parameters
    ----------
    na_thres : float or str
        The requested na_thres value, or 'auto' or 'cdo' to derive it from the method.
    method : str
        The remapping method (e.g. 'bil', 'bic', 'con', 'nn').
    loglevel : str
        The logging level to use.

    Returns
    -------
    float
        The resolved na_thres value.
    """
    loggy = setup_logger(level=loglevel, name='smmregrid.resolve_na_thres')
    if isinstance(na_thres, str) and na_thres in ('auto', 'cdo'):
        if method in ('bil', 'bic'):
            value = 1e-6
        elif method in CONSERVATIVE_METHODS and na_thres == 'auto':
            value = 0.5
        else:
            value = 1.0
        loggy.info('na_thres=%s with method %s: na_thres will be set to %s', na_thres, method, value)
        na_thres = value

    na_thres = float(na_thres)
    if na_thres < 0.0 or na_thres > 1.0:
        raise ValueError('The na_thres provided must be between 0.0 and 1.0')

    loggy.info('na_thres is %s: target cells with more than this fraction of missing weights are masked', na_thres)

    return na_thres


def deprecated_remap_area_min(remap_area_min, na_thres, method):
    """Translate the deprecated remap_area_min argument into na_thres.

    With the conservative methods the fraction of a target cell covered by valid source points is one minus
    the fraction of missing weights, so remap_area_min is na_thres = 1 - remap_area_min. For the other methods
    remap_area_min has no effect, as in CDO.

    Parameters
    ----------
    remap_area_min : float or None
        The deprecated argument, None if it was not provided.
    na_thres : float or str
        The na_thres provided by the user.
    method : str
        The remapping method.

    Returns
    -------
    float or str
        The na_thres to resolve.
    """
    if remap_area_min is None:
        return na_thres

    warnings.warn(
        "remap_area_min is deprecated and will be removed in future versions. "
        "Please use na_thres = 1 - remap_area_min instead (na_thres='cdo' replicates CDO, "
        "which does not mask by area by default).",
        DeprecationWarning, stacklevel=3
    )
    remap_area_min = float(remap_area_min)
    if remap_area_min < 0.0 or remap_area_min > 1.0:
        raise ValueError('The remap_area_min provided must be between 0.0 and 1.0')
    if method not in CONSERVATIVE_METHODS:
        return na_thres

    mapped = round(1.0 - remap_area_min, 12)
    if not (isinstance(na_thres, str) and na_thres == 'auto') and \
            not (not isinstance(na_thres, str) and abs(float(na_thres) - mapped) < 1e-9):
        raise ValueError(f'remap_area_min={remap_area_min} and na_thres={na_thres} are in conflict: '
                         f'remap_area_min is na_thres = 1 - remap_area_min, use only na_thres')
    return mapped
