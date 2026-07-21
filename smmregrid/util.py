"""Utility module with various support function"""

import os
import warnings
import xarray
from .cdogrid import CdoGrid
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

def resolve_na_thres(skipna, na_thres, method, loglevel='warning'):
    """Resolve the na_thres value to use for NaN-aware regridding.

    If na_thres is 'auto', pick a sensible default based on whether
    skipna is enabled and which remapping method is used. Otherwise,
    validate that the provided value is within [0.0, 1.0].

    Parameters
    ----------
    skipna : bool
        Whether NaN-skipping is enabled.
    na_thres : float or str
        The requested na_thres value, or 'auto' to derive it.
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
    if na_thres == "auto":
        if skipna:
            loggy.info(
                'skipna is enabled with na_thres=auto. na_thres will be set '
                'to 1e-6 for bilinear and bicubic, and 1.0 for conservative '
                'and nearest neighbor'
            )
            na_thres = 1e-6 if method in ('bil', 'bic') else 1.0
        else:
            loggy.info(
                'skipna is disabled with na_thres=auto. na_thres will be set to 0.5'
            )
            na_thres = 0.5

    na_thres = float(na_thres)
    if na_thres < 0.0 or na_thres > 1.0:
        raise ValueError('The na_thres provided must be between 0.0 and 1.0')

    if skipna:
        loggy.warning(
            'skipna is enabled with na_thres=%s. This will affect the regridding behavior.',
            na_thres
        )
    else:
        loggy.info(
            'skipna is disabled with na_thres=%s. This will affect the regridding behavior.',
            na_thres
        )


    return na_thres