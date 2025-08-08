"""Utility module with various support function"""

import os
import warnings
import xarray
from .cdogrid import CdoGrid


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
    """Convert value to list format."""
    if isinstance(value, list):
        return value
    return [value]

def nan_variation_check(field, time_dim, check_dims):
    """
    Check if NaN values vary along specific dimension(s) in an xarray object
    after removing the time dimension.
    
    Args: 
        field: Input xarray DataArray with dask backend.
        time_dim: Name of the time dimension to remove.
        check_dims:  list[str]. Dimension name(s) to check for NaN variation after time removal.
    """
    
    #nan_mask = field.isnull().any(dim=time_dims)
    if time_dim[0] in field.dims:
        # If time_dim is present, reduce it to a single value (e.g., first time step)
        field = field.isel({time_dim[0]: 0})
    nan_mask = field.isnull()
    dims_with_variation = []
    for dim in check_dims:
        # Variation along this dim
        variation_mask = (
            nan_mask.astype("int8")
            .diff(dim=dim)                # difference along this dim
            .astype(bool)
            .any(dim=dim)                  # any variation in this dim
        )
        count = variation_mask.sum().compute()
        #print(f"Dimension '{dim}' has {count} variations with NaN values.")
        if count > 0:
            dims_with_variation.append(dim)

    return dims_with_variation
