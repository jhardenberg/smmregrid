#!/usr/bin/env python
# Copyright 2018 ARC Centre of Excellence for Climate Extremes
# author: Scott Wales <scott.wales@unimelb.edu.au>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Refactored  by Jost von Hardenberg <jost.hardenberg@polito.it>
# and Paolo Davini <paolo.davini@cnr.it>

"""Dask-aware regridding

To apply a regridding you will need a set of weights mapping from the source
grid to the target grid.

Regridding weights can be generated online using CDO (:class:`CdoGenerate`),
or offline by calling these programs externally (this is recommended especially
for large grids).

Once calculated :func:`regrid` will apply these weights using a Dask sparse
matrix multiply, maintaining chunking in dimensions other than lat and lon.

:class:`Regrid` can create basic weights and store them to apply the weights to
multiple datasets.
"""


import math
import os
import xarray
import numpy
import dask.array
from .dimension import remove_degenerate_axes
from .cdogenerate import CdoGenerate
from .weights import compute_weights_matrix3d, compute_weights_matrix, mask_weights, check_mask, compute_total_weights
from .log import setup_logger
from .gridinspector import GridInspector
from .util import detect_nan_variation_dims, tolist, resolve_na_thres
from .default import DEFAULT_AREA_MIN, DEFAULT_NA_THRES

# default minimum area for conservative remapping



class Regridder(object):
    """Main smmregrid regridding class"""

    def __init__(self, source_grid=None, target_grid=None, weights=None,
                 method='con', remap_area_min=DEFAULT_AREA_MIN, transpose=True, mask_dim=None,
                 horizontal_dims=None, cdo_extra=None, cdo_options=None,
                 check_nan=False, skipna=False, na_thres=DEFAULT_NA_THRES,
                 cdo='cdo', loglevel='WARNING'):
        """
        Initialize the Regridder for performing regridding operations.

        This class allows for regridding between two grids, either by providing
        source and target grids or by using pre-computed weights. For large grids,
        it is recommended to pre-calculate weights using CDO. If weights are not
        supplied, they will be calculated using CDO's `genbil` function based on
        the provided source and target grids.

        Pre-computed weights can be generated externally or by using
        :class:`CdoGenerate()`.

        Args:
            source_grid (xarray.DataArray or str): Source grid dataset or file path.
            target_grid (xarray.DataArray): Target grid dataset for regridding.
            weights (xarray.Dataset): Pre-computed interpolation weights.
            mask_dim (str): Name of the masked coordinate for
                                3D weights generation (default: None).
            horizontal_dims (list): List of spatial dimensions to
                                    interpolate (default: None).
            method (str): Interpolation method to use (default: 'con').
            remap_area_min (float): Minimum area for remapping in conservative remapping.
                                    Larger values avoid mask erosion.
            transpose (bool): If True, transpose the output such that
                              the masked coordinate is placed just before the
                              other spatial coordinates (default: True).
            check_nan (bool): If True, check for NaN variation in the source grid
                              to define the masked dimension.
            cdo (str): Path to the CDO executable (default: 'cdo').
            cdo_extra (str): Extra command to be passed to CDO
            cdo_options(str): Extra options to be passed to CDO
            loglevel (str): Logging level for the operation (default: 'WARNING').

        Raises:
            ValueError: If neither weights nor source and target grids are provided.
            FileNotFoundError: If the specified source grid file does not exist.
            KeyError: If no grid types are found in the data.

        """
        if (source_grid is None or target_grid is None) and (weights is None):
            raise ValueError(
                "Either weights or source_grid/target_grid must be supplied"
            )

        # set up logger
        self.loggy = setup_logger(level=loglevel, name='smmregrid.Regrid')
        self.loglevel = loglevel
        self.transpose = transpose
        mask_dim = tolist(mask_dim)
        horizontal_dims = tolist(horizontal_dims)
        if mask_dim:
            self.loggy.info('Forcing mask_dim from input: expecting a single-gridtype dataset')
        if horizontal_dims:
            self.loggy.info('Forcing horizontal_dim from input: expecting a single-gridtype dataset')
        self.extra_dims = {'mask': mask_dim, 'horizontal': horizontal_dims}

        # TODO: this might be overridden at regrid level
        self.loggy.debug("Minimum remap area: %s", remap_area_min)
        self.remap_area_min = float(remap_area_min)
        if self.remap_area_min < 0.0 or self.remap_area_min > 1.0:
            raise ValueError('The remap_area_min provided must be between 0.0 and 1.0')

        # set up skipna option as ESMF
        self.skipna = skipna
        self.na_thres = resolve_na_thres(self.skipna, na_thres, method, loglevel)

        # Is there already a weights file?
        if weights is not None:
            self.loggy.info('Init from weights selected!')
            self.init_mode = 'weights'
            self.grids = self._gridtype_from_weights(weights)
        else:
            self.loggy.info('Init from grids selected!')
            self.init_mode = 'grids'
            if isinstance(source_grid, str):
                if os.path.isfile(source_grid):
                    source_grid_array = xarray.open_dataset(source_grid)
                else:
                    raise FileNotFoundError(f'Cannot find grid file {source_grid}')
            else:
                source_grid_array = source_grid

            self.grids = self._gridtype_from_data(source_grid_array)

            len_grids = len(self.grids)
            if len_grids == 0:
                raise ValueError('Cannot find any gridtype in your data, aborting!')
            if len_grids == 1:
                self.loggy.info('One gridtype found! Standard procedure')
            else:
                self.loggy.warning('%s gridtypes found! We are in uncharted territory!', len_grids)

            for index, gridtype in enumerate(self.grids):
                if check_nan:
                    # check for NaN variation in the source grid
                    gridtype = self._check_nan_variation(source_grid_array, gridtype)
                    self.grids[index] = gridtype  # update reference

                self.loggy.info('Processing grid number %s', index)
                self.loggy.debug('Processing grids %s', gridtype.dims)
                self.loggy.debug('Horizontal dimensions are %s', gridtype.horizontal_dims)
                self.loggy.debug('Mask dimension is %s', gridtype.mask_dim)
                self.loggy.debug('Other dimensions are %s', gridtype.other_dims)
                self.loggy.debug('Variables are %s', gridtype.variables)

                # always prefer to pass filename (i.e. source_grid) when possible to CdoGenerate()
                # this will limit errors from xarray and speed up CDO itself
                # it wil work only for single-gridtype dataset
                if isinstance(source_grid, str) and len_grids == 1:
                    source_grid_array_to_cdo = source_grid
                else:
                    # when feeding from xarray, select the variable and its bounds
                    if isinstance(source_grid_array, xarray.Dataset):
                        stored_vars = [list(gridtype.variables.keys())[0]] + gridtype.bounds
                        self.loggy.debug('Storing variables %s', stored_vars)
                        source_grid_array_to_cdo = source_grid_array[stored_vars]
                    else:
                        source_grid_array_to_cdo = source_grid_array

                    if gridtype.time_dims:
                        self.loggy.debug('Selecting only first time step for dimension %s', gridtype.time_dims[0])
                        source_grid_array_to_cdo = source_grid_array_to_cdo.isel({gridtype.time_dims[0]: 0})

                generator = CdoGenerate(source_grid_array_to_cdo, target_grid,
                                        cdo=cdo, cdo_options=cdo_options,
                                        cdo_extra=cdo_extra, loglevel=loglevel,
                                        skipna=skipna, na_thres=self.na_thres)
                gridtype.weights = generator.weights(method=method,
                                                     mask_dim=gridtype.mask_dim)

        for gridtype in self.grids:
            if gridtype.mask_dim:
                gridtype.weights_matrix = compute_weights_matrix3d(gridtype.weights,
                                                                   gridtype.mask_dim)
            else:
                gridtype.weights_matrix = compute_weights_matrix(gridtype.weights)

            # this section is used to create a target mask initializing the CDO weights (both 2d and 3d)
            # has a destination mask been precomputed?
            if "dst_grid_masked" in gridtype.weights.variables:
                gridtype.masked = gridtype.weights.dst_grid_masked.data
            else:
                # compute the destination mask now
                gridtype.weights = mask_weights(gridtype.weights, gridtype.weights_matrix, gridtype.mask_dim, nan_threshold=self.na_thres)
                gridtype.masked = check_mask(gridtype.weights, gridtype.mask_dim)

            # precompute quantities which depend only on the weights, to avoid repeating them at each call
            gridtype.total_weights = compute_total_weights(gridtype.weights_matrix)
            gridtype.target_coords = self._target_coords(gridtype.weights, gridtype.mask_dim)

    def _gridtype_from_weights(self, weights):
        """
        Initialize the gridtype reading from weights
        """

        self.loggy.info('Precomputed weights support so far single-gridtype datasets')

        if not isinstance(weights, xarray.Dataset):
            weights = xarray.open_mfdataset(weights)

        grid_info = GridInspector(weights, cdo_weights=True, extra_dims=self.extra_dims,
                                  clean=False, loglevel=self.loglevel)
        gridtype = grid_info.get_gridtype()

        # the vertical dimension has to show up into the extra dimensions
        # to cover the case that it is not a standard dimension: possibly better implementation available
        self.extra_dims['mask'] = [gridtype[0].mask_dim]

        return gridtype

    def _gridtype_from_data(self, source_grid_array):
        """
        Initialize the gridtype reading from source_data
        """
        grid_info = GridInspector(source_grid_array, extra_dims=self.extra_dims,
                                  clean=True, loglevel=self.loglevel)
        return grid_info.get_gridtype()

    def regrid(self, source_data):
        """
        Regrid the provided source data to match the target grid.

        This method applies the regridding process to either a single data array or
        a dataset containing multiple data variables.

        Args:
            source_data (xarray.DataArray or xarray.Dataset): The source data to be regridded.

        Returns:
            xarray.DataArray or xarray.Dataset: The regridded data, matching the target grid.

        Raises:
            TypeError: If the provided source data is neither an xarray DataArray nor Dataset.
        """

        # apply the regridder on each DataArray
        if isinstance(source_data, xarray.Dataset):

            # extra call to GridInspector for the raise with multiple grids
            grid_inspect = GridInspector(source_data,
                                         extra_dims=self.extra_dims,
                                         loglevel=self.loglevel)
            datagrids = grid_inspect.get_gridtype()
            if len(datagrids) > 1 and self.init_mode == 'weights':
                raise ValueError(f'Cannot process data with {len(datagrids)} GridType initializing from weights')

            # map on multiple dataarray
            out = source_data.map(self.regrid_array, keep_attrs=True)

            # clean from degenerated variables
            degen_vars = [var for var in out.data_vars if out[var].dims == ()]
            return out.drop_vars(degen_vars)

        if isinstance(source_data, xarray.DataArray):
            return self.regrid_array(source_data)

        raise TypeError('The object provided is not a Xarray object!')

    def regrid_array(self, source_data):
        """
        Perform regridding on a single 2D or 3D array.

        This method decides whether to call the 2D or 3D regridding function based
        on the dimensions of the source data.

        Args:
            source_data (xarray.DataArray): The source data array to regrid.

        Returns:
            xarray.DataArray: The regridded array.

        Raises:
            ValueError: If the input data does not match expected dimensions.
        """
        scalars = self._identify_scalar_coords(source_data)
        if scalars:
            self.loggy.warning(
                "Found scalar coordinates %s. If have selected a along a masked dimensions,"
                "regridding might fail. "
                "Please consider subsetting with [] or with slice", scalars)

        self.loggy.debug('Getting GridType from source_data')
        grid_inspect = GridInspector(source_data,
                                     extra_dims=self.extra_dims,
                                     loglevel=self.loglevel)
        datagrids = grid_inspect.get_gridtype()

        for datagridtype in datagrids:
            if datagridtype.mask_dim:
                # if this is a 3D we specified the masked coord and it has it
                return self.regrid3d(source_data, datagridtype)
            # 2d case
            return self.regrid2d(source_data, datagridtype)
        
        # fall back for the case that we have no gridtype,
        # or we have only scalar coordinates, 
        # or we are trying to regrid bounds variables
        return xarray.DataArray(data=None)

    def _identify_scalar_coords(self, source_data):
        """
        This method checks for coordinates that have no dimensions (i.e., scalar) and returns a list of their names.

        Args:
            source_data (xarray.DataArray): The source data array to check for scalar coordinates.
        """

        return [name for name, coord in source_data.coords.items() if coord.dims == ()]

    def _get_gridtype(self, datagridtype):

        # special case for CDO weights without any dimensional information
        # we derived this from the regridded data and we use it as it is
        if self.init_mode == 'weights':
            self.loggy.info('Assuming gridtype from data to be the same from weights')
            self.grids[0].dims = datagridtype.dims
            self.grids[0].horizontal_dims = datagridtype.horizontal_dims
            self.grids[0].other_dims = datagridtype.other_dims

        # match the grid
        gridtype = next((grid for grid in self.grids if grid == datagridtype), None)

        return gridtype

    def regrid3d(self, source_data, datagridtype):
        """
        Regrid a 3D source data array to match the target grid.

        This method applies the necessary weights and handles masked coordinates.

        Args:
            source_data (xarray.DataArray): The 3D source data to be regridded.
            datagridtype: The grid type information for the source data.

        Returns:
            xarray.DataArray: The regridded 3D data.

        Raises:
            ValueError: If there are issues with transposing the output dimensions.
        """

        self.loggy.debug('3D DataArray access: variable is %s', source_data.name)

        # get the gridtype from class that matches the data
        gridtype = self._get_gridtype(datagridtype)

        # if the current grid is not available in the Regrid gridtype
        if gridtype is None:
            self.loggy.info('%s will be excluded from the output', source_data.name)
            return xarray.DataArray(data=None)

        # select the gridtype to be used
        mask_dim = gridtype.mask_dim
        weights = gridtype.weights
        weights_matrix = gridtype.weights_matrix
        masked = gridtype.masked
        horizontal_dims = gridtype.horizontal_dims

        # CDO 2.2.0 fix
        if "numLinks" in weights.dims:
            links_dim = "numLinks"
        else:
            links_dim = "num_links"

        # this is necessary to remove lev-bounds, temporary hack since they should
        # be treated in a smarter way
        if ("bnds" in source_data.name or "bounds" in source_data.name):
            return source_data

        data3d_list = []
        # create pandas object for fast level index retrieval
        mask_index = weights.coords[mask_dim].to_index()
        for idx, lev in enumerate(source_data.coords[mask_dim].values):
            # get the index of the level for weights selection (widx, which might be different from idx)
            # tolerance is necessary to avoid numerical issues, hard coded
            widx = mask_index.get_indexer([lev], method="nearest", tolerance=1e-3)[0]
            if widx == -1:
                raise ValueError(
                    f"{lev} not found in mask_dim {mask_index.name}. "
                    f"Available levels: {list(mask_index.values)}"
                )
            #widx = mask_index.get_loc(lev)
            self.loggy.debug('Processing vertical level %s - level_index %s', lev, widx)
            # use isel since it faster
            xa = source_data.isel(**{mask_dim: idx})
            wa = weights.isel(**{mask_dim: widx})
            self.loggy.debug('Weight number of links is %s', wa.link_length.values)
            nl = wa.link_length.values
            wa = wa.isel(**{links_dim: slice(0, nl)})
            wm = weights_matrix[widx]
            mm = masked[widx]
            data3d_list.append(self.apply_weights(
                xa, wa, weights_matrix=wm,
                masked=mm, horizontal_dims=horizontal_dims,
                skipna=self.skipna, na_thres=self.na_thres,
                total_weights=gridtype.total_weights[widx],
                target_coords=gridtype.target_coords)
            )
        data3d = xarray.concat(data3d_list, dim=mask_dim, coords='different', compat='equals')

        # get dimensional info on target grid. TODO: can be moved at the init?
        target_gridtypes = GridInspector(data3d, clean=True, loglevel=self.loglevel).get_gridtype()
        target_horizontal_dims = target_gridtypes[0].horizontal_dims

        if self.transpose:
            data3d = self._transpose_output(data3d, target_horizontal_dims)
        return data3d

    def _transpose_output(self, data3d, target_horizontal_dims):
        """
        Transpose the output data array to place the vertical dimension as first
        """
        dims = list(data3d.dims)
        index = min((i for i, s in enumerate(dims) if s in target_horizontal_dims))
        dimst = dims[1:index] + [dims[0]] + dims[index:]
        return data3d.transpose(*dimst)

    def regrid2d(self, source_data, datagridtype):
        """
        Regrid a 2D source data array to match the target grid.

        This method applies the necessary weights for 2D data regridding.

        Args:
            source_data (xarray.DataArray): The 2D source data to be regridded.
            datagridtype: The grid type information for the source data.

        Returns:
            xarray.DataArray: The regridded 2D data.
        """
        self.loggy.debug('2D DataArray access: variables is %s', source_data.name)

        # get the gridtype from class that matches the data
        gridtype = self._get_gridtype(datagridtype)

        if gridtype is None:
            self.loggy.info('%s will be excluded from the output', source_data.name)
            return xarray.DataArray(data=None)

        return self.apply_weights(
            source_data,
            gridtype.weights,
            weights_matrix=gridtype.weights_matrix,
            masked=gridtype.masked,
            horizontal_dims=gridtype.horizontal_dims,
            skipna=self.skipna,
            na_thres=self.na_thres,
            total_weights=gridtype.total_weights,
            target_coords=gridtype.target_coords)

    def _target_coords(self, weights, mask_dim=None):
        """
        Compute the target grid shape, dimensions and lat/lon coordinates from the weights.
        These depend only on the weights, so they can be computed once and reused at each call.

        Args:
            weights (xarray.Dataset): The CDO weights.
            mask_dim (str, optional): The vertical dimension, if any. The target grid is the same for all levels.

        Returns:
            dict: with tgt_shape, tgt_dims, and lat and lon as xarray.DataArray.
        """

        # target grid is the same for all levels, so select the first one if needed
        if mask_dim and mask_dim in weights.dst_grid_center_lat.dims:
            weights = weights.isel({mask_dim: 0})

        dst_grid_shape = weights.dst_grid_dims.values
        dst_grid_rank = weights.dst_grid_rank

        if len(dst_grid_rank) == 2:
            tgt_shape = [dst_grid_shape[1], dst_grid_shape[0]]
            tgt_dims = ["i", "j"]
        elif len(dst_grid_rank) == 1:
            tgt_shape = [dst_grid_shape[0]]
            tgt_dims = ['cell']
        else:
            raise ValueError('Unknown dimensional target grid')

        # Weight lat/lon in radians
        axis_scale = 180.0 / math.pi
        attrs = {
            "lat": {"units": "degrees_north", "standard_name": "latitude", "axis": "Y"},
            "lon": {"units": "degrees_east", "standard_name": "longitude", "axis": "X"},
        }

        coords = {}
        for coord in ["lat", "lon"]:
            values = weights[f"dst_grid_center_{coord}"].values.reshape(dst_grid_shape[::-1])
            # Clean up coordinates
            da = remove_degenerate_axes(xarray.DataArray(values, dims=tgt_dims))
            # Convert to degrees if needed, rounding to avoid numerical errors
            da = numpy.round(da * axis_scale, 10)
            da.attrs = attrs[coord]
            coords[coord] = da

        return {"tgt_shape": tgt_shape, "tgt_dims": tgt_dims, **coords}

    def apply_weights(self, source_data, weights, weights_matrix=None,
                      masked=True, horizontal_dims=None, skipna=False, na_thres=1.0,
                      total_weights=None, target_coords=None):
        """
        Apply CDO weights to the source data, performing the regridding operation.

        This method multiplies the source data with the weights matrix to produce
        the regridded output.

        Args:
            source_data (xarray.DataArray): The source data to be regridded.
            weights (xarray.DataArray): The CDO weights for interpolation.
            weights_matrix (xarray.DataArray, optional): Pre-computed weights matrix (default: None).
            masked (bool): Indicates if the DataArray is masked (default: True).
            horizontal_dims (list): List of dimensions for interpolation (e.g., ['lon', 'lat']).
            skipna (bool): Whether to renormalize the output based on valid source points (default: False).
            na_thres (float): Maximum fraction of missing weights allowed when skipna is True (default: 1.0).
            total_weights (numpy.ndarray, optional): Pre-computed sum of weights per target cell (default: None).
            target_coords (dict, optional): Pre-computed target grid shape, dims and lat/lon (default: None).

        Returns:
            xarray.DataArray: The regridded version of the source dataset.

        Raises:
            KeyError: If none of the specified horizontal dimensions are found in the DataArray.
        """

        # Understand immediately if we need to return something or not
        # This is done if we have bounds variables
        if any(substring in source_data.name for substring in ["bnds", "bounds", "vertices"]):

            # we keep time bounds, and we ignore all the rest
            if 'time' in source_data.name:
                self.loggy.info('%s will not be interpolated in the output', source_data.name)
                return source_data

            self.loggy.info('%s will be excluded from the output', source_data.name)
            return xarray.DataArray(data=None)

        # CDO style weights
        # src_address = w.src_address - 1
        # dst_address = w.dst_address - 1
        # remap_matrix = w.remap_matrix[:, 0]
        # w_shape = (w.sizes["src_grid_size"], w.sizes["dst_grid_size"])
        # src_grid_rank = w.src_grid_rank

        # info on grids
        src_cdo_grid = weights.attrs['source_grid']
        dst_cdo_grid = weights.attrs['dest_grid']
        self.loggy.info('Interpolating from CDO %s to CDO %s', src_cdo_grid, dst_cdo_grid)

        # src grid properties
        src_grid_shape = weights.src_grid_dims.values

        # destination grid properties
        dst_grid_shape = weights.dst_grid_dims.values
        dst_frac_area = weights.dst_grid_frac
        dst_grid_mask = weights.dst_grid_imask

        # target grid shape, dims and lat/lon: use the precomputed ones if available
        if target_coords is None:
            target_coords = self._target_coords(weights)
        tgt_shape = target_coords["tgt_shape"]
        tgt_dims = target_coords["tgt_dims"]

        if not any(x in source_data.dims for x in horizontal_dims):
            self.loggy.error(
                "None of dimensions on which we can interpolate is found in the DataArray. Does your DataArray include any of these?")
            self.loggy.error(horizontal_dims)
            self.loggy.error('smmregrid can identify only %s', source_data.dims)
            raise KeyError('Dimensions mismatch')

        self.loggy.info('Regridding from %s to %s', src_grid_shape, dst_grid_shape)

        # Find dimensions to keep
        kept_dims = [dim for dim in source_data.dims if dim not in horizontal_dims]
        kept_shape = [source_data.sizes[dim] for dim in kept_dims]
        self.loggy.debug('Dimension to be ignored: %s', kept_dims)
        #if weights_matrix is None:
        #    weights_matrix = compute_weights_matrix(weights)

        # Remove the spatial axes, apply the weights, add the spatial axes back
        source_array = source_data.data
        if isinstance(source_array, dask.array.Array):
            source_array = dask.array.reshape(source_array, kept_shape + [-1])
        else:
            source_array = numpy.reshape(source_array, kept_shape + [-1])
        self.loggy.debug('Source array after reshape is: %s', source_array.shape)

        target_dask = self.tensordot(
            source_array, weights_matrix, masked, skipna, kept_shape, dst_grid_mask, na_thres,
            total_weights=total_weights, normalization=weights.attrs.get('normalization')
        )

        # use the frac area of the destination to further mask the data
        if self.remap_area_min > 0.0:
            area_min = dask.array.broadcast_to(dst_frac_area, target_dask.shape)
            target_dask = dask.array.where(area_min < self.remap_area_min, numpy.nan, target_dask)

        # reshape the target DataArray
        target_dask = dask.array.reshape(
            target_dask, kept_shape + tgt_shape
        )

        # Create a new DataArray for the output
        target_da = xarray.DataArray(
            target_dask,
            dims=kept_dims + tgt_dims,
            coords={
                k: v
                for k, v in source_data.coords.items()
                if set(v.dims).issubset(kept_dims)
            },
            name=source_data.name,
        )

        # Add the destination grid coordinate: shallow copy to avoid sharing attrs with the cached ones
        target_da.coords["lat"] = target_coords["lat"].copy(deep=False)
        target_da.coords["lon"] = target_coords["lon"].copy(deep=False)

        # If a regular grid drop the 'i' and 'j' dimensions
        if tgt_dims == ['i', 'j'] and target_da.coords["lat"].ndim == 1 and target_da.coords["lon"].ndim == 1:
            target_da = target_da.swap_dims({"i": "lat", "j": "lon"})

        # Copy attributes from the original
        target_da.attrs = source_data.attrs

        # Clean CDI gridtype (which can lead to issues with CDO interpretation)
        target_da.attrs.pop('CDI_grid_type', None)

        return target_da
    
    def tensordot(self, source_array, weights_matrix, masked, skipna, kept_shape, dst_grid_mask, na_thres,
                  total_weights=None, normalization=None):
        """
        Apply the sparse weights matrix to the flattened source array, handling NaNs.

        Three cases are possible:

        - ``skipna=False`` and ``masked=False``: plain ``source @ weights``.
        - ``skipna=False`` and ``masked=True``: NaNs in the source are set to zero, and the
          target cells outside the precomputed destination mask are set to NaN.
        - ``skipna=True``: NaNs in the source are set to zero and, unless the weights use the
          ``destarea`` normalization, the result is renormalized by the sum of the valid weights
          (see Notes). Target cells with no valid source point, or whose fraction of missing
          weights exceeds ``na_thres``, are set to NaN.

        Args:
            source_array (numpy.ndarray or dask.array.Array): Source data, with the horizontal
                dimensions flattened into the last axis (shape ``kept_shape + [n_src]``).
            weights_matrix (sparse matrix or dask.array.Array): Weights of shape ``(n_src, n_dst)``.
            masked (bool): Whether the source data contain a mask (i.e. NaNs) to be handled.
            skipna (bool): Whether to handle NaNs dynamically, based on the valid source points.
            kept_shape (list): Shape of the non-horizontal dimensions of the source array.
            dst_grid_mask (xarray.DataArray): Destination mask, used only when ``masked`` and not ``skipna``.
            na_thres (float): Maximum fraction of missing weights tolerated before setting
                a target cell to NaN, used only when ``skipna``.
            total_weights (numpy.ndarray, optional): Pre-computed sum of the weights for each target cell
                (NaN where no weights). Computed from ``weights_matrix`` if not provided.
            normalization (str, optional): The ``normalization`` attribute of the CDO weights
                (``'fracarea'``, ``'destarea'`` or ``'none'`` for non-conservative methods).
                Only ``'destarea'`` changes the behaviour, used only when ``skipna``.

        Returns:
            dask.array.Array: Regridded data, with shape ``kept_shape + [n_dst]``.

        Notes:
            **How skipna handles NaNs, and why destarea is special** (see discussion in
            https://github.com/jhardenberg/smmregrid/pull/70).

            With skipna, weights are generated by CDO from a source grid where NaNs are filled
            with zeros, i.e. as if all source points were valid. NaNs are then handled here, at each
            call, so that masks varying in time (or along any other dimension) are supported.
            For a target cell ``j``, with weights ``w_ij`` and valid-source indicator ``v_i``:

            - ``numerator_j   = sum_i v_i * w_ij * x_i``
            - ``denominator_j = sum_i v_i * w_ij``   (sum of the valid weights)
            - ``total_j       = sum_i w_ij``         (sum of all the weights, i.e. ``total_weights``)

            The goal is to reproduce what CDO would return if the weights were computed with
            the actual mask of the data, where a NaN source cell is treated as not existing:

            - ``fracarea`` (default for conservative) and interpolations (bil, nn, dis, ...):
              the weights of a target cell sum to 1 over the source cells which exist, so the
              result is ``numerator / denominator``. If all points are valid ``denominator == 1``,
              so the result is ``numerator`` anyway.
            - ``destarea``: the weights are the overlap areas divided by the FULL destination cell area,
              and they are NOT renormalized: their sum is the fraction of the target cell covered by
              (valid) source cells, and can be < 1. CDO thus returns ``numerator`` as it is:
              missing source cells contribute zero, as it makes sense e.g. for fluxes.
              Dividing by ``denominator`` would silently turn destarea into fracarea.

            In the past ``numerator / denominator`` was used for all cases, and the division was skipped
            when the source had no NaNs. With destarea this was wrong, and also non-local: a single NaN
            anywhere in the source changed the values of all the partially covered target cells
            (e.g. coasts, or the boundaries of a regional grid), even very far from it.

            In all cases target cells with no valid source point (``denominator_j == 0``) are NaN,
            and not zero, and ``1 - denominator_j / total_j`` is the fraction of missing weights
            used for the ``na_thres`` check.

            The computation is always done, even if the source has no NaNs: checking it in advance
            (e.g. with ``valid_data.all()``) would force dask to compute the whole source array,
            breaking laziness.
        """

        # identify valid data points (non-NaN) in the source array and set them to zero for the tensordot operation
        if masked or skipna:
            valid_data = dask.array.isfinite(source_array)
            source_array = dask.array.where(valid_data, source_array, 0.0)

        # compute tensor dot for cleaned matrix
        self.loggy.debug('Tensordot!')
        numerator = dask.array.tensordot(source_array, weights_matrix, axes=1)

        # default case: no skipna
        if not skipna:
            if masked:
                # if mask is precomputed, use it to mask the output
                mask_shape = [1 for d in kept_shape] + [-1]
                denominator = dst_grid_mask.data.reshape(mask_shape).astype(bool)
                return dask.array.where(denominator, numerator, numpy.nan)
            # this is not masked, so numerator is the final output
            return numerator

        # skipna is True: sum of the weights of the valid source points (see Notes in the docstring)
        self.loggy.debug('Tensordot with skipna, normalization is %s', normalization)
        denominator = dask.array.tensordot(valid_data.astype(weights_matrix.dtype), weights_matrix, axes=1)

        # no valid source point contributes: set to NaN to avoid division by zero,
        # so that these target cells will be NaN, and not zero, in the output
        denominator = dask.array.where(denominator > 0, denominator, numpy.nan)

        if normalization == 'destarea':
            # destarea weights must not be renormalized: missing source points contribute zero.
            # Check "!= 'destarea'" rather than "== 'fracarea'", since interpolation methods
            # have normalization 'none' and need renormalization too
            target_dask = dask.array.where(dask.array.isfinite(denominator), numerator, numpy.nan)
        else:
            target_dask = numerator / denominator

        # na_thres logic, safety check in init.
        # total_weights is NaN where a target cell receives no weights at all
        if total_weights is None:
            total_weights = compute_total_weights(weights_matrix)
        missing_fraction = 1.0 - denominator / total_weights
        return dask.array.where(missing_fraction > na_thres, numpy.nan, target_dask)

    def _check_nan_variation(self, source_grid, gridtype):
        """
        Check for NaN variation in the source grid and update the gridtype if necessary.
        """
        # check for NaN variation in the source grid
        if gridtype.mask_dim:
            self.loggy.info('Gridtype already has a masked dimension %s', gridtype.mask_dim)
            return gridtype

        var = next(iter(gridtype.variables))
        self.loggy.info('Checking for NaN variation in the source grid for variable %s', var)
        nan_dims = detect_nan_variation_dims(
            source_grid[var], time_dim=gridtype.time_dims,
            check_dims=gridtype.other_dims
        )

        if not nan_dims:
            return gridtype

        self.loggy.warning('Found NaN variation in dimensions: %s', nan_dims)
        # update extra dimension for future calls
        self.extra_dims['mask'] = nan_dims
        return GridInspector(source_grid, extra_dims=self.extra_dims,
                             loglevel=self.loglevel, clean=True).get_gridtype()[0]


def regrid(source_data, target_grid=None, weights=None, transpose=True, cdo='cdo'):
    """
    A simple regrid. Inefficient if you are regridding more than one dataset
    to the target grid because it re-generates the weights each time you call
    the function.

    To save the weights use :class:`Regridder`.

    Args:
        source_data (:class:`xarray.DataArray`): Source variable
        target_grid (:class:`coecms.grid.Grid` or :class:`xarray.DataArray`): Target grid / sample variable
        weights (:class:`xarray.Dataset`): Pre-computed interpolation weights
        transpose (bool): If True, transpose the output so that the masked
                          coordinate is just before the other spatial coordinates (default: True)

        cdo (path): path of cdo executable ["cdo"]
    Returns:
        :class:`xarray.DataArray` with a regridded version of the source variable
    """

    regridder = Regridder(source_data, target_grid=target_grid, weights=weights, cdo=cdo, transpose=transpose)
    return regridder.regrid(source_data)
