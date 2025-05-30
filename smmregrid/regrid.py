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
from .weights import compute_weights_matrix3d, compute_weights_matrix, mask_weights, check_mask
from .log import setup_logger
from .gridinspector import GridInspector
from .util import deprecated_argument

DEFAULT_AREA_MIN = 0.5  # default minimum area for conservative remapping

class Regridder(object):
    """Main smmregrid regridding class"""

    def __init__(self, source_grid=None, target_grid=None, weights=None,
                 method='con', remap_area_min=DEFAULT_AREA_MIN, transpose=True, vert_coord=None, vertical_dim=None,
                 space_dims=None, horizontal_dims=None, cdo_extra=None, cdo_options=None,
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
            vertical_dim (str): Name of the vertical coordinate for 
                                3D weights generation (default: None).
            horizontal_dims (list): List of spatial dimensions to 
                                    interpolate (default: None).
            method (str): Interpolation method to use (default: 'con').
            remap_area_min (float): Minimum area for remapping in conservative remapping.
                                    Larger values avoid mask erosion.
            transpose (bool): If True, transpose the output such that 
                              the vertical coordinate is placed just before the 
                              other spatial coordinates (default: True).
            cdo (str): Path to the CDO executable (default: 'cdo').
            cdo_extra (str): Extra command to be passed to CDO
            cdo_options(str): Extra options to be passed to CDO
            loglevel (str): Logging level for the operation (default: 'WARNING').

        Raises:
            ValueError: If neither weights nor source and target grids are provided.
            FileNotFoundError: If the specified source grid file does not exist.
            KeyError: If no grid types are found in the data.

        Warnings:
            DeprecationWarning: If deprecated arguments 'vert_coord' or 'space_dims' are used.
        """

        if (source_grid is None or target_grid is None) and (weights is None):
            raise ValueError(
                "Either weights or source_grid/target_grid must be supplied"
            )

        vertical_dim = deprecated_argument(vert_coord, vertical_dim, 'vert_coord', 'vertical_dim')
        horizontal_dims = deprecated_argument(space_dims, horizontal_dims, 'space_dims', 'horizontal_dims')
       
        # set up logger
        self.loggy = setup_logger(level=loglevel, name='smmregrid.Regrid')
        self.loglevel = loglevel
        self.transpose = transpose
        vertical_dim = [vertical_dim] if isinstance(vertical_dim, str) else vertical_dim
        horizontal_dims = [horizontal_dims] if isinstance(horizontal_dims, str) else horizontal_dims
        if vertical_dim:
            self.loggy.info('Forcing vertical_dim from input: expecting a single-gridtype dataset')
        if horizontal_dims:
            self.loggy.info('Forcing horizontal_dim from input: expecting a single-gridtype dataset')
        self.extra_dims = {'vertical': vertical_dim, 'horizontal': horizontal_dims}

        # TODO: this might be overridden at regrid level
        self.loggy.debug("Minimum remap area: %s", remap_area_min)
        self.remap_area_min = float(remap_area_min)
        if self.remap_area_min < 0.0 or self.remap_area_min > 1.0:
            raise ValueError('The remap_area_min provided must be between 0.0 and 1.0')
        
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

            for index, gridtype  in enumerate(self.grids):
                self.loggy.info('Processing grid number %s', index)
                self.loggy.debug('Processing grids %s', gridtype.dims)
                self.loggy.debug('Horizontal dimensions are %s', gridtype.horizontal_dims)
                self.loggy.debug('Vertical dimension is %s', gridtype.vertical_dim)
                self.loggy.debug('Other dimensions are %s', gridtype.other_dims)

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
                                               cdo_extra=cdo_extra, loglevel=loglevel)
                gridtype.weights = generator.weights(method=method,
                                                     vertical_dim=gridtype.vertical_dim)

        for gridtype in self.grids:
            if gridtype.vertical_dim:
                gridtype.weights_matrix = compute_weights_matrix3d(gridtype.weights,
                                                                   gridtype.vertical_dim)
            else:
                gridtype.weights_matrix = compute_weights_matrix(gridtype.weights)

            # this section is used to create a target mask initializing the CDO weights (both 2d and 3d)
            # has a destination mask been precomputed?
            if "dst_grid_masked" in gridtype.weights.variables:
                gridtype.masked = gridtype.weights.dst_grid_masked.data
            else:
                # compute the destination mask now
                gridtype.weights = mask_weights(gridtype.weights, gridtype.weights_matrix, gridtype.vertical_dim)
                gridtype.masked = check_mask(gridtype.weights, gridtype.vertical_dim)

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
        self.extra_dims['vertical'] = [gridtype[0].vertical_dim]

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
            if len(datagrids)>1 and self.init_mode == 'weights':
                raise ValueError(f'Cannot process data with {len(datagrids)} GridType initializing from weights')

            # map on multiple dataarray
            out = source_data.map(self.regrid_array, keep_attrs=False)

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
        
        self.loggy.debug('Getting GridType from source_data')
        grid_inspect = GridInspector(source_data,
                                     extra_dims=self.extra_dims,
                                     loglevel=self.loglevel)
        datagrids = grid_inspect.get_gridtype()

        for datagridtype in datagrids:
            if datagridtype.vertical_dim:
                # if this is a 3D we specified the vertical coord and it has it
                return self.regrid3d(source_data, datagridtype)
            # 2d case
            return self.regrid2d(source_data, datagridtype)

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

        This method applies the necessary weights and handles vertical coordinates.

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
        vertical_dim = gridtype.vertical_dim
        weights = gridtype.weights
        weights_matrix = gridtype.weights_matrix
        masked = gridtype.masked
        level_index = gridtype.level_index
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

        # If a special additional coordinate is present pick correct levels from weights
        coord = next((coord for coord in source_data.coords if coord.startswith(level_index)), None)
        if coord:  # if a coordinate starting with level_index is found
            levlist = source_data.coords[coord].values.tolist()
            levlist = [levlist] if numpy.isscalar(levlist) else levlist
        else:
            levlist = list(range(0, source_data.coords[vertical_dim].values.size))

        data3d_list = []
        for lev, levidx in enumerate(levlist):
            self.loggy.debug('Processing vertical level %s - level_index %s', lev, levidx)
            xa = source_data.isel(**{vertical_dim: lev})
            wa = weights.isel(**{vertical_dim: levidx})
            nl = wa.link_length.values
            wa = wa.isel(**{links_dim: slice(0, nl)})
            wm = weights_matrix[levidx]
            mm = masked[levidx]
            data3d_list.append(self.apply_weights(
                xa, wa, weights_matrix=wm,
                masked=mm, horizontal_dims=horizontal_dims)
            )
        data3d = xarray.concat(data3d_list, dim=vertical_dim)

        # get dimensional info on target grid. TODO: can be moved at the init?
        target_gridtypes = GridInspector(data3d, clean=True, loglevel=self.loglevel).get_gridtype()
        target_horizontal_dims = target_gridtypes[0].horizontal_dims

        if self.transpose:
            dims = list(data3d.dims)
            index = min([i for i, s in enumerate(dims) if s in target_horizontal_dims])
            dimst = dims[1:index] + [dims[0]] + dims[index:]
            data3d = data3d.transpose(*dimst)

            return data3d

        raise ValueError(f'Cannot transpose output dimensions {data3d.dims} over {target_horizontal_dims}')

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
            horizontal_dims=gridtype.horizontal_dims)

    def apply_weights(self, source_data, weights, weights_matrix=None,
                      masked=True, horizontal_dims=None):
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

        # destination grid properties
        dst_grid_shape = weights.dst_grid_dims.values
        dst_frac_area = weights.dst_grid_frac
        dst_grid_mask = weights.dst_grid_imask
        dst_grid_rank = weights.dst_grid_rank

        # target lon/lat
        dst_grid_center_lat = weights.dst_grid_center_lat.data.reshape(dst_grid_shape[::-1])
        dst_grid_center_lon = weights.dst_grid_center_lon.data.reshape(dst_grid_shape[::-1])

        axis_scale = 180.0 / math.pi  # Weight lat/lon in radians

        if not any(x in source_data.dims for x in horizontal_dims):
            self.loggy.error(
                "None of dimensions on which we can interpolate is found in the DataArray. Does your DataArray include any of these?")
            self.loggy.error(horizontal_dims)
            self.loggy.error('smmregrid can identify only %s', source_data.dims)
            raise KeyError('Dimensions mismatch')

        self.loggy.info('Regridding from %s to %s', source_data.shape, dst_grid_shape)

        # Find dimensions to keep
        kept_dims = [dim for dim in source_data.dims if dim not in horizontal_dims]
        kept_shape = [source_data.sizes[dim] for dim in kept_dims]

        self.loggy.debug('Dimension to be ignored: %s', kept_dims)
        if weights_matrix is None:
            weights_matrix = compute_weights_matrix(weights)

        # Remove the spatial axes, apply the weights, add the spatial axes back
        source_array = source_data.data
        if isinstance(source_array, dask.array.Array):
            source_array = dask.array.reshape(source_array, kept_shape + [-1])
        else:
            source_array = numpy.reshape(source_array, kept_shape + [-1])
        self.loggy.debug('Source array after reshape is: %s', source_array.shape)

        # Handle input mask
        dask.array.ma.set_fill_value(source_array, 1e20)
        source_array = dask.array.ma.fix_invalid(source_array)
        source_array = dask.array.ma.filled(source_array)

        self.loggy.debug('Tensordot!')
        target_dask = dask.array.tensordot(source_array, weights_matrix, axes=1)

        # define and compute the new mask
        if masked:
            # broadcast the mask on all the remaining dimensions and apply it with where
            target_mask = dask.array.broadcast_to(
                dst_grid_mask.data.reshape([1 for d in kept_shape] + [-1]), target_dask.shape
            )
            self.loggy.debug('Reshaped mask with %s', target_mask.shape)
            target_dask = dask.array.where(target_mask != 0.0, target_dask, numpy.nan)

        # use the frac area of the destination to further mask the data
        if self.remap_area_min > 0.0:
            target_dask = dask.array.where(
                dask.array.broadcast_to(dst_frac_area, target_dask.shape) < self.remap_area_min,
                numpy.nan, target_dask)

        # after the tensordot, bring the NaN back in
        # Use greater than 1e19 to avoid numerical noise from interpolation.
        target_dask = dask.array.where(target_dask > 1e19, numpy.nan, target_dask)

        if len(dst_grid_rank) == 2:
            tgt_shape = [dst_grid_shape[1], dst_grid_shape[0]]
            tgt_dims = ["i", "j"]
        elif len(dst_grid_rank) == 1:
            tgt_shape = [dst_grid_shape[0]]
            tgt_dims = ['cell']
        else:
            raise ValueError('Unknown dimensional target grid')

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

        # Add the destination grid coordinate
        target_da.coords["lat"] = xarray.DataArray(dst_grid_center_lat, dims=tgt_dims)
        target_da.coords["lon"] = xarray.DataArray(dst_grid_center_lon, dims=tgt_dims)

        # Clean up coordinates
        target_da.coords["lat"] = remove_degenerate_axes(target_da.lat)
        target_da.coords["lon"] = remove_degenerate_axes(target_da.lon)

        # Convert to degrees if needed, rounding to avoid numerical errors
        target_da.coords["lat"] = numpy.round(target_da.lat * axis_scale, 10)
        target_da.coords["lon"] = numpy.round(target_da.lon * axis_scale, 10)

        # If a regular grid drop the 'i' and 'j' dimensions
        if tgt_dims == ['i', 'j'] and target_da.coords["lat"].ndim == 1 and target_da.coords["lon"].ndim == 1:
            target_da = target_da.swap_dims({"i": "lat", "j": "lon"})

        # Add metadata to the coordinates
        target_da.coords["lat"].attrs.update(
            {"units": "degrees_north", "standard_name": "latitude", "axis": "Y"}
        )
        target_da.coords["lon"].attrs.update(
            {"units": "degrees_east", "standard_name": "longitude", "axis": "X"}
        )

        # Copy attributes from the original
        target_da.attrs = source_data.attrs

        # Clean CDI gridtype (which can lead to issues with CDO interpretation)
        target_da.attrs.pop('CDI_grid_type', None)


        return target_da


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
        transpose (bool): If True, transpose the output so that the vertical
                          coordinate is just before the other spatial coordinates (default: True)

        cdo (path): path of cdo executable ["cdo"]
    Returns:
        :class:`xarray.DataArray` with a regridded version of the source variable
    """

    regridder = Regridder(source_data, target_grid=target_grid, weights=weights, cdo=cdo, transpose=transpose)
    return regridder.regrid(source_data)


# def combine_2d_to_3d(array_list, dim_name, dim_values):
#     """
#     Function to combine a list of 2D xarrays into a 3D one adding a vertical coordinate lev
#     """
#     new_array = [x.assign_coords({dim_name: d}) for x, d in zip(array_list, dim_values)]
#     return xarray.concat(new_array, dim_name)
