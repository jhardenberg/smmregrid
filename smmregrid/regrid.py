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

# Modified slightly by Jost von Hardenberg <jost.hardenberg@polito.it>

"""Dask-aware regridding

To apply a regridding you will need a set of weights mapping from the source
grid to the target grid.

Regridding weights can be generated online using CDO (:func:`cdo_generate_weights`),
or offline by calling these programs externally (this is recommended especially
for large grids).

Once calculated :func:`regrid` will apply these weights using a Dask sparse
matrix multiply, maintaining chunking in dimensions other than lat and lon.

:class:`Regrid` can create basic weights and store them to apply the weights to
multiple datasets.
"""


import math
import sys
import xarray
import numpy
import dask.array
from .dimension import remove_degenerate_axes
from .cdo_weights import cdo_generate_weights
from .util import find_vert_coords
from .weights import compute_weights_matrix3d, compute_weights_matrix, mask_weights, check_mask
from .log import setup_logger
from .grid_inspector import GridInspector

# default spatial dimensions and vertical coordinates
default_space_dims = ['i', 'j', 'x', 'y', 'lon', 'lat', 'longitude', 'latitude',
                      'cell', 'cells', 'ncells', 'values', 'value', 'nod2', 'pix', 'elem']


class Regridder(object):
    """Set up the regridding operation

    Supply either both ``source_grid`` and ``dest_grid`` or just ``weights``.

    For large grids you may wish to pre-calculate the weights using
    CDO, if not supplied ``weights`` will be calculated from
    ``source_grid`` and ``dest_grid`` using CDO's genbil function.

    Weights may be pre-computed by an external program, or created using
    :func:`cdo_generate_weights`.

    Args:
        source_grid (:class:`xarray.DataArray`): Source grid / sample dataset
        target_grid (:class:`xarray.DataArray`): Target grid / sample dataset
        weights (:class:`xarray.Dataset`): Pre-computed interpolation weights
        vert_coord (str): Name of the vertical coordinate.
                          If provided, 3D weights are generated (default: None)
        level_idx (str): Prefix of helper vertical coordinate with original level indices.
                         If provided, 3D weights are selected from those levels (default: "idx_")
        method (str): Method to use for interpolation (default: 'con')
        space_dims (list): list of dimensions to interpolate (default: None)
        transpose (bool): transpose the output so that the vertical coordinate is
                          just before other spatial coords (dafault: True)
    """

    def __init__(self, source_grid=None, target_grid=None, weights=None,
                 method='con', space_dims=None, vert_coord=None, transpose=True,
                 cdo='cdo', level_idx="idx_", loglevel='WARNING'):

        if (source_grid is None or target_grid is None) and (weights is None):
            raise ValueError(
                "Either weights or source_grid/target_grid must be supplied"
            )
        
        self.loggy = setup_logger(level=loglevel, name='smmregrid.regrid')
        self.loglevel = loglevel

        self.transpose = transpose

        # Is there already a weights file?
        if weights is not None:
            raise KeyError('This version cannot initialize yet through weigths')
            #if not isinstance(weights, xarray.Dataset):
            #    self.weights = xarray.open_mfdataset(weights)
            #else:
            #    self.weights = weights
#
#            if not vert_coord:
#                self.vert_coord = find_vert_coords(self.weights)
#            else:
#                self.vert_coord = vert_coord
        else:
            
            if isinstance(source_grid, str):
                source_grid = xarray.open_dataset(source_grid)
            
            grid_info = GridInspector(source_grid, clean=True, loglevel=loglevel)
            self.grids = grid_info.get_grid_info()
            self.weights = {}
            self.vert_coord = {}

            for grid in self.grids:
                self.loggy.debug('Processing grids %s', grid)
                self.vert_coord[grid] = self.grids[grid]['vertical_dims']
                if self.vert_coord[grid]:
                    self.vert_coord[grid] = self.vert_coord[grid][0]
                self.loggy.debug('Vertical coordinate is %s', self.vert_coord[grid])
                #sample_variable = self.grids[grid]['variables'][0]
                #self.loggy.debug('Sample variable is %s', sample_variable)
                self.weights[grid] = cdo_generate_weights(source_grid, #[sample_variable],
                                                          target_grid, method=method,
                                                          vert_coord=self.vert_coord[grid],
                                                          cdo=cdo, loglevel=loglevel)

            # Check if there is a vertical coordinate for 3d oceanic data
            #if not vert_coord:
            #    self.vert_coord = find_vert_coords(source_grid)
            #else:
            #    self.vert_coord = vert_coord

            # Generate the weights with CDO
            #self.weights = cdo_generate_weights(source_grid, target_grid, method=method,
            #                                    vert_coord=self.vert_coord, cdo=cdo)

        for grid in self.grids:
            self.weights_matrix = {}
            self.masked = {}
            if self.vert_coord[grid]:
                self.weights_matrix[grid] = compute_weights_matrix3d(self.weights[grid], self.vert_coord[grid])
            else:
                self.weights_matrix[grid] = compute_weights_matrix(self.weights[grid])

            # this section is used to create a target mask initializing the CDO weights (both 2d and 3d)
            if "dst_grid_masked" in self.weights[grid].variables:  # has a destination mask been precomputed?
                self.masked[grid] = self.weights[grid].dst_grid_masked.data  # ok, let's use it
            else:
                # compute the destination mask now
                self.weights[grid] = mask_weights(self.weights[grid], self.weights_matrix[grid], self.vert_coord[grid])
                self.masked[grid] = check_mask(self.weights[grid], self.vert_coord[grid])

        self.space_dims = space_dims
        self.level_idx = level_idx

    def regrid(self, source_data):
        """Regrid ``source_data`` to match the target grid

        Args:
            source_data (:class:`xarray.DataArray` or xarray.Dataset): Source
            variable

        Returns:
            :class:`xarray.DataArray` or xarray.Dataset with a regridded
            version of the source variable
        """

        if isinstance(source_data, xarray.Dataset):

            # apply the regridder on each DataArray
            out = source_data.map(self.regrid_array, keep_attrs=False)

            # clean from degenerated variables
            degen_vars = [var for var in out.data_vars if out[var].dims == ()]
            return out.drop_vars(degen_vars)

        elif isinstance(source_data, xarray.DataArray):

            return self.regrid_array(source_data)

        else:
            sys.exit('The object provided is not a Xarray object!')

    def regrid_array(self, source_data):
        """Regridding selection through 2d and 3d arrays"""

        grid_inspect = GridInspector(source_data, clean=False, loglevel=self.loglevel)
        data_grids = grid_inspect.get_grid_info()
        for mygrid in data_grids.keys():
            if data_grids[mygrid]['vertical_dims']:
            #if (self.vert_coord and self.vert_coord in source_data.coords):
                # if this is a 3D we specified the vertical coord and it has it
                return self.regrid3d(source_data, mygrid)
            else:
                return self.regrid2d(source_data, mygrid)

    def regrid3d(self, source_data, mygrid):
        """Regrid ``source_data`` to match the target grid - 3D version

        Args:
            source_data (:class:`xarray.DataArray`): Source
            variable

        Returns:
            :class:`xarray.DataArray` with a regridded
            version of the source variable
        """

        self.loggy.debug('3D DataArray access: variable is %s', source_data.name)

        if mygrid in self.weights:
            vert_coord = self.vert_coord[mygrid]
            weights = self.weights[mygrid]
            weights_matrix = self.weights_matrix[mygrid]
            masked = self.masked[mygrid]

        else:
            self.loggy.info('%s will be excluded from the output', source_data.name)
            return xarray.DataArray(data=None)


        # CDO 2.2.0 fix
        if "numLinks" in self.weights[mygrid].dims:
            links_dim = "numLinks"
        else:
            links_dim = "num_links"

        # this is necessary to remove lev-bounds, temporary hack since they should
        # be treated in a smarter way
        if ("bnds" in source_data.name or "bounds" in source_data.name):
            return source_data

        # If a special additional coordinate is present pick correct levels from weights
        coord = next((coord for coord in source_data.coords if coord.startswith(self.level_idx)), None)
        if coord:  # if a coordinate starting with level_idx is found
            levlist = source_data.coords[coord].values.tolist()
            levlist = [levlist] if numpy.isscalar(levlist) else levlist
        else:
            levlist = list(range(0, source_data.coords[vert_coord].values.size))

        data3d_list = []
        for lev, levidx in enumerate(levlist):
            xa = source_data.isel(**{vert_coord: lev})
            wa = weights.isel(**{vert_coord: levidx})
            nl = wa.link_length.values
            wa = wa.isel(**{links_dim: slice(0, nl)})
            wm = weights_matrix[levidx]
            mm = masked[levidx]
            data3d_list.append(self.apply_weights(
                xa, wa, weights_matrix=wm,
                masked=mm, space_dims=self.space_dims)
            )
        data3d = xarray.concat(data3d_list, dim=vert_coord)

        if self.transpose:
            # Make sure that the vertical dimension is just before the spatial ones
            if self.space_dims is None:
                space_dims = default_space_dims
            else:
                space_dims = self.space_dims
            dims = list(data3d.dims)
            index = min([i for i, s in enumerate(dims) if s in space_dims])
            dimst = dims[1:index] + [dims[0]] + dims[index:]
            data3d = data3d.transpose(*dimst)

            return data3d
        else:
            raise ValueError('Cannot process this source data, are you sure it is an xarray?')

    def regrid2d(self, source_data, mygrid):
        """Regrid ``source_data`` to match the target grid, 2D version

        Args:
            source_data (:class:`xarray.DataArray`): Source
            variable

        Returns:
            :class:`xarray.DataArray` with a regridded
            version of the source variable
        """
        self.loggy.debug('2D DataArray access: variables is %s', source_data.name)
        # data_grids = GridInspector(source_data).get_grid_info()
        # data_variables = GridInspector(source_data).get_variable_grids()
        # for data_grid in data_grids:
        #     print(source_data[data_variables])
        #data_variables = grid_inspect.get_variable_grids()
        if mygrid in self.weights:
            weights = self.weights[mygrid]
            weights_matrix = self.weights_matrix[mygrid]
        else:
            self.loggy.info('%s will be excluded from the output', source_data.name)
            return xarray.DataArray(data=None)

        return self.apply_weights(
                source_data, weights, weights_matrix=weights_matrix,
                masked=self.masked, space_dims=self.space_dims)
    
    def apply_weights(self, source_data, weights, weights_matrix=None,
                  masked=True, space_dims=None):
        """
        Apply the CDO weights ``weights`` to ``source_data``, performing a regridding operation

        Args:
            source_data (xarray.DataArray): Source dataset
            weights (xarray.DataArray): CDO weights information
            masked (bool): if the DataArray is masked
            space_dims (list): dimensions on which the interpolation has to be done (e.g.['lon', 'lat'])

        Returns:
            xarray.DataArray: Regridded version of the source dataset
        """


        # Understand immediately if we need to return something or not
        # This is done if we have bounds variables
        if any(substring in source_data.name for substring in ["bnds", "bounds", "vertices"]):

            # we keep time bounds, and we ignore all the rest
            if 'time' in source_data.name:
                self.loggy.info('%s will not be interpolated in the output', source_data.name)
                return source_data
            else:
                self.loggy.info('%s will be excluded from the output', source_data.name)
                return xarray.DataArray(data=None)

        # Alias the weights dataset from CDO
        w = weights

        # The weights file contains a sparse matrix, that we need to multiply the
        # source data's horizontal grid with to get the regridded data.
        #
        # A bit of messing about with `.stack()` is needed in order to get the
        # dimensions to conform - the horizontal grid needs to be converted to a 1d
        # array, multiplied by the weights matrix, then unstacked back into a 2d
        # array

        # CDO style weights
        # src_address = w.src_address - 1
        # dst_address = w.dst_address - 1
        # remap_matrix = w.remap_matrix[:, 0]
        # w_shape = (w.sizes["src_grid_size"], w.sizes["dst_grid_size"])

        dst_grid_shape = w.dst_grid_dims.values
        dst_grid_center_lat = w.dst_grid_center_lat.data.reshape(
            # dst_grid_shape[::-1], order="C"
            dst_grid_shape[::-1]
        )
        dst_grid_center_lon = w.dst_grid_center_lon.data.reshape(
            # dst_grid_shape[::-1], order="C"
            dst_grid_shape[::-1]
        )

        dst_mask = w.dst_grid_imask
        # src_mask = w.src_grid_imask

        axis_scale = 180.0 / math.pi  # Weight lat/lon in radians

        # Dimension on which we can produce the interpolation
        if space_dims is None:
            space_dims = default_space_dims

        if not any(x in source_data.dims for x in space_dims):
            self.loggy.error("None of dimensions on which we can interpolate is found in the DataArray. Does your DataArray include any of these?")
            self.loggy.error(space_dims)
            self.loggy.error('smmregrid can identify only %s', source_data.dims)
            raise KeyError('Dimensions mismatch')

        # Find dimensions to keep
        nd = sum([(d not in space_dims) for d in source_data.dims])

        kept_shape = list(source_data.shape[0:nd])
        kept_dims = list(source_data.dims[0:nd])
        self.loggy.info('Dimension kept: %s', kept_dims)

        if weights_matrix is None:
            weights_matrix = compute_weights_matrix(weights)

        # Remove the spatial axes, apply the weights, add the spatial axes back
        source_array = source_data.data
        if isinstance(source_array, dask.array.Array):
            source_array = dask.array.reshape(source_array, kept_shape + [-1])
        else:
            source_array = numpy.reshape(source_array, kept_shape + [-1])

        # Handle input mask
        dask.array.ma.set_fill_value(source_array, 1e20)
        source_array = dask.array.ma.fix_invalid(source_array)
        source_array = dask.array.ma.filled(source_array)

        target_dask = dask.array.tensordot(source_array, weights_matrix, axes=1)

        # define and compute the new mask
        if masked:

            # target mask is loaded from above
            target_mask = dst_mask.data

            # broadcast the mask on all the remaining dimensions
            target_mask = numpy.broadcast_to(
                target_mask.reshape([1 for d in kept_shape] + [-1]), target_dask.shape
            )

            # apply the mask
            target_dask = dask.array.where(target_mask != 0.0, target_dask, numpy.nan)

        # after the tensordot, bring the NaN back in
        # Use greater than 1e19 to avoid numerical noise from interpolation.
        target_dask = xarray.where(target_dask > 1e19, numpy.nan, target_dask)

        # reshape the target DataArray
        target_dask = dask.array.reshape(
            target_dask, kept_shape + [dst_grid_shape[1], dst_grid_shape[0]]
        )

        # Create a new DataArray for the output
        target_da = xarray.DataArray(
            target_dask,
            dims=kept_dims + ["i", "j"],
            coords={
                k: v
                for k, v in source_data.coords.items()
                if set(v.dims).issubset(kept_dims)
            },
            name=source_data.name,
        )

        target_da.coords["lat"] = xarray.DataArray(dst_grid_center_lat, dims=["i", "j"])
        target_da.coords["lon"] = xarray.DataArray(dst_grid_center_lon, dims=["i", "j"])

        # Clean up coordinates
        target_da.coords["lat"] = remove_degenerate_axes(target_da.lat)
        target_da.coords["lon"] = remove_degenerate_axes(target_da.lon)

        # Convert to degrees if needed, rounding to avoid numerical errors
        target_da.coords["lat"] = numpy.round(target_da.lat * axis_scale, 10)
        target_da.coords["lon"] = numpy.round(target_da.lon * axis_scale, 10)

        # If a regular grid drop the 'i' and 'j' dimensions
        if target_da.coords["lat"].ndim == 1 and target_da.coords["lon"].ndim == 1:
            target_da = target_da.swap_dims({"i": "lat", "j": "lon"})

        # Add metadata to the coordinates
        target_da.coords["lat"].attrs["units"] = "degrees_north"
        target_da.coords["lat"].attrs["standard_name"] = "latitude"
        target_da.coords["lon"].attrs["units"] = "degrees_east"
        target_da.coords["lon"].attrs["standard_name"] = "longitude"
        target_da.coords["lon"].attrs["axis"] = "X"
        target_da.coords["lat"].attrs["axis"] = "Y"

        # Copy attributes from the original
        target_da.attrs = source_data.attrs

        # Clean CDI gridtype (which can lead to issues with CDO interpretation)
        if target_da.attrs.get('CDI_grid_type'):
            del target_da.attrs['CDI_grid_type']

        # Now rename to the original coordinate names
        # target_da = target_da.rename({"lat": source_lat.name, "lon": source_lon.name})

        return target_da



def regrid(source_data, target_grid=None, weights=None, vert_coord=None, transpose=True, cdo='cdo'):
    """
    A simple regrid. Inefficient if you are regridding more than one dataset
    to the target grid because it re-generates the weights each time you call
    the function.

    To save the weights use :class:`Regridder`.

    Args:
        source_data (:class:`xarray.DataArray`): Source variable
        target_grid (:class:`coecms.grid.Grid` or :class:`xarray.DataArray`): Target grid / sample variable
        vert_coord (str): Name of the vertical coordinate.
                          If provided, 3D weights are generated (default: None)
        weights (:class:`xarray.Dataset`): Pre-computed interpolation weights
        transpose (bool): If True, transpose the output so that the vertical
                          coordinate is just before the other spatial coordinates (default: True)

        cdo (path): path of cdo executable ["cdo"]
    Returns:
        :class:`xarray.DataArray` with a regridded version of the source variable
    """


    regridder = Regridder(source_data, target_grid=target_grid, weights=weights,
                          vert_coord=vert_coord, cdo=cdo, transpose=transpose)
    return regridder.regrid(source_data)


# def combine_2d_to_3d(array_list, dim_name, dim_values):
#     """
#     Function to combine a list of 2D xarrays into a 3D one adding a vertical coordinate lev
#     """
#     new_array = [x.assign_coords({dim_name: d}) for x, d in zip(array_list, dim_values)]
#     return xarray.concat(new_array, dim_name)
