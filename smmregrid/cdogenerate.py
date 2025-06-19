"""CDO-based generation of weights"""

import os
import sys
import copy
import tempfile
import subprocess
import warnings
from multiprocessing import Process, Manager
import numpy
import xarray
from .weights import compute_weights_matrix3d, compute_weights_matrix, mask_weights, check_mask
from .log import setup_logger
from .cdogrid import CdoGrid
from .util import deprecated_argument


class CdoGenerate():
    """CDO-based class to generate weights for smmregrid."""

    def __init__(self, source_grid, target_grid=None, extra=None, cdo_extra=None,
            cdo_options=None, cdo_download_path=None, cdo_icon_grids=None,
            cdo="cdo", loglevel='warning'):
        """
        Initialize GenerateWeights class for regridding using Climate Data Operators (CDO), 
        accommodating both 2D and 3D grid cases.

        Args:
            source_grid (str or xarray.Dataset): The source grid from which to generate weights.
                                                 This can be a file path or an xarray dataset.
            target_grid (str or xarray.Dataset): The target grid to which the source grid 
                                                 will be regridded. This can also be a file 
                                                 path or an xarray dataset.
            loglevel (str, optional): The logging level for messages. Default is 'warning'. 
                                      Options include 'debug', 'info', 'warning', 
                                      'error', and 'critical'.
            extra (list, optional): Deprecated. Previously used for additional CDO options. 
                                   Use `cdo_extra` instead.
            cdo_extra (list, optional): Additional CDO command-line options. 
                                               Defaults to None.
            cdo_options (list, optional): Options for CDO commands. Defaults to None.
            cdo (str, optional): The command to invoke CDO. Default is "cdo".
            cdo_icon_grids (str, optional): Path to the ICON grid 
                                            if applicable. Defaults to None.
            cdo_download_path (str, optional): Path to the grid download path 
                                                if applicable. Defaults to None.
        """

        self.loggy = setup_logger(level=loglevel, name='smmregrid.CdoGenerate')

        # deprecation warning
        cdo_extra = deprecated_argument(extra, cdo_extra, 'extra', 'cdo_extra')

        # cdo options and extra, ensure they are lists
        self.cdo_extra = cdo_extra if isinstance(cdo_extra, list) else ([cdo_extra] if cdo_extra else [])
        self.cdo_options = cdo_options if isinstance(cdo_options, list) else ([cdo_options] if cdo_options else [])

        # assign the two grids
        self.source_grid = source_grid
        self.target_grid = target_grid

        # define grid filenames
        self.source_grid_filename = None
        self.target_grid_filename = None

        # assign cdo bin file and get envrinment file
        self.cdo = cdo
        self.env = os.environ.copy()
        if cdo_download_path:
            self.env["CDO_DOWNLOAD_PATH"] = cdo_download_path
        if cdo_icon_grids:
            self.env["CDO_ICON_GRIDS"] = cdo_icon_grids

    @staticmethod
    def _safe_check(method, remap_norm):
        """Safety checks for weights generation """

        supported_methods = ["bic", "bil", "con", "con2", "dis", "laf", "nn", "ycon"]
        if method not in supported_methods:
            raise ValueError('The remap method provided is not supported!')
        if remap_norm not in ["fracarea", "destarea"]:
            raise ValueError('The remap normalization provided is not supported!')

    
    def _prepare_grid(self, grid, target=False):
        """Helper function to prepare grid (file or dataset)."""

        # if grid is a Dataset or DataArray, save it to a temporary file
        if isinstance(grid, (xarray.Dataset, xarray.DataArray)):
            grid_file = tempfile.NamedTemporaryFile(delete=False)
            grid.to_netcdf(grid_file.name)
            self.loggy.debug("Xarray source, temporary grid file created for areas/weights generation: %s", grid_file.name)
            return grid_file.name

        # prepare grid: if source_grid is a CDO grid, use it as is, otherwise prepare the file
        if CdoGrid(grid).grid_kind and not target:
            self.loggy.info('CDO grid as %s to be used for areas/weights generation', grid)
            return f"-const,1,{grid}"
      
        # if grid is a string, assume it's a file path
        if isinstance(grid, str):
            self.loggy.debug("Grid file path to be used for areas/weights generation: %s", grid)
            return grid
        
        raise TypeError('Grid must be a CDO grid string, a file path, or an xarray Dataset/DataArray.')

    def weights(self, method="con", extrapolate=True,
            remap_norm="fracarea",
            vert_coord=None, vertical_dim=None, nproc=1):
        """
        Generate weights for regridding using Climate Data Operators (CDO), 
        accommodating both 2D and 3D grid cases.

        Args:
            method (str, optional): The remapping method to use. 
                                    Default is "con" for conservative remapping.
                                    Other options may include 'bil', 'nearest', etc.
            extrapolate (bool, optional): Whether to allow extrapolation beyond the grid boundaries. 
                                          Defaults to True.
            remap_norm (str, optional): The normalization method to apply when remapping.
                                        Default is "fracarea" which normalizes by fractional area.
            nproc (int, optional): Number of processes to use for parallel processing. Default is 1.
            vertical_dim (str, optional): Name of the vertical dimension in the source grid, if applicable.
                                        Defaults to None. Use if the grid is 3D.
            vert_coord (str, optional): Deprecated. Previously used to specify the vertical coordinate.
                                        Use `vertical_dim` instead.

        Returns:
            xarray.Dataset: A dataset containing the generated weights 
                            and a mask indicating which grid cells
                            were successfully masked. 
                            The mask is stored in a variable named "dst_grid_masked".

        Raises:
            KeyError: If the specified vertical dimension cannot be found in the source grid.
            Warning: If deprecated arguments `extra` or `vert_coord` are used.

        Notes:
            This function handles both 2D and 3D grid cases:

            For 2D grids (when `vertical_dim` is None), it calls the `_weights_2d` method
            to generate weights. The weights are then masked based on a precomputed weights matrix.

            For 3D grids (when `vertical_dim` is specified), it uses multiprocessing to generate weights
            for each vertical level. It requires the vertical dimension to be present in the source grid,
            and it will generate a mask indicating valid and invalid weights for each vertical level.

            The function logs the progress of weight generation, including the length of vertical dimensions
            and each level being processed.

            Deprecation Warning: The `extra` and `vert_coord` parameters are deprecated and will be removed in future versions.
            Users should migrate to using `cdo_extra` and `vertical_dim`, respectively.

        """
        if self.target_grid is None:
            raise TypeError('Target grid is not specified, cannot provide any regridding')
        if self.source_grid is None:
            raise TypeError('Source grid is not specified, cannot provide any regridding')

        # verify that method and normalization are suitable
        self._safe_check(method, remap_norm)

        # vertical dimension
        vertical_dim = deprecated_argument(vert_coord, vertical_dim, 'vert_coord', 'vertical_dim')

        #prepare grid
        self.source_grid_filename = self._prepare_grid(self.source_grid)
        self.target_grid_filename = self._prepare_grid(self.target_grid, target=True)

        # Generate weights for 2D or 3D grid based on vertical_dim presence
        if not vertical_dim:
            return self._weights_2d(method, extrapolate, remap_norm)
        return self._weights_3d(method, extrapolate, remap_norm,
                            nproc, vertical_dim)

    def _weights_2d(self, method, extrapolate, remap_norm):
        """Generate 2D weights using CDO."""

        weights = self._cdo_generate_weights(method, extrapolate,
                                             remap_norm)
        weights_matrix = compute_weights_matrix(weights)
        weights = mask_weights(weights, weights_matrix)
        masked = int(check_mask(weights))
        masked_xa = xarray.DataArray(masked, name="dst_grid_masked")
        return xarray.merge([weights, masked_xa])

    def _weights_3d(self, method, extrapolate, remap_norm,
                nproc, vertical_dim):
        """Generate 3D weights using multiprocessing."""

        if isinstance(self.source_grid, str):
            sgrid = xarray.open_dataset(self.source_grid)
        else:
            sgrid = self.source_grid

        if vertical_dim not in sgrid.dims:
            raise KeyError(f'Cannot find vertical dim {vertical_dim} in {list(sgrid.dims)}')

        #nvert = sgrid[vertical_dim].values.size
        nvert = sgrid.sizes[vertical_dim]
        self.loggy.info('Vertical dimension has length: %s', nvert)

        mgr = Manager()
        wlist = mgr.list(range(nvert))

        num_blocks, remainder = divmod(nvert, nproc)
        num_blocks += 0 if remainder == 0 else 1

        blocks = numpy.array_split(numpy.arange(nvert), num_blocks)
        for block in blocks:
            processes = []
            for lev in block:
                self.loggy.info("Generating level: %s", str(lev))
                cdo_extra_vertical = [f"-sellevidx,{lev + 1}"]
                ppp = Process(target=self._weights_worker,
                              args=(wlist, lev),
                              kwargs={
                                  "method": method,
                                  "extrapolate": extrapolate,
                                  "remap_norm": remap_norm,
                                  "cdo_extra_vertical": cdo_extra_vertical
                              })
                ppp.start()
                processes.append(ppp)

            for proc in processes:
                proc.join()

        weights = self.weightslist_to_3d(wlist, vertical_dim)
        weights_matrix = compute_weights_matrix3d(weights, vertical_dim)
        weights = mask_weights(weights, weights_matrix, vertical_dim)
        masked = check_mask(weights, vertical_dim)
        masked = [int(x) for x in masked]
        masked_xa = xarray.DataArray(masked,
                                     coords={vertical_dim: range(0, len(masked))},
                                     name="dst_grid_masked")

        return xarray.merge([weights, masked_xa])

    def _weights_worker(self, wlist, nnn, *args, **kwargs):
        """Run a worker process."""
        wlist[nnn] = self._cdo_generate_weights(*args, **kwargs).compute()

    def _cdo_generate_weights(self, method="con", extrapolate=True,
                              remap_norm="fracarea",
                              cdo_extra_vertical=None):
        """
        Generate weights for regridding using CDO

        Available weight generation methods are:

        * bic: SCRIP Bicubic
        * bil: SCRIP Bilinear
        * con: SCRIP First-order conservative
        * con2: SCRIP Second-order conservative
        * dis: SCRIP Distance-weighted average
        * laf: YAC Largest area fraction
        * ycon: YAC First-order conservative
        * nn: Nearest neighbour

        Args:
            method (str): Regridding method - default "con"
            extrapolate (bool): Extrapolate output field - default True
            remap_norm (str): Normalisation method for conservative methods
            cdo_extra_vertical (list): Command to select vertical levels for 3d case

        Returns:
            :obj:`xarray.Dataset` with regridding weights
        """

        sgrid = self.source_grid_filename
        tgrid = self.target_grid_filename

        cdo_extra_vertical = cdo_extra_vertical if isinstance(cdo_extra_vertical, list) else ([cdo_extra_vertical] if cdo_extra_vertical else [])

        # Log method and remapping information
        self.loggy.info("CDO remapping method: %s", method)
        self.loggy.info("Extrapolation enabled: %s", extrapolate)
        self.loggy.debug("Normalization method: %s", remap_norm)

        weight_file = tempfile.NamedTemporaryFile()
        self.loggy.debug("Weight file name is: %s", weight_file.name)

        self.loggy.debug("Source grid file name is: %s", sgrid)
        self.loggy.debug("Target grid file name is: %s", tgrid)

        env = copy.deepcopy(self.env)
        env["REMAP_EXTRAPOLATE"] = "on" if extrapolate else "off"
        env["CDO_REMAP_NORM"] = remap_norm

        try:
            self.loggy.info("Additional CDO commands: %s", self.cdo_extra)
            self.loggy.info("Additional CDO options: %s", self.cdo_options)

            command = [
                self.cdo,
                *self.cdo_options,
                f"gen{method},{tgrid}",
                *self.cdo_extra+cdo_extra_vertical,
                sgrid,
                weight_file.name
            ]
            self.loggy.debug("Final CDO command: %s", command)
            subprocess.check_output(command, stderr=subprocess.STDOUT, env=env)

            weights = xarray.open_dataset(weight_file.name, engine="netcdf4")
            return weights
    
        except subprocess.CalledProcessError as err:
            print(err.output.decode(), file=sys.stderr)
            raise
        finally:
            weight_file.close()

    # def _remove_tmpfile(self, tmpfile):
    #     """Helper function to clean the tempfile grid (file or dataset)."""
    #     if not isinstance(tmpfile, str):
    #         os.remove(tmpfile.name)

    def weightslist_to_3d(self, ds_list, vertical_dim='lev'):
        """Combine a list of 2D CDO weights into a 3D one."""

        links_dim = "numLinks" if "numLinks" in ds_list[0].dims else "num_links"
        dim_values = range(len(ds_list))
        nl = [ds.src_address.size for ds in ds_list]
        nl0 = max(nl)
        nlda = xarray.DataArray(nl, coords={vertical_dim: range(0, len(nl))}, name="link_length")

        new_array = []
        varlist = ["src_address", "dst_address", "remap_matrix", "src_grid_imask", "dst_grid_imask"]
        ds0 = ds_list[0].drop_vars(varlist)

        for x, d in zip(ds_list, dim_values):
            nl1 = x.src_address.size
            xplist = [x[vname].pad(**{links_dim: (0, nl0 - nl1), "mode": 'constant', "constant_values": 0})
                      for vname in varlist]
            xmerged = xarray.merge(xplist)
            new_array.append(xmerged.assign_coords({vertical_dim: d}))

        return xarray.merge([nlda, ds0, xarray.concat(new_array, vertical_dim)],
                            combine_attrs='no_conflicts')
    
    def areas(self, target=False):
        """Generate source areas or target areas"""

        if not target:
            self.loggy.info('Generating areas for source grid!')
            #if not self.source_grid:
            #    raise TypeError('Source grid is not specified, cannot provide any area')
            return self._areas(self.source_grid, cdo_extra=self.cdo_extra,
                                cdo_options=self.cdo_options)

        if self.target_grid:
            
            self.loggy.info('Generating areas for target grid!')
            return self._areas(self.target_grid)

        raise TypeError('Target grid is not specified, cannot provide any area')

            
    def _areas(self, filename, cdo_extra=None, cdo_options=None):
        """Generate areas in a similar way of what done for weights"""

        # safety listing
        cdo_extra = cdo_extra if isinstance(cdo_extra, list) else ([cdo_extra] if cdo_extra else [])
        cdo_options = cdo_options if isinstance(cdo_options, list) else ([cdo_options] if cdo_options else [])

        # Make some temporary files that we'll feed to CDO
        areas_file = tempfile.NamedTemporaryFile()

        # prepare grid
        sgrid = self._prepare_grid(filename)

        try:
            self.loggy.info("Additional CDO commands: %s", cdo_extra)
            self.loggy.info("Additional CDO options: %s", cdo_options)

            command = [
                self.cdo,
                *cdo_options + ["-f", "nc4"],
                "gridarea",
                *cdo_extra,
                sgrid,
                areas_file.name
            ]
            self.loggy.debug("Final CDO command: %s", command)
            subprocess.check_output(command, stderr=subprocess.STDOUT, env=self.env)

            areas = xarray.open_dataset(areas_file.name, engine="netcdf4")
            areas.cell_area.attrs['units'] = 'm2'
            areas.cell_area.attrs['standard_name'] = 'area'
            areas.cell_area.attrs['long_name'] = 'area of grid cell'
            return areas
    
        except subprocess.CalledProcessError as err:
            print(err.output.decode(), file=sys.stderr)
            raise
        finally:
            areas_file.close()


def cdo_generate_weights(source_grid, target_grid, method="con", extrapolate=True,
                         remap_norm="fracarea", gridpath=None,
                         icongridpath=None, extra=None, cdo_extra=None, cdo_options=None,
                         vertical_dim=None, vert_coord=None,
                         cdo="cdo", nproc=1, loglevel='warning'):
    """
    Wrapper function for the new cdo class to provide the old access
    """
    warnings.warn("cdo_generate_weights() is now deprecated, please use CdoGenerate().weights()",
                    DeprecationWarning)
    generator = CdoGenerate(source_grid=source_grid, target_grid=target_grid, loglevel=loglevel,
                            extra=extra, cdo_extra=cdo_extra, cdo_options=cdo_options, cdo=cdo, 
                            cdo_icon_grids=icongridpath, cdo_download_path=gridpath)
    return generator.weights(method=method, extrapolate=extrapolate,
                         remap_norm=remap_norm,
                         vertical_dim=vertical_dim, vert_coord=vert_coord, nproc=nproc)

