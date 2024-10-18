"""CDO-based generation of weights"""

import os
import sys
import tempfile
import subprocess
import warnings
from multiprocessing import Process, Manager
import numpy
import xarray
from .weights import compute_weights_matrix3d, compute_weights_matrix, mask_weights, check_mask
from .log import setup_logger


def worker(wlist, nnn, *args, **kwargs):
    """Run a worker process"""
    wlist[nnn] = cdo_generate_weights2d(*args, **kwargs).compute()


def cdo_generate_weights(source_grid, target_grid, method="con", extrapolate=True,
                         remap_norm="fracarea", remap_area_min=0.0, icongridpath=None,
                         gridpath=None, extra=None, cdo_extra=None, cdo_options=None, vertical_dim=None,
                         vert_coord=None,
                         cdo="cdo", nproc=1, loglevel='warning'):
    """
    Generate weights for regridding using Climate Data Operators (CDO), accommodating both 2D and 3D grid cases.

    Args:
        source_grid (str or xarray.Dataset): The source grid from which to generate weights.
                                              This can be a file path or an xarray dataset.
        target_grid (str or xarray.Dataset): The target grid to which the source grid will be regridded.
                                              This can also be a file path or an xarray dataset.
        method (str, optional): The remapping method to use. Default is "con" for conservative remapping.
                                Other options may include 'bilinear', 'nearest', etc.
        extrapolate (bool, optional): Whether to allow extrapolation beyond the grid boundaries. Defaults to True.
        remap_norm (str, optional): The normalization method to apply when remapping.
                                     Default is "fracarea" which normalizes by fractional area.
        remap_area_min (float, optional): Minimum area for remapping. Defaults to 0.0.
        icongridpath (str, optional): Path to the ICON grid if applicable. Defaults to None.
        gridpath (str, optional): Path to the grid information if applicable. Defaults to None.
        extra (any, optional): Deprecated. Previously used for additional CDO options. Use `cdo_extra` instead.
        cdo_extra (list or any, optional): Additional CDO command-line options. Defaults to None.
        cdo_options (dict, optional): Options for CDO commands. Defaults to None.
        vertical_dim (str, optional): Name of the vertical dimension in the source grid, if applicable.
                                       Defaults to None. Use if the grid is 3D.
        vert_coord (str, optional): Deprecated. Previously used to specify the vertical coordinate.
                                     Use `vertical_dim` instead.
        cdo (str, optional): The command to invoke CDO. Default is "cdo".
        nproc (int, optional): Number of processes to use for parallel processing. Default is 1.
        loglevel (str, optional): The logging level for messages. Default is 'warning'. Options include
                                   'debug', 'info', 'warning', 'error', and 'critical'.

    Returns:
        xarray.Dataset: A dataset containing the generated weights and a mask indicating which grid cells
                        were successfully masked. The mask is stored in a variable named "dst_grid_masked".

    Raises:
        KeyError: If the specified vertical dimension cannot be found in the source grid.
        Warning: If deprecated arguments `extra` or `vert_coord` are used.

    Notes:
        This function handles both 2D and 3D grid cases:

        - For 2D grids (when `vertical_dim` is None), it calls the `cdo_generate_weights2d` function
          to generate weights. The weights are then masked based on a precomputed weights matrix.

        - For 3D grids (when `vertical_dim` is specified), it uses multiprocessing to generate weights
          for each vertical level. It requires the vertical dimension to be present in the source grid,
          and it will generate a mask indicating valid and invalid weights for each vertical level.

        The function logs the progress of weight generation, including the length of vertical dimensions
        and each level being processed.

        Deprecation Warning: The `extra` and `vert_coord` parameters are deprecated and will be removed in future versions.
        Users should migrate to using `cdo_extra` and `vertical_dim`, respectively.
    """

    loggy = setup_logger(level=loglevel, name='smmregrid.cdo_generate_weights')

    # Check for deprecated 'extra' argument
    if extra is not None:
        warnings.warn(
            "'extra' is deprecated and will be removed in future versions. "
            "Please use 'cdo_extra' instead.",
            DeprecationWarning
        )
        # If cdo_extra is not provided, use the value from extra
        if cdo_extra is None:
            cdo_extra = extra

        # Check for deprecated 'extra' argument
    if vert_coord is not None:
        warnings.warn(
            "'vert_coord' is deprecated and will be removed in future versions. "
            "Please use 'vertical_dim' instead.",
            DeprecationWarning
        )
        # If cdo_extra is not provided, use the value from extra
        if vertical_dim is None:
            vertical_dim = vert_coord

    if not vertical_dim:  # Are we 2D? Use default method
        weights = cdo_generate_weights2d(
            source_grid,
            target_grid,
            method=method,
            extrapolate=extrapolate,
            remap_norm=remap_norm,
            remap_area_min=remap_area_min,
            icongridpath=icongridpath,
            gridpath=gridpath,
            cdo_extra=cdo_extra,
            cdo_options=cdo_options,
            cdo=cdo,
            nproc=nproc)

        # Precompute destination weights mask
        weights_matrix = compute_weights_matrix(weights)
        weights = mask_weights(weights, weights_matrix, vertical_dim)
        masked = int(check_mask(weights, vertical_dim))
        masked_xa = xarray.DataArray(masked, name="dst_grid_masked")

        return xarray.merge([weights, masked_xa])

    else:  # we are 3D
        cdo_extra = cdo_extra if isinstance(cdo_extra, list) else ([cdo_extra] if cdo_extra else [])

        if isinstance(source_grid, str):
            sgrid = xarray.open_dataset(source_grid)
        else:
            sgrid = source_grid

        if not vertical_dim in sgrid:
            raise KeyError(f'Cannot find vertical dim {vertical_dim} in {list(sgrid.dims)}')

        nvert = sgrid[vertical_dim].values.size
        loggy.info('Vertical dimensions has length: %s', nvert)

        # for lev in range(0, nvert):
        mgr = Manager()

        # dictionaries are shared, so they have to be passed as functions
        wlist = mgr.list(range(nvert))

        num_blocks, remainder = divmod(nvert, nproc)
        num_blocks = num_blocks + (0 if remainder == 0 else 1)

        blocks = numpy.array_split(numpy.arange(nvert), num_blocks)
        for block in blocks:
            processes = []
            for lev in block:
                loggy.info("Generating level: %s", str(lev))
                cdo_extra_vertical = [f"-sellevidx,{lev + 1}"]
                ppp = Process(target=worker,
                              args=(wlist, lev, source_grid, target_grid),
                              kwargs={
                                  "method": method,
                                  "extrapolate": extrapolate,
                                  "remap_norm": remap_norm,
                                  "remap_area_min": remap_area_min,
                                  "icongridpath": icongridpath,
                                  "gridpath": gridpath,
                                  "cdo_extra": cdo_extra + cdo_extra_vertical,
                                  "cdo_options": cdo_options,
                                  "cdo": cdo,
                                  "nproc": nproc
                              })
                ppp.start()
                processes.append(ppp)

            for proc in processes:
                proc.join()

        weights = weightslist_to_3d(wlist, vertical_dim)

        # Precompute destination weights mask
        weights_matrix = compute_weights_matrix3d(weights, vertical_dim)
        weights = mask_weights(weights, weights_matrix, vertical_dim)
        masked = check_mask(weights, vertical_dim)
        masked = [int(x) for x in masked]  # convert to list of int
        masked_xa = xarray.DataArray(masked,
                                     coords={vertical_dim: range(0, len(masked))},
                                     name="dst_grid_masked")

        return xarray.merge([weights, masked_xa])


def cdo_generate_weights2d(source_grid, target_grid, method="con", extrapolate=True,
                           remap_norm="fracarea", remap_area_min=0.0, icongridpath=None,
                           gridpath=None, cdo_extra=None, cdo_options=None, cdo="cdo",
                           nproc=1):
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

    Run ``cdo gen${method} --help`` for details of each method

    Args:
        source_grid (xarray.DataArray): Source grid
        target_grid (xarray.DataArray): Target grid
            description
        method (str): Regridding method - default conservative
        extrapolate (bool): Extrapolate output field
        remap_norm (str): Normalisation method for conservative methods
        remap_area_min (float): Minimum destination area fraction
        gridpath (str): where to store downloaded grids
        icongridpath (str): location of ICON grids (e.g. /pool/data/ICON)
        cdo_extra: command(s) to apply to source grid before weight generation (can be a list)
        cdo_options: command(s) to apply to cdo (can be a list)
        cdo: the command to launch cdo ["cdo"]
        nproc: number of processes to use for weight generation (NOT USED!)

        cdo: path to cdo binary
    Returns:
        :obj:`xarray.Dataset` with regridding weights
    """

    supported_methods = ["bic", "bil", "con", "con2", "dis", "laf", "nn", "ycon"]
    if method not in supported_methods:
        raise ValueError('The remap method provided is not supported!')
    if remap_norm not in ["fracarea", "destarea"]:
        raise ValueError('The remap normalization provided is not supported!')

    # Make some temporary files that we'll feed to CDO
    weight_file = tempfile.NamedTemporaryFile()

    if isinstance(source_grid, str):
        sgrid = source_grid
    else:
        source_grid_file = tempfile.NamedTemporaryFile()
        source_grid.to_netcdf(source_grid_file.name)
        sgrid = source_grid_file.name

    if isinstance(target_grid, str):
        tgrid = target_grid
    else:
        target_grid_file = tempfile.NamedTemporaryFile()
        target_grid.to_netcdf(target_grid_file.name)
        tgrid = target_grid_file.name

    # Setup environment
    env = os.environ
    if extrapolate:
        env["REMAP_EXTRAPOLATE"] = "on"
    else:
        env["REMAP_EXTRAPOLATE"] = "off"

    env["CDO_REMAP_NORM"] = remap_norm
    env["REMAP_AREA_MIN"] = f"{remap_area_min:f}"

    if gridpath:
        env["CDO_DOWNLOAD_PATH"] = gridpath
    if icongridpath:
        env["CDO_ICON_GRIDS"] = icongridpath

    try:

        # condense cdo options
        cdo_extra = cdo_extra if isinstance(cdo_extra, list) else ([cdo_extra] if cdo_extra else [])
        cdo_options = cdo_options if isinstance(cdo_options, list) else ([cdo_options] if cdo_options else [])

        command = [
            cdo,
            *cdo_options,
            f"gen{method},{tgrid}",  # Method and target grid
            *cdo_extra,
            sgrid,
            weight_file.name
        ]

        # call to subprocess in compact way
        subprocess.check_output(
            command,
            stderr=subprocess.STDOUT,
            env=env
        )

        # Grab the weights file it outputs as a xarray.Dataset
        weights = xarray.open_dataset(weight_file.name, engine="netcdf4")
        return weights

    except subprocess.CalledProcessError as err:

        print(err.output.decode(), file=sys.stderr)
        raise

    finally:
        # Clean up the temporary files
        if not isinstance(source_grid, str):
            source_grid_file.close()
        if not isinstance(target_grid, str):
            target_grid_file.close()
        weight_file.close()


def weightslist_to_3d(ds_list, vertical_dim='lev'):
    """
    Function to combine a list of 2D cdo weights into a 3D one adding a vertical coordinate lev
    """
    # CDO 2.2.0 fix
    if "numLinks" in ds_list[0].dims:
        links_dim = "numLinks"
    else:
        links_dim = "num_links"

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
    return xarray.merge([nlda, ds0, xarray.concat(new_array, vertical_dim)])
