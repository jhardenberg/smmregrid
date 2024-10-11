"""CDO-based generation of weights"""

import os
import sys
import tempfile
import subprocess
import warnings
from multiprocessing import Process, Manager
import numpy
import xarray
from .util import find_vert_coords
from .weights import compute_weights_matrix3d, compute_weights_matrix, mask_weights, check_mask
from .log import setup_logger



def worker(wlist, nnn, *args, **kwargs):
    """Run a worker process"""
    wlist[nnn] = cdo_generate_weights2d(*args, **kwargs).compute()


def cdo_generate_weights(source_grid, target_grid, method="con", extrapolate=True,
                         remap_norm="fracarea", remap_area_min=0.0, icongridpath=None,
                         gridpath=None, extra=None, cdo_extra=None, cdo_options=None, vert_coord=None,
                         cdo="cdo", nproc=1, loglevel='warning'):
    """Generate the weights using CDO, handling both 2D and 3D cases"""

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

    # Check if there is a vertical coordinate for 3d oceanic data
    if not vert_coord:
        vert_coord = find_vert_coords(source_grid)
        loggy.info('vert_coord is %s', str(vert_coord))

    if not vert_coord:  # Are we 2D? Use default method
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
        weights = mask_weights(weights, weights_matrix, vert_coord)
        masked = int(check_mask(weights, vert_coord))
        masked_xa = xarray.DataArray(masked, name="dst_grid_masked")

        return xarray.merge([weights, masked_xa])

    else:  # we are 3D
        cdo_extra = cdo_extra if isinstance(cdo_extra, list) else ([cdo_extra] if cdo_extra else [])

        if isinstance(source_grid, str):
            sgrid = xarray.open_dataset(source_grid)
        else:
            sgrid = source_grid

        nvert = sgrid[vert_coord].values.size
        # print(nvert)

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
                cdo_extra_vertical = [f"-sellevidx,{lev+1}"]
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

        weights = weightslist_to_3d(wlist, vert_coord)

        # Precompute destination weights mask
        weights_matrix = compute_weights_matrix3d(weights, vert_coord)
        weights = mask_weights(weights, weights_matrix, vert_coord)
        masked = check_mask(weights, vert_coord)
        masked = [int(x) for x in masked]  # convert to list of int
        masked_xa = xarray.DataArray(masked, 
                                     coords={vert_coord: range(0, len(masked))},
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
            env=env,
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


def weightslist_to_3d(ds_list, vert_coord='lev'):
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
    nlda = xarray.DataArray(nl, coords={vert_coord: range(0, len(nl))}, name="link_length")
    new_array = []
    varlist = ["src_address", "dst_address", "remap_matrix", "src_grid_imask", "dst_grid_imask"]
    ds0 = ds_list[0].drop_vars(varlist)
    for x, d in zip(ds_list, dim_values):
        nl1 = x.src_address.size
#        xplist = [x[vname].pad(num_links=(0, nl0-nl1), mode='constant', constant_values=0)
        xplist = [x[vname].pad(**{links_dim: (0, nl0 - nl1), "mode": 'constant', "constant_values": 0})
                  for vname in varlist]
        xmerged = xarray.merge(xplist)
        new_array.append(xmerged.assign_coords({vert_coord: d}))
    return xarray.merge([nlda, ds0, xarray.concat(new_array, vert_coord)])
