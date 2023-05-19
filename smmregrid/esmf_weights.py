"""ESMF-based generation of weights"""

import tempfile
import subprocess
import xarray


def esmf_generate_weights(
    source_grid,
    target_grid,
    method="bilinear",
    extrap_method="nearestidavg",
    norm_type="dstarea",
    line_type=None,
    pole=None,
    ignore_unmapped=False,
):
    """Generate regridding weights with ESMF

    https://www.earthsystemcog.org/projects/esmf/regridding

    Args:
        source_grid (:obj:`xarray.Dataarray`): Source grid. If masked the mask
            will be used in the regridding
        target_grid (:obj:`xarray.Dataarray`): Target grid. If masked the mask
            will be used in the regridding
        method (str): ESMF Regridding method, see ``ESMF_RegridWeightGen --help``
        extrap_method (str): ESMF Extrapolation method, see ``ESMF_RegridWeightGen --help``

    Returns:
        :obj:`xarray.Dataset` with regridding information from
            ESMF_RegridWeightGen
    """
    # Make some temporary files that we'll feed to ESMF
    source_file = tempfile.NamedTemporaryFile(suffix=".nc")
    target_file = tempfile.NamedTemporaryFile(suffix=".nc")
    weight_file = tempfile.NamedTemporaryFile(suffix=".nc")

    rwg = "ESMF_RegridWeightGen"

    if "_FillValue" not in source_grid.encoding:
        source_grid.encoding["_FillValue"] = -1e20

    if "_FillValue" not in target_grid.encoding:
        target_grid.encoding["_FillValue"] = -1e20

    try:
        source_grid.to_netcdf(source_file.name)
        target_grid.to_netcdf(target_file.name)

        command = [
            rwg,
            "--source",
            source_file.name,
            "--destination",
            target_file.name,
            "--weight",
            weight_file.name,
            "--method",
            method,
            "--extrap_method",
            extrap_method,
            "--norm_type",
            norm_type,
            # '--user_areas',
            "--no-log",
            "--check",
        ]

        if isinstance(source_grid, xarray.DataArray):
            command.extend(["--src_missingvalue", source_grid.name])
        if isinstance(target_grid, xarray.DataArray):
            command.extend(["--dst_missingvalue", target_grid.name])
        if ignore_unmapped:
            command.extend(["--ignore_unmapped"])
        if line_type is not None:
            command.extend(["--line_type", line_type])
        if pole is not None:
            command.extend(["--pole", pole])

        out = subprocess.check_output(args=command, stderr=subprocess.PIPE)
        print(out.decode("utf-8"))

        weights = xarray.open_dataset(weight_file.name, engine="netcdf4")
        # Load so we can delete the temp file
        return weights.load()

    except subprocess.CalledProcessError as err:
        print(err)
        print(err.output.decode("utf-8"))
        raise

    finally:
        # Clean up the temporary files
        source_file.close()
        target_file.close()
        weight_file.close()
