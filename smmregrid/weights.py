"""Weight calculation utilities"""

import dask
import numpy
import sparse


def compute_weights_matrix3d(weights, mask_dim='lev'):
    """
    Convert the weights from CDO to a list of numpy arrays
    """

    # CDO 2.2.0 fix
    links_dim = "numLinks" if "numLinks" in weights.dims else "num_links"

    # precompute link lengths
    link_length = weights["link_length"].values

    sparse_weights = []
    for i, nl in enumerate(link_length):
        w = weights.isel({mask_dim: i, links_dim: slice(0, nl)})
        sparse_weights.append(compute_weights_matrix(w))

    return sparse_weights

def compute_weights_matrix(weights):
    """
    Convert the weights from CDO to a numpy array
    """

    # CDO style weights: load them eagerly, they are read only once here
    src_address = weights.src_address.values - 1
    dst_address = weights.dst_address.values - 1
    remap_matrix = weights.remap_matrix.values[:, 0]
    w_shape = (weights.sizes["src_grid_size"], weights.sizes["dst_grid_size"])

    # drop the links with zero weight: CDO writes some (e.g. bilinear, from masked source points),
    # and they would turn into NaN the target cells of a NaN source point (NaN * 0 = NaN)
    nonzero = remap_matrix != 0
    if not nonzero.all():
        src_address, dst_address, remap_matrix = src_address[nonzero], dst_address[nonzero], remap_matrix[nonzero]

    # Build the sparse matrix once, so that it is not rebuilt (and the weights
    # are not re-read from disk) at every compute of the regridded data
    sparse_weights = sparse.COO(
        [src_address, dst_address], remap_matrix, shape=w_shape
    )

    # Wrap it as a single-chunk dask array. name=False avoids hashing the whole
    # matrix, meta keeps the numpy-like behaviour of the previous delayed version
    return dask.array.from_array(
        sparse_weights, chunks=-1, asarray=False, name=False,
        meta=numpy.empty((0, 0), dtype=remap_matrix.dtype)
    )


def mask_na_thres(target, covered_fraction, na_thres, fill_value=numpy.nan):
    """
    Mask the target cells covered by too few valid source points.

    ``na_thres`` is the maximum fraction of missing weights of a target cell, so the cell is masked when its
    covered fraction is below ``1 - na_thres`` (as CDO, where ``dst_grid_frac < REMAP_AREA_MIN``). The cells
    without any valid source point are always masked, whatever ``na_thres``.

    Args:
        target: Data (or mask) to be masked.
        covered_fraction: Fraction of each target cell covered by valid source points, from 0 to 1: the
                          ``dst_grid_frac`` of the weights, or the sum of the valid weights with ``skipna``
                          (see ``Regridder.tensordot``).
        na_thres (float): Maximum fraction of missing weights, between 0 and 1.
        fill_value: Value set where the target cell is masked (default: NaN).

    Returns:
        ``target`` with ``fill_value`` where ``covered_fraction < 1 - na_thres`` or ``covered_fraction <= 0``.
    """
    masked = (covered_fraction < 1.0 - na_thres) | (covered_fraction <= 0.0)
    return dask.array.where(masked, fill_value, target)


def compute_target_mask(weights, na_thres):
    """
    Target mask applied without ``skipna``, computed once from the weights: ``dst_grid_imask`` (cells with
    at least one valid link, see ``mask_weights``) and, for the conservative methods, ``mask_na_thres`` on the
    ``dst_grid_frac`` of the weights (for the other methods it is not a fraction, see ``Regridder.tensordot``).

    Args:
        weights (xarray.Dataset): The CDO weights (2D or 3D), with ``dst_grid_imask`` from ``mask_weights``.
        na_thres (float): Maximum fraction of missing weights, between 0 and 1.

    Returns:
        xarray.DataArray: Boolean mask with the dims of ``dst_grid_imask``, True where the target cell is kept.
    """
    imask = weights['dst_grid_imask']
    mask = dask.array.asarray(imask.data).astype(bool)
    conservative = weights.attrs.get('normalization') in ('fracarea', 'destarea')
    if conservative and 'dst_grid_frac' in weights and na_thres < 1.0:
        mask = mask_na_thres(mask, dask.array.asarray(weights['dst_grid_frac'].data), na_thres, fill_value=False)
    return imask.copy(data=mask.compute())


def mask_tensordot(src_mask, weights_matrix):
    """Apply the weights to the source mask to return the destination mask: 1 where the target cell receives
    at least one link from a valid source point, 0 where it receives none. No threshold is applied here:
    na_thres acts on the dst_grid_frac of the weights, in compute_target_mask"""

    valid_weights = dask.array.tensordot(src_mask, weights_matrix, axes=1)
    return dask.array.where(valid_weights > 0, 1, 0)


def mask_weights(weights, weights_matrix, mask_dim=None):
    """Precompute the target mask of the CDO weights, as the dst_grid_imask variable: 1 where the target
    cell has at least one link from a valid source point (src_grid_imask), 0 otherwise.
    Takes as input the weights from CDO and the precomputed weights matrix. Handles the 3D case.

    Lazy evaluation and no in-place modification.

    Args:
        weights: xarray Dataset containing the weights and source mask.
        weights_matrix: Precomputed weights matrix (sparse arrays).
        mask_dim: Name of the mask dimension if 3D regridding is used.
    """

    src_mask = weights.src_grid_imask
    if mask_dim is not None:
        # Create list of masked arrays lazily
        masked_levels = [
            mask_tensordot(src_mask.isel({mask_dim: i}).data, weights_matrix[i])
            for i in range(weights.sizes[mask_dim])
        ]
        # Stack all levels at once (lazy operation)
        dst_mask = dask.array.stack(masked_levels, axis=0).compute()
        # Create new DataArray without in-place modification
        weights = weights.assign(dst_grid_imask=([mask_dim] + list(weights['dst_grid_imask'].dims[1:]), dst_mask))
    else:
        mask = src_mask.data
        dst_mask = mask_tensordot(mask, weights_matrix).compute()  # Compute the result to avoid lazy evaluation issues
        weights = weights.assign(dst_grid_imask=(weights['dst_grid_imask'].dims, dst_mask))

    return weights


def check_mask(weights, mask_dim=None):
    """
    Check if the target mask has all values equal to 1 (or True).
    Returns a boolean indicating whether the mask is full (a boolean per level in the 3D case).
    Takes the weights Dataset (dst_grid_imask) or a mask DataArray (compute_target_mask).
    """
    wdst = weights['dst_grid_imask'] if 'dst_grid_imask' in getattr(weights, 'variables', {}) else weights

    if mask_dim is not None:
        # Reduce along all dimensions except the vertical dimension
        dims_to_reduce = [dim for dim in wdst.dims if dim != mask_dim]
        mask = ~(wdst == 1).all(dim=dims_to_reduce)
    else:
        # Check if all values are 1 across all dimensions
        mask = ~(wdst == 1).all()

    # Compute the result if it's a dask array
    return mask.compute()
