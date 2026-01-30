"""Weight calculation utilities"""

import dask
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

    # CDO style weights
    src_address = weights.src_address - 1
    dst_address = weights.dst_address - 1
    remap_matrix = weights.remap_matrix[:, 0]
    w_shape = (weights.sizes["src_grid_size"], weights.sizes["dst_grid_size"])

    # Create a sparse array from the weights
    sparse_weights_delayed = dask.delayed(sparse.COO)(
        [src_address.data, dst_address.data], remap_matrix.data, shape=w_shape
    )
    sparse_weights = dask.array.from_delayed(
        sparse_weights_delayed, shape=w_shape, dtype=remap_matrix.dtype
    )

    return sparse_weights


def mask_tensordot(src_mask, weights_matrix):
    """Apply tensor dot product to source mask to return destination mask"""

    target_mask = dask.array.tensordot(src_mask, weights_matrix, axes=1)
    target_mask = dask.array.where(target_mask < 0.5, 0, 1)
    return target_mask


def mask_weights(weights, weights_matrix, mask_dim=None):
    """This functions precompute the mask for the target interpolation
    Takes as input the weights from CDO and the precomputed weights matrix
    Return the target mask: handle the 3d case.
    
    Claude-based solution with full lazy evaluation and no in-place modification.
    
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
        dst_mask = dask.array.stack(masked_levels, axis=0)
        # Create new DataArray without in-place modification
        weights = weights.assign(dst_grid_imask=([mask_dim] + list(weights['dst_grid_imask'].dims[1:]), dst_mask))
    else:
        mask = src_mask.data
        dst_mask = mask_tensordot(mask, weights_matrix)
        weights = weights.assign(dst_grid_imask=(weights['dst_grid_imask'].dims, dst_mask))

    return weights


# def mask_weights(weights, weights_matrix, mask_dim=None):
#     """This functions precompute the mask for the target interpolation
#     Takes as input the weights from CDO and the precomputed weights matrix
#     Return the target mask: handle the 3d case"""

#     src_mask = weights.src_grid_imask
#     if mask_dim is not None:
#         for i in range(weights.sizes[mask_dim]):
#             mask = src_mask.isel({mask_dim: i}).data
#             weights['dst_grid_imask'].isel({mask_dim: i}).data[:] = mask_tensordot(mask, weights_matrix[i])
#     else:
#         mask = src_mask.data
#         weights['dst_grid_imask'].data = mask_tensordot(mask, weights_matrix)

#     return weights

def check_mask(weights, mask_dim=None):
    """
    Check if the target mask has all values equal to 1.
    Returns a boolean indicating whether the mask is full.
    Handles the 3D case efficiently.
    """
    wdst = weights['dst_grid_imask']

    if mask_dim is not None:
        # Reduce along all dimensions except the vertical dimension
        dims_to_reduce = [dim for dim in wdst.dims if dim != mask_dim]
        mask = ~(wdst == 1).all(dim=dims_to_reduce)
    else:
        # Check if all values are 1 across all dimensions
        mask = ~(wdst == 1).all()

    # Compute the result if it's a dask array
    return mask.compute()
