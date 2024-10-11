"""Weight calculation utilities"""

import dask.array
import sparse
import logging

loggy = logging.getLogger(__name__)


def compute_weights_matrix3d(weights, vert_coord='lev'):
    """
    Convert the weights from CDO to a list of numpy arrays
    """

    # CDO 2.2.0 fix
    if "numLinks" in weights.dims:
        links_dim = "numLinks"
    else:
        links_dim = "num_links"

    sparse_weights = []
    nvert = weights[vert_coord].values.size
    for i in range(0, nvert):
        w = weights.loc[{vert_coord: i}]
        nl = w.link_length.values
        w = w.isel(**{links_dim: slice(0, nl)})
        sparse_weights.append(compute_weights_matrix(w))

    return sparse_weights


def compute_weights_matrix(weights):
    """
    Convert the weights from CDO to a numpy array
    """
    w = weights

    # CDO style weights
    src_address = w.src_address - 1
    dst_address = w.dst_address - 1
    remap_matrix = w.remap_matrix[:, 0]
    w_shape = (w.sizes["src_grid_size"], w.sizes["dst_grid_size"])

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


def mask_weights(weights, weights_matrix, vert_coord=None):
    """This functions precompute the mask for the target interpolation
    Takes as input the weights from CDO and the precomputed weights matrix
    Return the target mask: handle the 3d case"""

    src_mask = weights.src_grid_imask
    if vert_coord is not None:
        for nlev in range(len(weights[vert_coord])):
            mask = src_mask.loc[{vert_coord: nlev}].data
            weights['dst_grid_imask'].loc[{vert_coord: nlev}] = mask_tensordot(mask, weights_matrix[nlev])
    else:
        mask = src_mask.data
        weights['dst_grid_imask'].data = mask_tensordot(mask, weights_matrix)

    return weights


def check_mask(weights, vert_coord=None):
    """Check if the target mask is empty or full and
    return a bool to be passed to the regridder.
    Handle the 3d case (5x time faster)"""

    wdst = weights['dst_grid_imask']
    if vert_coord is not None:
        check = wdst.mean(dim=tuple(dim for dim in wdst.dims if dim != vert_coord))
        out = (check != 1).data.tolist()
    else:
        check = wdst.mean()
        out = (check != 1).data

    return out


# def check_mask_old(weights, vert_coord=None):
#     """Old version: check if the target mask is empty or full and
#     return a bool to be passed to the regridder.
#     Handle the 3d case"""

#     if vert_coord is not None:
#         result = []
#         for nlev in range(len(weights[vert_coord])):
#             w = weights['dst_grid_imask'].loc[{vert_coord: nlev}]
#             v = w.sum() / len(w)
#             if v == 1:
#                 result.append(False)
#             else:
#                 result.append(True)
#         return result
#     else:
#         w = weights['dst_grid_imask']
#         v = w.sum() / len(w)
#         if v == 1:
#             return False
#         else:
#             return True
