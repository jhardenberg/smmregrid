Masking: ``na_thres``
=====================

Regridding is a weighted average: each target cell takes the values of the source cells that overlap it.
When some of these source cells are missing (NaN, typically land in ocean data) the average rests on fewer
points, and may stop being representative. smmregrid sets these target cells to NaN with one parameter,
``na_thres``, and one rule. How the missing values enter the weights (``mask_dim``, ``skipna``) is described
in :doc:`regridder`.

The rule
--------

A target cell is set to NaN when the fraction of it covered by valid source cells is smaller than
``1 - na_thres``, and always when no valid source cell contributes to it, whatever ``na_thres``.

``na_thres`` is the *maximum fraction of missing weights* of a target cell, a number between 0 and 1 (the name and
the meaning come from xESMF): with 0 a single missing source cell is enough to mask the target cell, with 1 a target
cell is masked only when it has no valid source cell at all. For conservative remapping it is the same test as the
``REMAP_AREA_MIN`` of CDO, which masks the target cells whose covered fraction (the ``dst_grid_frac`` of the weights)
is below the threshold: ``na_thres = 1 - remap_area_min``.

The rule is the same with and without ``skipna``. What changes is where the covered fraction comes from:

* **without** ``skipna`` CDO generates the weights with the mask of the data (one set per level with ``mask_dim``),
  and writes the covered fraction of each target cell in the weights (``dst_grid_frac``). smmregrid reads it.
* **with** ``skipna`` CDO generates the weights from a grid without mask, so the weights know nothing about the
  missing values. At each call smmregrid computes the covered fraction from the data: the sum of the weights of
  the target cell on valid source cells. With ``destarea`` the weights are the overlap areas over the whole target
  cell, so this sum is already the covered fraction. With ``fracarea`` they are normalized by the part of the cell
  covered by the source grid, so the sum is multiplied by that coverage, written by CDO in the unmasked weights
  (``dst_grid_frac``: 1 inside a global grid, the geometric coverage at the edge of a regional grid). For
  conservative remapping this is the same number that CDO would write as ``dst_grid_frac`` with that mask, to
  machine precision and for both normalizations: with ``fracarea`` the factor carries the rounding of the areas
  computed by CDO (1 ± 1e-13), so that the cells covered by exactly ``1 - na_thres`` (half of the cell, a common
  case) are decided as CDO does. Without it a few of these cells flip.

Per method
----------

The covered fraction is a real number only for the conservative methods, whose weights are areas. For the interpolation
methods the weights are coefficients between points: without ``skipna`` CDO has already decided which target cells get
a value, with ``skipna`` the share of the weights on valid points plays the role of the covered fraction.

=====================  ============================================  =================================================
Method                 ``skipna=False``                              ``skipna=True``
=====================  ============================================  =================================================
``con`` (fracarea)     ``dst_grid_frac`` of the weights, relative    share of the valid weights times ``dst_grid_frac``
                       to the part of the cell covered by the        of the unmasked weights: the same number.
                       source grid. ``na_thres`` effective.          ``na_thres`` effective.
``con`` (destarea)     ``dst_grid_frac``, absolute (relative to      sum of the valid weights (areas over the whole
                       the whole target cell). ``na_thres``          target cell): the same number, absolute.
                       effective.                                    ``na_thres`` effective.
``bil``, ``bic``       CDO drops a target cell as soon as one of      share of the weights on valid neighbours (quarters
                       its neighbours is missing: ``na_thres`` has    for ``bil``). ``na_thres`` effective: 1e-6 masks
                       no effect, a warning is logged if a value is   the cell as soon as a neighbour is missing, as
                       given.                                         CDO; larger values keep it, renormalized on the
                                                                      valid neighbours.
``nn``, others         one weight: the cell is valid or not.          one weight: ``na_thres`` has no effect.
                       ``na_thres`` has no effect.
=====================  ============================================  =================================================

Without ``skipna`` a target cell is NaN if it has no valid source cell, and for the conservative methods also when its
``dst_grid_frac`` is below ``1 - na_thres``. With ``skipna`` the same test is applied to the covered fraction computed
from the data.

With ``skipna`` and ``bil``, CDO and smmregrid can differ on a few cells: when the bilinear search fails CDO falls back to
a distance-weighted average of the valid neighbours (the same renormalization as ``skipna``), and keeps cells that
``na_thres=1e-6`` masks. The weights do not say which cells these are.

``'auto'`` and ``'cdo'``
------------------------

Instead of a number, ``na_thres`` can be ``'auto'`` (the default) or ``'cdo'``:

========================  =========  =================  ==============
``na_thres``              ``con``    ``bil``, ``bic``   ``nn``, others
========================  =========  =================  ==============
``'auto'`` (smmregrid)    0.5        1e-6               1.0
``'cdo'``                 1.0        1e-6               1.0
========================  =========  =================  ==============

``'cdo'`` replicates CDO, where ``REMAP_AREA_MIN`` is 0 by default: for conservative remapping only the target cells without
any valid source cell are masked. ``'auto'`` is the smmregrid default, and it additionally masks the conservative target
cells covered by less than half of valid source cells (``na_thres=0.5``), to avoid the erosion of coasts by cells made of
a thin sliver of ocean. For bilinear and bicubic both mask the target cell as soon as a neighbour is missing, as CDO.

``remap_area_min``
------------------

``remap_area_min`` is the name of the CDO option, and it was a separate argument of ``Regridder`` up to version 0.2.0.
It is deprecated: for the conservative methods it is translated to ``na_thres = 1 - remap_area_min`` with a
``DeprecationWarning``, for the other methods it is ignored (as in CDO, where it only acts on conservative remapping).
It cannot be used together with a different ``na_thres``.

Examples
--------

.. code-block:: python

    # the smmregrid default: conservative, mask the target cells with less than half of their area covered by valid source cells
    regridder = Regridder(source_grid, target_grid, method='con', skipna=True)

    # replicate CDO: no masking by area
    regridder = Regridder(source_grid, target_grid, method='con', skipna=True, na_thres='cdo')

    # conservative, mask the target cells with less than 80% of their area covered (CDO remap_area_min=0.8)
    regridder = Regridder(source_grid, target_grid, method='con', skipna=True, na_thres=0.2)

    # bilinear: set to NaN the target cells with more than 30% of the weights on missing source cells
    regridder = Regridder(source_grid, target_grid, method='bil', skipna=True, na_thres=0.3)

In the code the rule is one function of ``smmregrid.weights``, ``mask_na_thres``, applied to the ``dst_grid_frac`` of the
weights (without ``skipna``) or to the covered fraction computed from the data at each call by ``Regridder.tensordot`` (with
``skipna``). The target mask stored in the weights (``dst_grid_imask``, built by ``mask_weights``) only marks the target cells
with at least one valid source cell, and does not depend on ``na_thres``. Without ``skipna`` both depend only on the weights,
so ``Regridder`` combines them once at init in a single target mask (``compute_target_mask``) applied at each call.
