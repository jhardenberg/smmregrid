Masking: ``na_thres``
=====================

Regridding is a weighted average: each target cell takes the values of the source cells that overlap it.
When some of these source cells are missing (NaN, typically land in ocean data) the average rests on fewer
points, and may stop being representative. smmregrid sets to NaN these target cells with one threshold,
``na_thres``. For conservative remapping it is the same check as the ``remap_area_min`` of CDO, which is
deprecated in smmregrid in favour of ``na_thres``. How the missing values enter the weights (``mask_dim``,
``skipna``) is described in :doc:`regridder`.

The two numbers of a target cell
--------------------------------

For every target cell there are two numbers, both between 0 and 1:

* **missing weights**: the share of the weights of the cell that belongs to missing source cells. Only the
  source cells that overlap the target cell count.
* **covered area**: the fraction of the target cell covered by valid source cells, the quantity that CDO computes for
  conservative remapping. With the default normalization (``fracarea``) the weights are normalized by the part of the
  target cell that the source grid covers, so the covered area is *relative to that part*: it is one minus the missing
  weights. With ``destarea`` the weights are divided by the whole area of the target cell and the covered area is
  *absolute*: a part of the target cell outside the source grid counts as not covered.

The figure shows a real case: an ocean model on a curvilinear grid (IPSL, blue and grey, the cells are tilted along the coast
of Norway) regridded to a regular 3-degree grid (black squares) with ``skipna=True``. The source cells do not line up with
the target cells, and each target cell overlaps several of them, some with data (ocean) and some without (land).

.. image:: _static/masking_thresholds.png
   :alt: Curvilinear source grid over a regular target grid on a coast, with the missing weights and covered area of each target cell, and the cells set to NaN by na_thres 0.2 and by remap_area_min 0.8
   :width: 100%

The left panel gives the two numbers of each target cell. In the open sea nothing is missing and the covered area is 1.
Where the cell is partly on land they sum to 1: the cell at the top with 0.81 covered has 0.19 of its weights missing.
The two other panels apply a threshold on each number. ``na_thres=0.2`` masks the cells with more than 20% of the weights
missing, and ``remap_area_min=0.8`` (the CDO name) the cells with less than 80% covered: they are the same cells (red). The
cell at 0.81 is kept by both, just above the threshold, and all the others except the fully valid one are masked.

``na_thres``
------------

``na_thres`` is the *maximum fraction of missing weights*: a target cell is set to NaN when the share of its
weights coming from missing source cells is larger than ``na_thres``. It is a number between 0 and 1: with 0 a single
missing source cell is enough to mask the target cell, with 1 a target cell is masked only when no valid source
cell contributes. It is applied with and without ``skipna``, and with ``mask_dim``: without ``skipna`` the weights
are generated with the mask of the data and the covered fraction of each target cell comes from CDO (per level with
``mask_dim``), with ``skipna`` it is computed at each call from the valid source cells.

Instead of a number, ``na_thres`` can be ``'auto'`` (the default) or ``'cdo'``:

========================  =========  =================  ==============
``na_thres``              ``con``    ``bil``, ``bic``   ``nn``, others
========================  =========  =================  ==============
``'auto'`` (smmregrid)    0.5        1e-6               1.0
``'cdo'``                 1.0        1e-6               1.0
========================  =========  =================  ==============

``'cdo'`` replicates CDO, where ``REMAP_AREA_MIN`` is 0 by default: for conservative remapping only the target cells without
any valid source cell are masked. For bilinear and bicubic a missing neighbour makes the target cell NaN, as in CDO. ``'auto'`` is
the smmregrid default, and it additionally masks the conservative target cells covered by less than half
of valid source cells (``na_thres=0.5``), to avoid the erosion of coasts by cells made of a thin sliver of ocean.

Relation with ``remap_area_min`` of CDO
---------------------------------------

This comes from what CDO computes, not from smmregrid. In conservative remapping the weight of a source cell is the area
where it overlaps the target cell, so the weights are pieces of the target cell, and "weights on missing source cells" and
"area of the target cell on missing source cells" are the same quantity seen from the two sides. For example, a target
cell overlaps four source cells, with 0.4, 0.3, 0.2 and 0.1 of the cell, and the last two are land. The missing weights are
0.2 + 0.1 = 0.3 (what ``na_thres`` looks at) and the covered area is 0.4 + 0.3 = 0.7 (what ``remap_area_min`` looks at):
0.7 = 1 - 0.3. So ``na_thres = 1 - remap_area_min``: it is the same threshold on the same number (the figure shows it,
``na_thres=0.2`` and ``remap_area_min=0.8`` mask the same cells).

This holds when the weights sum to the area of the target cell that the source grid covers, as with ``fracarea``. It is a
convention of CDO: with ``fracarea`` a part of the target cell outside the source grid is ignored. With ``destarea`` the covered
area is absolute, and ``na_thres`` is compared with one minus that area, as the ``remap_area_min`` of CDO: a target cell cut
by the edge of the source grid can be masked even if none of its weights is missing.

For interpolation methods the weights are not areas but coefficients between points, and CDO has no covered area: its
fraction is 0 or 1. With ``bil`` the weights of a target cell are spread on several source cells and a share of them can be
missing, which only ``na_thres`` looks at; ``remap_area_min`` does nothing in CDO. With ``nn`` there is a single weight, so the
share is 0 or 1 and ``na_thres`` does not change the result.

``remap_area_min`` is still accepted by ``Regridder`` with a ``DeprecationWarning``: for the conservative methods it is
translated to ``na_thres = 1 - remap_area_min``, and it is ignored for the other methods. It cannot be used together
with a different ``na_thres``.

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

In the code the mask is a small function of ``smmregrid.weights``, ``mask_na_thres``, used by ``Regridder.tensordot`` and by
``mask_weights`` to build the target mask.
