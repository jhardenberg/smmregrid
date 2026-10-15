Installation
============


smmregrid is a lightweight python package, but it depends on ``cdo`` for weights installation, thus both installation options requires conda/mamba. 
We recommend to use `mamba <https://mamba.readthedocs.io/en/latest/user_guide/mamba.html>`_ since it provides a lighter and deal in a better way with dependencies.

Using conda/mamba
--------------------

Simplest way is to install ``smmregrid`` using conda or mamba getting the code from conda-forge channel.

    > mamba create -n smmregrid -c conda-forge smmregrid

This will install ``smmregrid`` along with all required binary dependencies, including ``cdo`` and ``eccodes``.
On macOS, see the dedicated section below.

.. note::
   Make sure to have [mamba](https://mamba.readthedocs.io/en/latest/user_guide/mamba.html) 
   or [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your system. 


Using PyPi
----------

It will bring you the last version available on PyPi.
You can create a conda/mamba environment which incudes the python, `eccodes <https://github.com/ecmwf/eccodes-python>`_ and `cdo <https://code.mpimet.mpg.de/projects/cdo/>`_ dependencies, and then install smmregrid.

    > mamba create -n smmregrid "python>=3.8" cdo eccodes
    > mamba activate smmregrid
    > pip install smmregrid


Using GitHub
------------

This method will allow you to have access at the most recent smmregrid (i.e. unreleased main) version but it requires a bit more of effort.

As before, should clone from the Github Repository ::

    > git clone https://github.com/jhardenberg/smmregrid.git
    
.. note ::

    Please note that if you clone with HTTPS you will not be able to contribute to the code, even if you are listed as collaborator.
    If you want to be a developer you should clone with SSH and you should add your own SSH key on the GitHub portal: 
    please check the `procedure on the Github website <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_ .

Then you can through the smmregrid folder ::

    > cd smmregrid

and then you can set up the conda/mamba environment ::

    > mamba env create --name smmregrid -f environment.yml

Then you should activate the environment ::

    > mamba activate smmregrid


macOS
-----

Recent CDO versions (> 2.5.0) are not available from conda-forge for macOS. 
On macOS, CDO must therefore be installed with `Homebrew <https://brew.sh>`_, while python and the other dependencies 
are still installed with conda/mamba. The conda-forge package of smmregrid does not include CDO on macOS ::

    > brew install cdo
    > mamba create -n smmregrid -c conda-forge smmregrid

Or, installing from PyPI ::

    > brew install cdo
    > mamba create -n smmregrid "python>=3.8" eccodes numba
    > mamba activate smmregrid
    > pip install smmregrid

If you are working from the GitHub repository, use the ``environment-macos.yml`` file, which does not include CDO ::

    > brew install cdo
    > mamba env create --name smmregrid -f environment-macos.yml


Which CDO is used
-----------------

smmregrid looks for the CDO executable in this order:

1. the ``cdo`` argument of :class:`Regridder` / :class:`CdoGenerate`, if provided;
2. the ``CDO`` environment variable (the same used by python-cdo);
3. ``cdo`` in the ``PATH``.

On macOS, make sure that no ``cdo`` from conda-forge is installed in your environment (e.g. pulled in by another package), 
since it would shadow the Homebrew one in the ``PATH``. If you cannot avoid it, point smmregrid to the Homebrew CDO explicitly ::

    > export CDO=$(brew --prefix)/bin/cdo

The CDO version is checked at runtime: if it is older than the minimum supported version an error is raised.
You can check which version is used with ::

    > python -c "import smmregrid; print(smmregrid.cdo_version())"


Requirements
------------

The required packages are listed in ``environment.yml`` (Linux) and ``environment-macos.yml`` (macOS) and in ``pyproject.toml``.

.. note::
	Linux and macOS are supported. Python >=3.8 and CDO >= 2.6.2 are required. On macOS, CDO must be installed with Homebrew.
