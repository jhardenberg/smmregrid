Installation
============


smmregrid is a lightweight python package, but it depends on ``cdo`` for weights installation, thus both installation options requires conda/mamba. 
We recommend to use `mamba <https://mamba.readthedocs.io/en/latest/user_guide/mamba.html>`_ since it provides a lighter and deal in a better way with dependencies.

Using PyPi
----------

.. warning::

    Please note that although smmregrid is distributed via PyPi, it depends on packages that currently are available 
    only on conda-forge and on configuration files available from the GitHub repository.
    Therefore, the installation via pip requires the creation of a conda environment as well as the clone from the repository.


It will bring you the last version available on PyPi.
You can create a conda/mamba environment which incudes the python, `eccodes <https://github.com/ecmwf/eccodes-python>`_ and `cdo <https://code.mpimet.mpg.de/projects/cdo/>`_ dependencies, and then install smmregrid.
However, you should start by cloning the repository from GitHub, since the configuration files used for running ECmean4 are placed there ::

    > git clone https://github.com/jvonhard/smmregrid.git
    > mamba create -n smmregrid "python>=3.8" cdo eccodes
    > mamba activate smmregrid
    > pip install smmregrid


Using GitHub
------------

This method will allow you to have access at the most recent ECmean4 version but it requires a bit more of effort.

As before, should clone from the Github Repository ::

    > git clone https://github.com/jvonhard/smmregrid.git
    
.. note ::

    Please note that if you clone with HTTPS you will not be able to contribute to the code, even if you are listed as collaborator.
    If you want to be a developer you should clone with SSH and you should add your own SSH key on the GitHub portal: 
    please check the `procedure on the Github website <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_ .

Then you can through the ECmean4 folder ::

    > cd smmregrid

and then you can set up the conda/mamba environment ::

    > mamba env create --name smmregrid -f environment.yml

Then you should activate the environment ::

    > mamba activate smmregrid


Requirements
------------

The required packages are listed in ``environment.yml`` and in ``pyproject.toml``.
A secondary environment available in  ``dev-environment.yml`` can be used for development, including testing capabilities and jupyter notebooks. 

.. note::
	Both Unix and MacOS are supported. Python >=3.8 is requested.




