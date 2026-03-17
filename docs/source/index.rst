Welcome to the documentation for Fast Forward
************************************************

Fast forward is a simple and fast tool for forward mapping trajectories and
computing interactions. Atomistic trajectories can be forward mapped from mapping
files that correspond to backwards mapping style format. Interactions are computed
from itp files automatically creating the time-series and distribution. Note that we
use the numba library for parallel acceleration.



Installation
============

.. code-block::

    git clone https://github.com/fgrunewald/fast_forward.git
    cd fast_forward
    pip install ./

Contents
========

.. toctree::
    :maxdepth: 2

    ff_map
    ff_inter
    ff_assess
