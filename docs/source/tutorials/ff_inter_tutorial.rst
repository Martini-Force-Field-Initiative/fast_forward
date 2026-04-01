``ff_inter``
*************

This tutorial assumes you have already completed the :doc:`ff_map_tutorial` tutorial

The aim of this tutorial is to show how ``ff_inter`` can be used to generate initial
coarse-grained (CG) parameters for a new molecule.

Preparing a topology
====================

To begin to generate a complete set of parameters, we must first sketch how the
molecule looks at a CG resolution. Fast-Forward has a flexible syntax for this,
which means that in the first instance, we need only to define the bonds between
beads in the molecule, from which other bonded interactions can be generated.

What this means in practice is that we can write ``GROMACS itp files`` for the new molecule
containing only the ``[ moleculetype ]``, ``[ atoms ]``, and ``[ bonds ]`` directives.

For our GSH molecule, this means the itp file can simply look like this:

.. code-block::

    [ moleculetype ]
    GSH 1

    [ atoms ]
    1 SQ5n 1 GSH CAC1 1 -1.0
    2 Q4p  1 GSH AMC1 2  1.0
    3 P2   1 GSH AMD1 3  0.0
    4 TC6  1 GSH SUL1 4  0.0
    5 P2   1 GSH AMD2 5  0.0
    6 SQ5n 1 GSH CAC2 6 -1.0

    [ bonds ]
    1 2 1 0.300 7500 ; CAC1_AMC1
    2 3 1 0.300 7500 ; AMC1_AMD1
    3 4 1 0.300 7500 ; AMD1_SUL1
    3 5 1 0.300 7500 ; AMD1_AMD2
    5 6 1 0.300 7500 ; AMD2_CAC2


In the above example, also available in the `data <AA <https://github.com/Martini-Force-Field-Initiative/fast_forward/tree/main/fast_forward/tests/data/GSH>`_
folder, we have given some default values to the bond parameters for now; they do not matter
in the first instance. The most important aspect of this topology file is that it
has been annotated with comments in the bonds, indicating how the interactions are grouped.
For our GSH molecule, because we assume that each interaction is unique, they each get
a different annotation. This may be different for a molecule like a polymer, where we have
repeated interactions between monomeric units. In this case, the same grouping can be
indicated in the comments, which will then average over `all` interactions in the
group to generate the final parameters.


Running ``ff_inter``
=====================

With the initial topology prepared, we can run the ``ff_inter`` subprogram

.. code-block::

    ff_inter -f mapped.xtc -s mapped.tpr -i GSH.itp -itp-mode all -max-dihedral 10 -plots -dist-matrix

The above command should result in four sets of files:

* \*.dat - dat files containing time series interaction data
* \*_distr.dat - dat files containing binned interaction data
* GSH.itp - An updated topology file
* fitted_interactions.png - a compound plot showing how ``ff_inter`` has fitted each interaction distribution

The program will have also backed up the input ``GSH.itp`` to a file called ``#GSH.itp.1#``
in the standard Gromacs way.

The distribution files will be useful later when we come to assess the interactions, so we
can compare the input distributions to what was simulated.

``ff_inter`` output
====================

Let's now inspect part of the new ``GSH.itp`` file, taking the ``[ bonds ]`` directive.
The new directive looks like this:

.. code-block::

    [ bonds ]
    1 2 1 0.264 5925.634 ; CAC1_AMC1
    3 4 1 0.289 6211.916 ; AMD1_SUL1
    3 5 1 0.359 9607.543 ; AMD1_AMD2
    5 6 1 0.329 5794.11 ; AMD2_CAC2

    #ifdef FLEXIBLE
    2 3 1 0.383 10000 ; AMC1_AMD1
    #endif

    [ constraints ]
    #ifndef FLEXIBLE
    2 3 1 0.383 ; AMC1_AMD1
    #endif

We can see a few things:

1. The bond lengths and force constants have been updated according to the fitted data.
2. One of the bonds has been converted into a constraint. The constraint is decorated with a Gromacs conditional, for energy minimisation purposes.
3. The comments have been retained.

The updated itp file should also have ``[ angles ]`` and ``[ dihedrals ]`` directives,
covering the bonds that were originally defined.

With the new topology prepared, you should now be ready to run a first simulation with the
new molecule. Once the simulation has been completed, you will be ready to do the :doc:`ff_assess_tutorial` tutorial.













