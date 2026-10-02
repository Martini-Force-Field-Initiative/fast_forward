``ff_inter``
*************

This tutorial assumes you have already completed the :doc:`ff_map_tutorial` tutorial

The aim of this tutorial is to show how ``ff_inter`` can be used to generate initial
coarse-grained (CG) parameters for a new molecule.

Preparing a topology
====================

To begin to generate a complete set of parameters, we must first sketch how the
molecule looks at a CG resolution. Fast-Forward has a flexible syntax for this: by
default (``-interactions comments``), it will fit exactly the bonds, angles and
dihedrals that you have annotated with a comment in the input itp, and nothing else.
This gives you full control over which interactions get fitted.

What this means in practice is that we can write ``GROMACS itp files`` for the new molecule
containing the ``[ moleculetype ]``, ``[ atoms ]``, ``[ bonds ]``, ``[ angles ]`` and
``[ dihedrals ]`` directives, with dummy placeholder parameters, and annotate every
interaction we want fitted with a trailing comment.

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

    [ angles ]
    1 2 3 2 10 10 ; CAC1_AMC1_AMD1
    2 3 4 2 10 10 ; AMC1_AMD1_SUL1
    2 3 5 2 10 10 ; AMC1_AMD1_AMD2
    4 3 5 2 10 10 ; SUL1_AMD1_AMD2
    3 5 6 2 10 10 ; AMD1_AMD2_CAC2

    [ dihedrals ]
    2 3 5 6 1 10 1 1 ; AMC1_AMD1_AMD2_CAC2
    1 2 3 5 1 10 1 1 ; CAC1_AMC1_AMD1_AMD2
    1 2 3 4 1 10 1 1 ; CAC1_AMC1_AMD1_SUL1


In the above example, also available in the `AA <https://github.com/Martini-Force-Field-Initiative/fast_forward/tree/main/examples/GSH/AA>`_
folder, we have given some default, dummy values to the parameters for now; they do not
matter in the first instance, only the atom indices and function types need to be sensible,
since they are overwritten with fitted values. The most important aspect of this topology
file is that it has been annotated with comments, indicating how the interactions are
grouped. For our GSH molecule, because we assume that each interaction is unique, they
each get a different annotation.

Note that one dihedral which the bonds would otherwise permit, ``SUL1_AMD1_AMD2_CAC2``,
has been deliberately left out. Not every combination of bonds makes for a well-behaved
dihedral to fit: this one sits across a branch point in the molecule (``AMD1`` is bonded to
three other beads) and its distribution is too poorly sampled and structured to fit
reliably, producing wildly unphysical parameters. Leaving it out of the annotated itp is
how you tell ``ff_inter`` to skip it.

.. admonition:: Automatically guessing interactions

    Instead of manually annotating every interaction, ``-interactions guess`` can be
    used to automatically derive every possible angle and dihedral from the bonds you
    have defined, without needing to write them out by hand. This is convenient for a
    first pass, but as above, not every automatically guessed interaction is
    guaranteed to fit well, so it is worth inspecting the resulting ``[ angles ]`` and
    ``[ dihedrals ]`` directives (and the accompanying ``fitted_interactions.png``
    plot) before trusting them, and switching to ``-interactions comments`` to leave
    out any that don't fit well.

.. admonition:: Using repeated or common interactions

    This may be useful for a molecule like a polymer, where we have
    repeated interactions between monomeric units. In this case, the same grouping can be
    indicated in the comments, which will then average over `all` interactions in the
    group to generate the final parameters.

    If for example we have the following trimer molecule, consisting of 3 monomers which
    have a simple one bead side chain:

    .. code-block::

        [ atoms ]
        1 P4   1 RES BB  1  0.0
        2 SC1  1 RES SC1 2  0.0
        3 P4   1 RES BB  3  0.0
        4 SC1  1 RES SC1 4  0.0
        5 P4   1 RES BB  5  0.0
        6 SC1  1 RES SC1 6  0.0

        [ bonds ]
        ;;; backbone-backbone bonds
        1 3 1 0.3 10000 ; BB_BB
        3 5 1 0.3 10000 ; BB_BB
        ;;; backbone-sidechain bonds
        1 2 1 0.25 5000 ; BB_SC1
        3 4 1 0.25 5000 ; BB_SC1
        5 6 1 0.25 5000 ; BB_SC1
        ...

    In this molecule, we have 2 backbone-backbone bonds, and 3 backbone-sidechain bonds.
    Were this run through ``ff_inter`` with the annotations shown above, the subprogram
    would generate average distributions over all the interactions in each annotated group.




Running ``ff_inter``
=====================

With the initial topology prepared, we can run the ``ff_inter`` subprogram

.. code-block::

    ff_inter -f mapped.xtc -s mapped.tpr -i GSH.itp -interactions comments -max-dihedral 10 -plots -dists -dist-matrix

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
    1 2 1 0.278 5717.075 ; CAC1_AMC1
    3 4 1 0.273 4900.223 ; AMD1_SUL1
    3 5 1 0.357 8109.482 ; AMD1_AMD2

    #ifdef FLEXIBLE
    2 3 1 0.415 10000 ; AMC1_AMD1
    5 6 1 0.284 10000 ; AMD2_CAC2
    #endif

    [ constraints ]
    #ifndef FLEXIBLE
    2 3 1 0.415 ; AMC1_AMD1
    5 6 1 0.284 ; AMD2_CAC2
    #endif

We can see a few things:

1. The bond lengths and force constants have been updated according to the fitted data.
2. Two of the bonds have been converted into constraints. Each constraint is decorated with a Gromacs conditional, for energy minimisation purposes.
3. The comments have been retained.

The updated itp file will also have fitted values in its ``[ angles ]`` and
``[ dihedrals ]`` directives, for every interaction that was annotated in the input file.

With the new topology prepared, you should now be ready to run a first simulation with the
new molecule. Once the simulation has been completed, you will be ready to do the :doc:`ff_assess_tutorial` tutorial.













