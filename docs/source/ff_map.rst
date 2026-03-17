``ff_map``
***********

Background
==========

``ff_map`` is the mapping subprogram of fast forward. Using (a collection of)
mapping files, it will convert a trajectory of a system at all-atom resolution to
one at a coarse grained resolution.


Options
=======

.. code-block:: none

    Input Files:
      -f TRAJ               trajectory file (XTC/TRR/GRO/PDB ...) (default: None)
      -s TPR                tpr file (TPR) (default: None)
      -cgs [CGSMILES_STRS ...]
                            any number of cgsmiles strings. (default: None)
      -cgsl CGSMILES_LIBRARY
                            any number of cgsmiles strings. (default: None)
      -m MAP_FILE           .mapping file or path to directory of .map files (default: )
      -o OUTPATH            name of the mapped trajectory (XTC/GRO). If trajectory, first frame will also be written as a separate gro file (default: None)

    Mapping Options:
      -mode MODE            COG or COM mapping (default: COG)
      -pbc                  complete pbc with MDAnalysis; this is slow! (default: False)
      -mols MOL_NAMES [MOL_NAMES ...]
                            names of molecules to consider when mapping as in the [moleculetypes] directive (default: None)
      -add_H H_ASSOCIATION [H_ASSOCIATION ...]
                            atom-types corresponding to CH3, CH2, CH1 for aliphatic groups and CH2d for double bonds. (default: [])
      -[no]cgsorder         Whether cgsmiles strings passed with -cgs or -cgsl are in the same order as -mols (default: assume yes if passed only with -cgs and no if
                            -cgsl is used)
      -lib LIBRARY          loads library files (default: None)


Example
=======

The following is an example command for mapping trajectories:

.. code-block::

    ff_map -f trajectory.xtc -s topology.tpr -m molecule.map -o mapped.xtc -mols moleculename

This command will take the (atomistic) system described by the trajectory and topology by ``trajectory.xtc``
and ``topology.tpr``, and using the mapping description contained in the mapping files (see below), convert the
molecules ``moleculename`` to a trajectory at lower (i.e. coarse grained) resolution. It is strongly recommended
that the originating system is already pbc-complete. Although the ``-pbc`` flag may be used to correct pbc
artifacts, it is not fast.

The terms "mapped" and "pseudo-atomistic" are often used interchangeably. The ``mapped.xtc``
file will be a "fake" trajectory. The mapped trajectory has not been simulated `ab initio`, but the coordinates are
such that if a lower resolution simulation had been performed, and the beads followed a trajectory that perfectly
resembled the underlying higher resolution one. This is why we often refer to the ``mapped.xtc`` file as a
`pseudo-atomistic` one: the higher resolution trajectory was atomistic, and the new file is a coarse grained trajectory
whose coordinates are derived from it. However, there is nothing inherent in Fast-Forward that means the higher
resolution trajectory necessarily must be atomistic.

Mapping file (.map/.mapping) format
===================================

Note that the molecule names defined by the ``-map`` flag need to correspond to the ``[moleculetype]`` entries
in the itp files of the molecules that are to be mapped. Entries in mapping files
are written per residue and follow the backwards style mapping file format:

.. code-block::

    [ molecule ]
    <resnameAA> <resnameCG>
    [ martini ]
    <list of Martini beads>
    [ atoms ]
         1     <atomname AA> <bead CG> <bead CG>

For example, the Martini3 .mapping entry for Alanine (not including the
hydrogens) would look as follows and maps from the CHARMM force-field:

.. code-block::

    [ molecule ]
    ALA ALA
    [ martini ]
    BB SC1
    [ atoms ]
        1     N    BB
        2    HN    BB
        3    CA    BB
        5    CB    SC1
        9     C    BB
       10     O    BB

Alternatively, a directory of .map files can be passed to the  ``-m`` argument, such as
those used in `Vermouth-style <https://github.com/marrink-lab/vermouth-martinize>`_ mappings.
In this format, we use a separate map file for each residue. For example, the CHARMM map file
for Alanine is:

.. code-block::

    [ molecule ]
    ALA

    [from]
    charmm

    [to]
    martini3001

    [ martini ]
    BB SC1

    [ mapping ]
    charmm27 charmm36

    [ atoms ]
        1     N    BB
        2    HN    BB
        3    CA    BB
        4    HA    !BB
        5    CB    SC1
        6   HB1    !SC1
        7   HB2    !SC1
        8   HB3    !SC1
        9     C    BB
       10     O    BB

    [ chiral ]
      CB     CA    N    C
      HB1    CA    N    C
      HB2    CA    N    C
      HB3    CA    N    C

    [ chiral ]
      HA     CA    N    CB    C ; L-Ala
    ; HA     CA    N    C    CB ; D-Ala

Mapping trajectories + hydrogen reconstruction
==============================================

The Martini3 model is based on center of geometry mappings, for which
it is important to include the hydrogen atoms. However, certain atomistic
united-atom models have no explicit hydrogen atoms. In order to still obtain
good bonded interactions fast_forward can reconstruct the hydrogen atoms
using geometric rules. The only input required is a list of atom-types and
weather they correspond to CH, CH2, CH3 or double bonded CH group (CHd).

The algorithm is directly taken from the `buildH <https://github.com/patrickfuchs/buildH>`_ package.
If you use this feature please cite the `describing paper <https://joss.theoj.org/papers/10.21105/joss.03521>`_.

An example for POPC can be found in the example folder of the repository,
where a CHARMM36m POPC was stripped of all hydrogen and mapped with reconstruction.

Note the reconstruction only affects the mapping, atomistic coordinates with
hydrogen are not written out.


Mapping using CGSmiles
======================

WIP

