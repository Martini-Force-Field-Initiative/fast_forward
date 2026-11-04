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
      -sasa                 Calculate the solvent accessible surface area (SASA) of the atomistic trajectory. Save text files with time
                            series and distribution data. Optionally provide a path (default: current dir). The path
                            should be the same directory as you use with ff_inter -dists. (default: None)
      -sasa-surface         With -sasa: for the frame whose SASA is closest to the median, write the atomistic structure
                            with its surface dots (<mol>_sasa_dots_AA.pdb) and the mapped CG structure of the same
                            frame (<mol>_sasa_mapped.pdb) to the -sasa path. (default: False)
      -sasa-ndots SASA_NDOTS
                            Minimum number of surface dots per atom for -sasa-surface. (default: 240)

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

Solvent Accessible Surface Area (SASA)
======================================

The solvent accessible surface area (SASA) is a measure of the molecular shape and volume.

In fast_forward, the SASA is calculated with the Shrake-Rupley algorithm (as implemented in MDTraj), which
places test points on spheres around every atom, enlarged by the radius of a probe representing a water molecule
(0.191 nm), and counts the points that are not buried by neighbouring atoms. Atomistic structures use the van der
Waals radii of Rowland and Taylor; coarse-grained beads use 0.264, 0.230 and 0.191 nm for regular, small and tiny
beads.

The SASA calculation is enabled with the ``-sasa`` option, which requires MDTraj (``pip install mdtraj``).
The SASA is calculated for every copy of a molecule on its own, so copies do not bury each other. As for the
interaction distributions, the time series of all copies are concatenated (all frames of the first copy, then
of the second, ...). For each molecule type, the time series (``<mol>_sasa.dat``) and its distribution
(``<mol>_sasa_distr.dat``) are written; ``ff_assess -sasa`` treats the CG trajectory in the same way and uses the
distribution as the reference. The SASA surfaces described below only use the first copy.

.. _sasa-surfaces:

SASA surfaces
-------------

With ``-sasa-surface`` the solvent accessible surface itself is written as a dot PDB file, for visual comparison
of atomistic and coarse-grained models. Each file holds the structure the surface was computed for as ATOM
records (chain A), followed by the surface dots as HETATM records (chain X). The dots are generated as in
``gmx sasa -q`` (``-sasa-ndots`` sets the minimum number of dots per sphere) and lie on the probe-expanded spheres. All dots have the atom and residue name
``DOT`` and carry the residue id of the particle they belong to. The files are written with MDAnalysis and carry the citations
of fast_forward and the radii as REMARK records.

``ff_map -sasa -sasa-surface`` selects the frame of the first molecule of each type whose SASA is closest to the
median and writes the atomistic structure with its surface (``<mol>_sasa_dots_AA.pdb``) and the mapped structure
of the same frame (``<mol>_sasa_mapped.pdb``).

``ff_assess -sasa -sasa-surface -d <reference dir>`` reads ``<mol>_sasa_mapped.pdb``, picks the simulated CG
frame with the lowest RMSD to it and superposes that frame onto the mapped structure. It writes the CG structure
with its surface (``<mol>_sasa_dots_CG.pdb``). All files share the same coordinate frame and can be loaded
together, e.g.:

.. code-block::

    vmd -m GSH_sasa_dots_AA.pdb GSH_sasa_mapped.pdb GSH_sasa_dots_CG.pdb

Molecules are assumed to be whole; correct the trajectories for PBC beforehand (or use ``ff_map -pbc``).

Mapping using CGSmiles
======================

WIP

