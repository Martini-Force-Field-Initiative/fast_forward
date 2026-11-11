#!/usr/bin/env python3
"""
Main exe for scoring simulations against mapped trajectories
"""
import argparse
import sys
from glob import glob
from pathlib import Path
import numpy as np
import MDAnalysis as mda
import vermouth.forcefield
import fast_forward
from fast_forward.interaction_distribution import interaction_distribution
from fast_forward.itp_to_ag import ITPInteractionMapper
from fast_forward.itp_parser_sub import read_itp
from fast_forward.interaction_plots import make_distribution_plot, make_matrix_plot, make_distances_distribution_plot
from fast_forward.score import hellinger, score_matrix, calc_score
from fast_forward.sasa import MoleculeSASA
from fast_forward.report import report_interactions, report_distances, report_sasa
from collections import defaultdict

VERSION = 'fast_forward version {}'.format(fast_forward.__version__)

def _main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument('-f', type=str, dest="trajfile", help="simulated trajectory file")
    parser.add_argument('-s', type=str, dest="tprfile", help="simulated tpr file", default=None)
    parser.add_argument('-i', type=str, dest="itp_files", help="itp file", nargs='*')
    parser.add_argument('-d', type=str, dest='reference',
                        help="Path to directory with reference distributions")
    parser.add_argument('-pref', type=str, dest="prefix", help="common prefix to filename", default="")
    parser.add_argument('-plots', default=None, const=".", nargs="?",
                        dest="plots",
                        help="Make plots comparing distributions. Optionally provide a path (default: current dir)")
    parser.add_argument('-outliers', default=False, action='store_true', dest='exclude_outliers',
                        help='exclude outliers from overall score')
    parser.add_argument('-plot-data', default=False, action='store_true', dest="plot_data",
                        help='save data for making plots as single pickle file (implies -plots)')
    parser.add_argument('-score-weight', dest='hellinger_weight', type=float, default=0.7,
                        help='weight of the Hellinger distance in the distance score')
    parser.add_argument('-include-constraints', default=False, action='store_true', dest='include_constraints',
                        help='fully include constrained distances in the distance score score calculation')
    parser.add_argument('-dists', default=None, const=".", nargs="?",
                        dest="distribution_data",
                        help="Save text files with time series and distribution data for interactions. Optionally provide a path (default: current dir).")
    parser.add_argument('-interactions-only', default=False, action='store_true', dest='interactions_only',
                        help="Only assess bonded interaction distributions (bonds, angles, dihedrals); skip "
                             "the pairwise distance assessment entirely.")
    parser.add_argument('-sasa', default=False, action='store_true', dest='sasa',
                        help="Assess the solvent accessible surface area (SASA) of every molecule with fastsasa "
                             "(Martini bead radii from the bead types) against the reference written by "
                             "'ff_map -sasa' (<prefix><mol>_sasa_distr.dat in the -d directory). Requires -s (tpr) "
                             "and -i. With -dists, the CG time series and distribution are saved as well, also for every "
                             "bead except dummy beads of type U (<prefix><mol>_<resid>_<bead>_sasa[_distr].dat).")
    parser.add_argument('-sasa-surface', default=False, action='store_true', dest='sasa_surface',
                        help="With -sasa: write the CG structure and its surface dots (<prefix><mol>_sasa_dots_CG.pdb) "
                             "to the -dists path (default: current dir), in the frame closest to the mapped structure "
                             "from 'ff_map -sasa-surface' (<prefix><mol>_sasa_mapped.pdb in the -d directory), "
                             "superposed onto it; without it, in the frame with median SASA.")

    args = parser.parse_args()
    # saving the plot data implies making the plots (in the current dir by default)
    if args.plots is None and args.plot_data:
        args.plots = "."
    if args.sasa_surface and not args.sasa:
        parser.error("-sasa-surface requires -sasa")
    if args.sasa and not (args.tprfile and args.itp_files):
        parser.error("-sasa requires -s (tpr, for the molecule types) and -i (itp files)")

    distribution_files = glob(f'{args.reference}/*distr.dat')
    # create file map
    file_map = {}
    for f in distribution_files:
        p = Path(f)
        if p.name in file_map:
            print(f"Warning: duplicate reference file name '{p.name}', using the first one.")
            continue
        file_map[p.name] = p

    # load trajectory
    if args.tprfile:
        u = mda.Universe(args.tprfile, args.trajfile, in_memory=True)
    else:
        u = mda.Universe(args.trajfile, in_memory=True)

    if args.plots:
        plot_data = defaultdict(dict) # initialize only if needed

    # if itp file is provided use it
    if args.itp_files:
        ff = vermouth.forcefield.ForceField("dummy")
        for file_handle in args.itp_files:
            with open(file_handle) as _file:
                lines = _file.readlines()
            read_itp(lines, ff)

        interaction_mapper = ITPInteractionMapper(u, ff.blocks.values(), ff.blocks.keys())
        
        # evaluation of interactions
        h_score_dict = {'bonds': {},
                        'angles': {},
                        'dihedrals': {}}
        score_dict = {'bonds': {},
                        'angles': {},
                        'dihedrals': {}}
        # loop over molecules
        for molname, block in ff.blocks.items():
            interaction_groups, _, _ = interaction_mapper.get_interactions_group(molname)
            for inter_type in ['bonds', 'angles', 'dihedrals']:
                for group_name, pair_idxs in interaction_groups[inter_type].items():
                    distr, _ = interaction_distribution(u, inter_type, pair_idxs, group_name, args.prefix, save=args.distribution_data)
                    # calculate simulation distribution
                    probs = distr.T[1]
                    # read in reference distribution
                    reference_name = f"{args.prefix}{group_name}_{inter_type}_distr.dat"
                    try:
                        ref_file = file_map[reference_name]
                    except KeyError:
                        msg = f"{reference_name} file not found!"
                        if args.prefix == "":
                            msg += (" your prefix is set to the default value: consider changing it to the "
                                    "actual prefix.")
                        raise FileNotFoundError(msg) from None

                    reference_data = np.genfromtxt(ref_file)
                    # calculate hellinger distance between simulated and reference distributions
                    h_score = hellinger(probs/probs.sum(), reference_data.T[1] / reference_data.T[1].sum())
                    h_score_dict[inter_type][group_name] = h_score

                    score = calc_score(
                        reference_data.T[1]/ reference_data.T[1].sum(), probs/probs.sum(), 
                        [args.hellinger_weight, 1-args.hellinger_weight], inter_type
                        )
                    score_dict[inter_type][group_name] = score
                    if args.plots:
                        plot_data[inter_type][group_name] = {"x": reference_data.T[0],
                                                         "Reference": reference_data.T[1],
                                                         'Simulated': probs}
        if args.plots:
            make_distribution_plot(plot_data, args.plot_data, name=f'{args.plots}/interaction_distributions_{molname}')
        
        report_interactions(h_score_dict, score_dict, args.exclude_outliers)

        # evaluation of distances
        if args.interactions_only:
            print("Skipping distance assessment ('-interactions-only' given).")
        else:
            file_map = {}
            for f in distribution_files:
                p = Path(f)
                if p.name in file_map:
                    print(f"Warning: duplicate reference file name '{p.name}', using the first one.")
                    continue
                file_map[p.name] = p

            if not any(name.endswith('_distances_distr.dat') for name in file_map):
                print("WARNING: no '*_distances_distr.dat' reference files found in "
                     f"'{args.reference}'; skipping distance assessment. Re-run ff_inter "
                     "with '-dist-matrix' to generate them.")
            else:
                for molname, block in ff.blocks.items():
                    score_m, plot_data = score_matrix(molname=molname,
                                           block=block,
                                           universe=u,
                                           file_map=file_map,
                                           file_prefix=args.prefix,
                                           hellinger_weight=args.hellinger_weight,
                                           include_constraints=args.include_constraints)

                    atom_names = [f'{resid}_{name}' for (_, name), (_, resid) in zip(block.nodes(data='atomname'), block.nodes(data='resid'))]

                    if args.plots:
                        make_matrix_plot(score_m, atom_names, name=f'{args.plots}/score_matrix_{molname}')
                        make_distances_distribution_plot(
                            plot_data, atom_names, name=f'{args.plots}/distance_distributions_{molname}',save_plot_data=args.plot_data
                        )

                    report_distances(score_m, molname, atom_names)

        # evaluation of the solvent accessible surface area: score the SASA distribution of every
        # molecule of the CG trajectory against the reference written by 'ff_map -sasa'
        if args.sasa:
            print("INFO - Evaluating SASA")
            sasa_h_scores = {}
            sasa_scores = {}
            sasa_means = {}
            sasa_plot_data = defaultdict(dict)
            for molname in ff.blocks.keys():
                print(f"INFO - Calculating SASA of {molname}")
                mol_atoms = u.select_atoms(f"moltype {molname}")
                if mol_atoms.n_atoms == 0:
                    print(f"WARNING - No molecules of type {molname} in the trajectory; skipping its SASA.")
                    continue
                reference_name = f"{args.prefix}{molname}_sasa_distr.dat"
                try:
                    ref_file = file_map[reference_name]
                except KeyError:
                    available = sorted(name for name in file_map if name.endswith('_sasa_distr.dat'))
                    msg = (f"{reference_name} file not found! Generate the reference SASA with "
                           f"'ff_map -sasa {args.reference} -mols <AA name>:{molname}' to name it after the CG "
                           f"molecule.")
                    if available:
                        msg += f" SASA references found: {', '.join(available)}."
                    if args.prefix == "":
                        msg += (" your prefix is set to the default value: consider changing it to the "
                                "actual prefix.")
                    raise FileNotFoundError(msg) from None

                mol_sasa = MoleculeSASA(mol_atoms, molname, cg=True, prefix=args.prefix)
                if args.distribution_data:
                    # also save the SASA of every bead (without dummies), as the interactions
                    mol_sasa.cg_bead_sasa(ff.blocks[molname])
                    mol_sasa.save(args.distribution_data)
                probs = mol_sasa.distribution.T[1]
                reference_data = np.genfromtxt(ref_file)
                ref_probs = reference_data.T[1]
                if probs.sum() == 0 or ref_probs.sum() == 0:
                    print(f"WARNING - Empty SASA distribution for {molname}; skipping its score.")
                    continue
                
                # calculate scores
                sasa_h_scores[molname] = hellinger(probs/probs.sum(), ref_probs/ref_probs.sum())
                sasa_scores[molname] = calc_score(ref_probs/ref_probs.sum(), probs/probs.sum(),
                                                 [args.hellinger_weight, 1-args.hellinger_weight], 'sasa')
                sasa_means[molname] = (mol_sasa.series.mean(), np.average(reference_data.T[0], weights=ref_probs))

                if args.sasa_surface:
                    mol_sasa.write_cg_surface(reference_dir=args.reference, outdir=args.distribution_data or ".")
                if args.plots:
                    sasa_plot_data['sasa'][molname] = {"x": reference_data.T[0],
                                                       "Reference": ref_probs,
                                                       "Simulated": probs}
            if not sasa_scores:
                print("WARNING - No SASA could be assessed.")
            else:
                if args.plots:
                    make_distribution_plot(sasa_plot_data, args.plot_data, name=f'{args.plots}/sasa_distributions',
                                           data_name='plot_data_sasa.p')
                report_sasa(sasa_h_scores, sasa_scores, sasa_means)


def main():
    try:
        _main()
    except FileNotFoundError as err:
        sys.exit(f"Error: {err}")

if __name__ == '__main__':
    main()
