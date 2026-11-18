"""
Assessment of the SASA of a CG trajectory (``ff_assess -sasa``): the SASA distribution of
every molecule, and of every bead, is scored against the references written by
``ff_map -sasa`` (``<prefix><mol>_sasa_distr.dat``, ``<prefix><mol>_<resid>_<bead>_sasa_distr.dat``).
"""
from collections import defaultdict
import numpy as np
from fast_forward.interaction_plots import make_distribution_plot
from fast_forward.report import report_sasa
from fast_forward.sasa import MoleculeSASA, REFERENCE
from fast_forward.score import hellinger, calc_score


def score_sasa(probs, reference_data, hellinger_weight):
    """
    Score a SASA distribution against its reference, as the bonded interactions.

    Parameters
    ----------
    probs : np.ndarray
        (n_bins,) simulated (CG) probability density
    reference_data : np.ndarray
        (n_bins, 2) reference distribution: bin centre (nm^2), probability density
    hellinger_weight : float
        weight of the Hellinger distance in the score

    Returns
    -------
    tuple of float or None
        (Hellinger distance, score, mean reference SASA in nm^2); None if either
        distribution is empty or not finite, or the scores are not finite (e.g. a
        reference without spread)
    """
    ref_probs = reference_data.T[1]
    for distr in (probs, ref_probs):
        # NaN would compare unequal to 0 and slip into the scores and their mean
        if not (np.all(np.isfinite(distr)) and distr.sum() > 0):
            return None
    h_score = hellinger(probs/probs.sum(), ref_probs/ref_probs.sum())
    score = calc_score(ref_probs/ref_probs.sum(), probs/probs.sum(), [hellinger_weight, 1-hellinger_weight], 'sasa')
    # e.g. a reference in a single bin has no spread, so the mean difference is NaN
    if not (np.isfinite(h_score) and np.isfinite(score)):
        return None
    return h_score, score, np.average(reference_data.T[0], weights=ref_probs)


def _missing_reference(molname, file_map, reference_dir, prefix):
    """Warn that the reference SASA of ``molname`` is missing."""
    reference_name = f"{prefix}{molname}_sasa_distr.dat"
    msg = (f"WARNING - No reference SASA for {molname} ({reference_name} in {reference_dir}); "
           f"skipping its SASA. Generate it with 'ff_map -sasa {reference_dir} "
           f"-mols <AA name>:{molname}' to name it after the CG molecule.")
    available = sorted(name for name in file_map if name.endswith('_sasa_distr.dat'))
    if available:
        msg += f" SASA references found: {', '.join(available)}."
    if prefix == "":
        msg += " your prefix is set to the default value: consider changing it to the actual prefix."
    print(msg)


def assess_bead_sasa(mol_sasa, file_map, reference_dir, hellinger_weight):
    """
    Score the SASA of every bead of ``mol_sasa`` against its reference
    ``<prefix><mol>_<resid>_<bead>_sasa_distr.dat``; beads without a reference or with an
    empty or invalid distribution are skipped with a warning.

    Returns
    -------
    bead_results : dict
        {bead: (Hellinger distance, score, CG mean, reference mean)}
    plot_data : dict
        {'sasa': {bead: plot entry}}, for :func:`make_distribution_plot`
    """
    molname, prefix = mol_sasa.mol_name, mol_sasa.prefix
    bead_results, plot_data = {}, defaultdict(dict)
    missing, invalid = [], []
    for bead, series, distr in zip(mol_sasa.bead_names, mol_sasa.bead_series, mol_sasa.bead_distributions):
        ref_file = file_map.get(f"{prefix}{molname}_{bead}_sasa_distr.dat")
        if ref_file is None:
            missing.append(bead)
            continue
        reference_data = np.genfromtxt(ref_file)
        result = score_sasa(distr.T[1], reference_data, hellinger_weight)
        if result is None:
            invalid.append(bead)
            continue
        h_score, score, ref_mean = result
        bead_results[bead] = (h_score, score, series.mean(), ref_mean)
        plot_data['sasa'][bead] = {"x": reference_data.T[0], "Reference": reference_data.T[1], "Simulated": distr.T[1]}
    if missing:
        print(f"WARNING - No reference SASA for the bead(s) {', '.join(missing)} of {molname} "
              f"({prefix}{molname}_<bead>_sasa_distr.dat in {reference_dir}); they are not "
              "assessed. 'ff_map -sasa' writes them for every bead of the mapping except dummy beads.")
    if invalid:
        print(f"WARNING - Empty or invalid (NaN) SASA distribution for the bead(s) {', '.join(invalid)} "
              f"of {molname}; skipping their score.")
    return bead_results, plot_data


def assess_sasa(universe, blocks, file_map, reference_dir, prefix="", hellinger_weight=0.7,
                distribution_data=None, plots=None, plot_data=False, sasa_surface=False,
                sasa_stat=REFERENCE):
    """
    Score the SASA of every molecule type in ``blocks``, and of its beads, against the
    references from ``ff_map -sasa``, plot the distributions and print the report.
    Molecule types without molecules in ``universe`` or without a reference are skipped
    with a warning.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        CG trajectory, with molecule types (tpr)
    blocks : dict
        {molecule name: vermouth.molecule.Block} from the itp files
    file_map : dict
        {file name: path} of the reference distributions
    reference_dir : str
        directory of the references, for the messages
    prefix : str
        common prefix of the file names
    hellinger_weight : float
        weight of the Hellinger distance in the score
    distribution_data : str or None
        directory to save the CG SASA time series and distributions to, also of the surfaces
    plots : str or None
        directory to save the plots to; no plots if None
    plot_data : bool
        also save the data of the plots
    sasa_surface : bool
        write the CG surface of every molecule type
    sasa_stat : str or float
        statistic choosing the frame of the surface (see :meth:`MoleculeSASA.write_cg_surface`)
    """
    print("INFO - Evaluating SASA")
    h_scores, scores, means, bead_results = {}, {}, {}, {}
    plot_entries = defaultdict(dict)
    for molname, block in blocks.items():
        print(f"INFO - Calculating SASA of {molname}")
        mol_atoms = universe.select_atoms(f"moltype {molname}")
        if mol_atoms.n_atoms == 0:
            print(f"WARNING - No molecules of type {molname} in the trajectory; skipping its SASA.")
            continue
        ref_file = file_map.get(f"{prefix}{molname}_sasa_distr.dat")
        if ref_file is None:
            # e.g. another solute or ions in the itp files: assess the others, as for missing beads
            _missing_reference(molname, file_map, reference_dir, prefix)
            continue

        mol_sasa = MoleculeSASA(mol_atoms, molname, cg=True, prefix=prefix)
        mol_sasa.cg_bead_sasa(block) # SASA of every bead (without dummies)
        if distribution_data:
            mol_sasa.save(distribution_data)
        probs = mol_sasa.distribution.T[1]
        reference_data = np.genfromtxt(ref_file)
        result = score_sasa(probs, reference_data, hellinger_weight)
        if result is None:
            print(f"WARNING - Empty or invalid (NaN) SASA distribution for {molname}; skipping its score.")
            continue
        h_scores[molname], scores[molname], ref_mean = result
        means[molname] = (mol_sasa.series.mean(), ref_mean)
        plot_entries['sasa'][molname] = {"x": reference_data.T[0], "Reference": reference_data.T[1], "Simulated": probs}

        if sasa_surface:
            mol_sasa.write_cg_surface(reference_dir=reference_dir, outdir=distribution_data or ".",
                                      statistic=sasa_stat)

        beads, bead_plot_data = assess_bead_sasa(mol_sasa, file_map, reference_dir, hellinger_weight)
        if beads:
            bead_results[molname] = beads
        if plots and bead_plot_data:
            make_distribution_plot(bead_plot_data, plot_data, name=f'{plots}/sasa_bead_distributions_{molname}',
                                   data_name=f'plot_data_sasa_beads_{molname}.p')

    if not scores:
        print("WARNING - No SASA could be assessed.")
        return
    if plots:
        make_distribution_plot(plot_entries, plot_data, name=f'{plots}/sasa_distributions',
                               data_name='plot_data_sasa.p')
    report_sasa(h_scores, scores, means, bead_results)
