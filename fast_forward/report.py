import numpy as np

def report_interactions(h_score_dict, score_dict, exclude_outliers=False):
    """
    Print out a report of the interaction.

    Parameters
    ----------
    score_dict : dict
        Dictionary with interaction types as keys and another dictionary as values.
    exclude_outliers : bool, optional
        Whether to exclude outliers (scores >= 0.5) from the overall score calculation, defaults to False
    """

    buff = ''

    buff += " [ Interaction Distribution Report ]\n"
    buff += f"   Overall Score : {{mean:.3}} ± {{sem:.3}}\n\n"

    buff += " Interaction Scores:\n"
    buff += " 0 - identical, 1 - no overlap\n\n"

    buff += " Score guide:\n"
    buff += "   0.0-0.3 : good\n"
    buff += "   0.3-0.5 : ok\n"
    buff += "   0.5-1.0 : bad\n\n"
    buff += "\t\tHellinger distance (Scoring function)\n"


    if exclude_outliers:
        buff += "\n\tNB: values over 0.5 have been excluded from the above summary value\n\n"
    else:
        buff += '\n'

    overall_scores = []
    for interaction_type in h_score_dict.keys():
        buff += f" {interaction_type} \n"
        for atom_group, h_score in h_score_dict[interaction_type].items():
            score = score_dict[interaction_type][atom_group]
            if exclude_outliers and h_score >= 0.5: # arbitrary cutoff for now
                buff += f"\t{atom_group:20s}: {h_score:.2f} ({score:.2f})(excluded)\n"
            else:
                buff += f"\t{atom_group:20s}: {h_score:.2f} ({score:.2f})\n"
                overall_scores.append(score)
        buff += '\n'

    printable = buff.format(mean=np.array(overall_scores).mean(),
                            sem=np.array(overall_scores).std()/np.sqrt(len(overall_scores))
                            )

    print(printable)
    with open('report_interactions.out', 'w') as fout:
        fout.writelines(printable)


def report_distances(score_matrix, molname, atom_names):
    """
    Print out a report of the distance score matrix.

    Parameters
    ----------
    score_matrix : np.ndarray
        Score matrix for pairwise distances.
    molname : str
        Name of the molecule.
    atom_names : list of str
        List of atom names corresponding to the score matrix.
    """
    score_matrix_report = np.vstack([atom_names, score_matrix])
    score_matrix_report = np.column_stack([[''] + atom_names, score_matrix_report])
    buff = ''
    buff += f" [ Distance Distribution Report for {molname} ]\n "
    buff += f" Overall Score : {{mean:.3}} ± {{sem:.3}}\n\n"
    buff += f"   Max Score : {{max:.2f}}\n\n"
    buff += " Score guide:\n"
    buff += "   0.0-0.3 : good\n"
    buff += "   0.3-0.5 : ok\n"
    buff += "   0.5-1.0 : bad\n"
    buff += "\n   Score Matrix:\n"
    buff += " 0 - identical, 1 - no overlap\n\n"

    for i, row in enumerate(score_matrix_report):
        if i == 0:
            buff += "" + " ".join(f"{name:>5}" for name in row) + "\n" # Header row
        else:
            buff += f"{row[0]:>5} " + " ".join(f"{float(score):>5.2f}" for score in row[1:]) + "\n" # First element is the atom name, rest are scores

    off_diag = score_matrix[~np.eye(score_matrix.shape[0], dtype=bool)] # Exclude diagonal (0) for mean and std calculation
    sem = np.std(off_diag, ddof=1) / np.sqrt(len(off_diag)) # Standard error of the mean
    printable = buff.format(molname=molname,
                            mean=np.mean(off_diag),
                            sem=sem,
                            max=np.max(score_matrix))
    print(printable)
    with open('report_distances.out', 'w') as fout:
        fout.writelines(printable)

def report_sasa(h_score_dict, score_dict, mean_dict):
    """
    Print out a report of the solvent accessible surface area (SASA) scores.

    Parameters
    ----------
    h_score_dict : dict
        Hellinger distance of the SASA distribution of every molecule.
    score_dict : dict
        Score of the SASA distribution of every molecule.
    mean_dict : dict
        (mean CG SASA, mean reference SASA) of every molecule, nm^2.
    """
    buff = ''
    buff += " [ Solvent Accessible Surface Area (SASA) Report ]\n"
    buff += f"   Overall Score : {{mean:.3}} ± {{sem:.3}}\n\n"

    buff += " SASA Scores:\n"
    buff += " 0 - identical, 1 - no overlap\n\n"

    buff += " Score guide:\n"
    buff += "   0.0-0.3 : good\n"
    buff += "   0.3-0.5 : ok\n"
    buff += "   0.5-1.0 : bad\n\n"
    buff += "\t\tHellinger distance (Scoring function)   <SASA> CG / reference [nm^2]\n\n"

    overall_scores = []
    for molname, h_score in h_score_dict.items():
        score = score_dict[molname]
        cg_mean, ref_mean = mean_dict[molname]
        buff += f"\t{molname:20s}: {h_score:.2f} ({score:.2f})   {cg_mean:.2f} / {ref_mean:.2f}\n"
        overall_scores.append(score)

    printable = buff.format(mean=np.array(overall_scores).mean(),
                            sem=np.array(overall_scores).std()/np.sqrt(len(overall_scores)))
    print(printable)
    with open('report_sasa.out', 'w') as fout:
        fout.writelines(printable)