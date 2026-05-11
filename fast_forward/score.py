import numpy as np
from fast_forward.interaction_distribution import INTERACTIONS, interaction_distribution
from fast_forward.itp_to_ag import find_mol_indices
from collections import defaultdict

def hellinger(p,q):
    return np.round(np.sqrt(np.sum(np.power((np.sqrt(p) - np.sqrt(q)),2))) / np.sqrt(2),2)

def std_dev_hist(hist, bins):
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return np.sqrt(np.cov(bin_centers, aweights=hist, bias=True))

def calc_score(ref, test, weights=[0.7, 0.3], interaction_type='distances'):
    '''
    Compute the score between two distributions.
    The score is a weighted sum of the Hellinger distance and the difference in means of the distributions
    normalized by the standard deviation of the reference distribution. If mean is outside of the standard deviation
    of the reference distribution, it is capped at 1.

    ----------
    ref : numpy 1D-array
        Reference distribution.
    test : numpy 1D-array
        Test distribution.  
    bins : numpy 1D-array
        Bins for the distributions, used to calculate the mean difference.

    Returns
    -------
    float
        Score between the two distributions, between [0, 1].
    '''
    bins=INTERACTIONS[interaction_type]['bins']
    ref = ref / np.sum(ref) if np.sum(ref) > 0 else ref # normalize distributions
    test = test / np.sum(test) if np.sum(test) > 0 else test

    bin_centers = (bins[:-1] + bins[1:]) / 2

    mean_diff = np.average(bin_centers, weights=ref) - np.average(bin_centers, weights=test)
    mean_diff_norm = mean_diff / (2 * std_dev_hist(ref, bins)) # normalize mean difference by standard deviation of reference distribution
    mean_diff_norm = np.min([np.abs(mean_diff), 1]) # 1 as maximum penalty

    score = hellinger(ref, test) * weights[0] + mean_diff_norm * weights[1] # score is a weighted sum of Hellinger distance and mean difference normalized by standard deviation
    return np.round(score, 2)

def score_matrix(molname, block, universe, distribution_files, hellinger_weight=0.7, include_constrains=False):
    """
    Calculate the score matrix for all pairwise distances in the molecule block.

    Parameters
    ----------
    molname : str
        Name of the molecule.
    block : vermouth.molecule.Block
        Block containing the molecule information.
    universe : MDAnalysis.Universe
        Universe containing the trajectory data.
    distribution_files : list of str
        List of file paths to the distribution data files.
        These files should contain the reference distributions for the pairwise distances.
    """

    plot_data = defaultdict(dict)
    natoms = len(block.nodes)
    score_matrix = np.zeros((natoms, natoms))

    constraints = []
    for constraint in block.interactions['constraints']:
        constraints.append(set(constraint.atoms))

    for node1, name1 in block.nodes(data='atomname'):
        for node2, name2 in list(block.nodes(data='atomname'))[node1+1:]:
            atoms = np.array([node1, node2])
            group_name = f'{name1}_{name2}' # following the naming convention introduced in ITPInteractionMapper
            indices = find_mol_indices(universe,
                            atoms,
                            molname)
            distr = interaction_distribution(universe, 'distances', indices)
            # calculate simulation distribution
            probs = distr[0].T[1]
            # read in reference distribution
            try:
                reference_data = np.loadtxt([i for i in distribution_files if group_name in i and 'distances' in i][0])
            except IndexError:
                print(f"{group_name} file not found!")
                continue
            
            # if the distance is constrained, the mean difference is weighted more
            if {node1, node2} in constraints and not include_constrains:
                weigths = [0.2,0.8] # can be adjusted in the future
            else:
                weigths = [hellinger_weight, 1-hellinger_weight]

            # calculate score and populate matrix
            score = calc_score(reference_data.T[1], probs, weigths, interaction_type='distances')
            score_matrix[node1, node2] = float(score)
            score_matrix[node2, node1] = float(score)
            plot_data['distances'][group_name] = {"x": reference_data.T[0],
                                                    "Reference": reference_data.T[1],
                                                    'Simulated': probs}
    return score_matrix, plot_data