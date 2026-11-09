import sys
import numpy as np
from fast_forward.bonded_functions import NORMAL_FUNCS

INTERACTIONS = {'bonds': {'bins': np.arange(0, 7, 0.01)
                          },
                'constraints': {'bins': np.arange(0, 7, 0.01)
                          },
                'angles': {'bins': np.arange(181)
                           },
                'dihedrals': {'bins': np.arange(-180, 181)
                              },
                'distances': {'bins': np.arange(0, 30, 0.05) # might need adjustment to allow for larger molecules
                              },
                'sasa': {'bins': np.arange(0, 100, 0.01) # nm^2, solvent accessible surface area per molecule
                         },
                'virtual_sitesn': {'interaction_types': {'1': 'virtual_sitesn'},
                                   'array_shape': {'virtual_sitesn': 1}
                                   },
                'virtual_sites2': {'interaction_types': {'1': 'virtual_sites2',
                                                         '2': 'virtual_sites2fd'},
                                   'array_shape': {'virtual_sites2': 1,
                                                   'virtual_sites2fd': 1}
                                   },
                'virtual_sites3': {'interaction_types': {'1': 'virtual_sites3',
                                                         '2': 'virtual_sites3fd',
                                                         '4': 'virtual_sites3out'},
                                   'array_shape': {'virtual_sites3': 2,
                                                   'virtual_sites3fd': 2,
                                                   'virtual_sites3out': 3}
                                   }
                }
VS_TYPES = [key for key in INTERACTIONS.keys() if 'virtual' in key]
# sasa is a property of whole molecules, not a bonded interaction
RECOGNISED_INTERACTIONS = [key for key in INTERACTIONS.keys() if key != 'sasa']

def distribution(series, inter_type, name=""):
    """
    Distribution of the values of an interaction (or of the SASA) in the bins
    defined for its type in :data:`INTERACTIONS`, which are also used for scoring.

    Values outside of the bins are not counted; a warning reports how many.

    Parameters
    ----------
    series : np.ndarray
        values of the interaction, e.g. its time series
    inter_type : str
        interaction type, a key of :data:`INTERACTIONS` with bins
    name : str
        name of the interaction group or molecule, for the warning

    Returns
    -------
    np.ndarray
        (n_bins, 2) columns: bin centre, probability density
    """
    bins = INTERACTIONS[inter_type]['bins']
    series = np.asarray(series)
    outside = np.count_nonzero((series < bins[0]) | (series > bins[-1]))
    if outside:
        label = f"{name} {inter_type}" if name else inter_type
        print(f"WARNING - {outside} of {len(series)} values of {label} lie outside of the histogram range "
              f"({bins[0]:g} to {bins[-1]:g}) and are ignored in the distribution.")
    probs, edges = np.histogram(series, density=True, bins=bins)
    center_points = edges[:-1] + np.diff(edges) / 2.
    return np.transpose((center_points, probs))

def interaction_distribution(u, inter_name, pair_idxs, group_name="", prefix="", save=None, parameters=[]):
    # i.e. if inter_type not one of virtual_sitesn, virtual_sites3fd, etc.
    if inter_name not in VS_TYPES:
        inter_type = inter_name
        nframes = u.trajectory.n_frames
        time_series = np.zeros((len(pair_idxs) * nframes))
        for idx, idxs in enumerate(pair_idxs):
            pair_pos = [u.trajectory.coordinate_array[:, pair, :] for pair in idxs]
            time_series[idx*nframes:(idx+1)*nframes] = NORMAL_FUNCS[inter_name](*pair_pos)

        if save:
            np.savetxt("{dir}/{prefix}{name}_{inter_type}.dat".format(dir=save,
                                                                name=group_name,
                                                                inter_type=inter_type,
                                                                prefix=prefix),
                       time_series)
        distr = distribution(time_series, inter_type, group_name)
        if save:
            np.savetxt("{dir}/{prefix}{name}_{inter_type}_distr.dat".format(dir=save,
                                                                      name=group_name,
                                                                      inter_type=inter_type,
                                                                      prefix=prefix),
                       distr)
        return distr, inter_type

    else:
        inter_type = INTERACTIONS[inter_name].get('interaction_types').get(parameters[0], None)
        if inter_type is not None:
            vs_fitted = np.zeros((len(pair_idxs), int(INTERACTIONS[inter_name].get('array_shape').get(inter_type))))
            for idx, idxs in enumerate(pair_idxs):
                pair_pos = [u.trajectory.coordinate_array[:, pair, :] for pair in idxs]
                vs_fitted[idx] = NORMAL_FUNCS[inter_type](*pair_pos)
            if save:
                np.savetxt("{prefix}{name}_{inter_type}.dat".format(name=group_name,
                                                                    inter_type=inter_type,
                                                                    prefix=prefix),
                           vs_fitted)
            return vs_fitted, inter_type
        else:
            raise NotImplementedError(f'Virtual site interaction type specified ({inter_name} itp type {parameters[0]})'
                                      f' for interaction group {group_name} is not currently implemented. '
                                      f'Please reconsider topology.')
