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
RECOGNISED_INTERACTIONS = list(INTERACTIONS.keys())

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
        probs, edges = np.histogram(time_series, density=True, bins=INTERACTIONS[inter_type].get('bins'))
        center_points = edges[:-1] + np.diff(edges)/2.
        distr = np.transpose((center_points, probs))
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
