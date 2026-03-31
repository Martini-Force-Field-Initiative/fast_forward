
'''
Functions for plotting fitted interaction distributions
'''

import matplotlib.pyplot as plt
import numpy as np
import pickle
from .interaction_distribution import INTERACTIONS

X_LABELS={'bonds': 'Distance [Å]',
          'angles': 'Angle [°]',
          'dihedrals': 'Angle [°]',
          'distances': 'Distance [Å]'}

def _plotter(data, atom_list, inter_type, ax):

    cols = ['#6970E0', '#E06B69']
    needed_keys = [key for key in list(data.keys()) if key != 'x']
    for idx, key in enumerate(needed_keys):
        ax.plot(data['x'],
                data[key],
                c=cols[idx],
                label=key)
    ax.legend()
    ax.set_title(f'{atom_list} {inter_type}')
    ax.set_xlim(INTERACTIONS[inter_type].get('bins')[0],
                INTERACTIONS[inter_type].get('bins')[-1])
    ax.set_xlabel(X_LABELS[inter_type])


def _plotter_distance_distribution(data, ax, y_lower_threshold: float = 0.01):
    """
    Plot distance distribution curves and automatically zoom in on
    regions where the signal exceeds a given fraction of its maximum.

    Parameters
    ----------
    data : dict
        Dictionary containing the distance distribution data.
        Must include a key ``"x"`` for the shared x-axis values.
        All other keys correspond to y-data series to be plotted and
        must be array-like and of the same length as ``data["x"]``.
    ax : :class:`matplotlib.axes._axes.Axes`
        Axes object on which the curves will be plotted.
    y_lower_threshold : float, optional
        Fraction of each curve's maximum used to determine the
        region of interest. Only x-values where ``y > y.max() * threshold``
        are considered when computing the zoomed x-axis limits.
        Default is ``0.01``
    """
    cols = ['#6970E0', '#E06B69']
    needed_keys = [key for key in list(data.keys()) if key != 'x']
    x_min = 100
    x_max = 0
    x_pad = 0.25

    for idx, key in enumerate(needed_keys):
        ax.plot(data['x'],
                data[key],
                c = cols[idx],
                label=key)
        threshold = np.max(data[key]) * y_lower_threshold
        significant_indices = np.where(data[key] > threshold)[0]

        if significant_indices.size > 0: # Only update if we found significant data
            x_min = np.min([x_min, data['x'][np.min(significant_indices)]])
            x_max = np.max([x_max, data['x'][np.max(significant_indices)]])
    ax.yaxis.set_ticks([])
    ax.set_xlim(x_min - x_pad, x_max + x_pad)

def make_distribution_plot(fit_data, save_plot_data=None, axarr=None, name='distribution_plots'):
    '''

    Parameters
    ----------
    fit_data: dict
        Dictionary containing distributions and fitting parameters for interactions.
        Nested as {interaction_type: {group_name: {'data': distribution, 'fitted_params': list(params)}}
    axarr: :class:`matplotlib.axes._axes.Axes`
        array of axes to plot the fitted distributions on
    save_plot_data: bool
        if True, save the underlying data for plots as a pickle file
    name: str
        name of the output file (default: distribution_plots)
    '''

    total_interactions = sum([len(fit_data[i]) for i in fit_data.keys()])

    if not axarr:
        ncols = 5
        nrows = -(total_interactions // -5) # upside-down floor division
        if nrows == 0:
            print("No interactions to plot!")
            return
        fig, axarr = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(ncols*4, nrows*4))

    count = 0
    for interaction_type in fit_data.keys():
        for atom_list in fit_data[interaction_type].keys():
            _plotter(fit_data[interaction_type][atom_list], atom_list, interaction_type, axarr.flatten()[count])
            count += 1

    # set ylabel
    if len(axarr.shape) == 2:
        for ax in axarr[:,0]:
            ax.set_ylabel('Probability Density')
    else:
        axarr[0].set_ylabel('Probability Density')

    if save_plot_data:
        pickle.dump(fit_data, open('plot_data.p', 'wb'))

    # remove unused axes
    for ax in axarr.flatten()[count:]:
        fig.delaxes(ax)

    # need to make room for the title
    fig.subplots_adjust(hspace = 0.3)

    fig.savefig(f'{name}.png', bbox_inches='tight')

def make_matrix_plot(matrix, atom_names, axarr=None, name='score_matrix'):
    '''

    Parameters
    ----------
    matrix: :class:`~numpy.ndarray`
        Quatratic 2D array representing the matrix
    atom_names: list
        List of atom names corresponding to the rows and columns of the matrix
    axarr: :class:`matplotlib.axes._axes.Axes`
        array of axes to plot the fitted distributions on
    name: str
        name of the output file (default: distribution_plots)
    '''
    if not axarr:
        nrows = 1
        ncols = 1
        fig, axarr = plt.subplots(nrows=nrows, ncols=ncols,
                               figsize=(ncols*5+1, nrows*5))
    cax = axarr.imshow(matrix, cmap='coolwarm', vmin=0, vmax=1)
    axarr.set_xticks(np.arange(len(atom_names)), labels=atom_names)
    axarr.set_yticks(np.arange(len(atom_names)), labels=atom_names)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = axarr.text(j, i, f'{matrix[i, j]:.2f}',
                        ha="center", va="center", color="w", fontsize=8)
    cbar = plt.colorbar(cax)
    cbar.set_label('Score', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=10)
    axarr.set_title('Distance Score Matrix')

    # need to make room for the title
    fig.subplots_adjust(hspace = 0.3)

    fig.savefig(f'{name}.png', bbox_inches='tight')

def make_distances_distribution_plot(plot_data, atom_names, save_plot_data=False, axarr=None, name='distance_distribution_plots'):
    '''

    Parameters
    ----------
    matrix: :class:`~numpy.ndarray`
        Quatratic 2D array representing the matrix
    atom_names: list
        List of atom names corresponding to the rows and columns of the matrix
    axarr: :class:`matplotlib.axes._axes.Axes`
        array of axes to plot the fitted distributions on
    name: str
        name of the output file (default: distribution_plots)
    '''

    natoms = len(atom_names)
    if not axarr:
        fig ,axarr = plt.subplots(natoms-1,natoms-1,figsize=(natoms*2,natoms),gridspec_kw={'wspace':0.05,'hspace':0.4})

    parsed_names = [name.split("_") for name in atom_names]
    for i in range(natoms-1):
        resid1, name1 = parsed_names[i]
        for j in range(1,natoms):
            resid2, name2 = parsed_names[j]
            ax = axarr[i, j-1]
            if i < j: # plot only upper triangle of the matrix
                atoms_key = f'{resid1}_{name1}_{resid2}_{name2}'
                if atoms_key in plot_data['distances']:
                    _plotter_distance_distribution(plot_data['distances'][atoms_key], ax)
            else:
                fig.delaxes(ax) # remove lower triangle of the matrix
    
    # add labels to the axes
    for i in range(natoms-1):
        axarr[0, i].set_title(atom_names[i+1], fontsize=14)
        axarr[i, i].set_ylabel(atom_names[i], fontsize=14)
        axarr[i, i].set_xlabel(X_LABELS['distances'])

    # add legend next to last plot
    axarr[natoms-2, natoms-2].legend(loc='upper left', fontsize=10, bbox_to_anchor=(-1, 0.75), frameon=False)
    
    fig.suptitle('Distance Distribution Plots', fontsize=16)

    if save_plot_data:
        pickle.dump(plot_data, open('plot_data_distribution.p', 'wb'))

    fig.savefig(f'{name}.png', bbox_inches='tight')