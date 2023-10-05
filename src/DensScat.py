import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import pandas as pd 


def smooth1D(Xin, lambda_val):
    N = Xin.shape[0]
    E = np.eye(N)
    D1 = np.diff(E, axis=0)
    D2 = np.diff(D1, axis=0)
    P = (lambda_val**2) * D2.T @ D2 + 2 * lambda_val * D1.T @ D1
    Xout = np.linalg.solve(E + P, Xin)
    return Xout

def DensScat(x, y, title, marker_type='.', m_size=50, color_map='viridis', 
             log_density=True, axis_type='square', smooth_density=True, 
             lambda_val=30, nbin_x=200, nbin_y=200, remove_points=True, 
             target_axes=None, color_bar=True, max_dens=np.inf, 
             points_to_exclude=None):
    
    # Check inputs
    if len(x) != len(y):
        raise ValueError("x & y must all have equal length")
    
    # Remove NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    
    # Remove points to exclude if provided
    if points_to_exclude is not None:
        mask = np.isin(np.vstack([x, y]).T, points_to_exclude, invert=True).all(axis=1)
        x = x[mask]
        y = y[mask]
    
    # Define edges
    edges_x = np.linspace(min(x), max(x), nbin_x+1)
    edges_x[[0, -1]] = [-np.inf, np.inf]
    edges_y = np.linspace(min(y), max(y), nbin_y+1)
    edges_y[[0, -1]] = [-np.inf, np.inf]
    
    # Get number of counts
    H, _, _ = np.histogram2d(x, y, bins=[edges_x, edges_y])
    
    # Smoothing
    if smooth_density:
        H = smooth1D(H, nbin_y/lambda_val)
        H = smooth1D(H.T, nbin_x/lambda_val).T
    
    # Get density for each point
    bin_x = np.digitize(x, edges_x) - 1
    bin_y = np.digitize(y, edges_y) - 1
    density = H[bin_y, bin_x]
    
    if log_density:
        density = np.log10(density + 1)
    
    if max_dens:
        density[density > max_dens] = max_dens
    
    # Make sure that high density points are plotted last
    sort_idx = np.argsort(density)
    x = x[sort_idx]
    y = y[sort_idx]
    density = density[sort_idx]
    
    if remove_points:
        _, unique_idx = np.unique(np.round(np.vstack([x, y]), 3), axis=1, return_index=True)
        x = x[unique_idx]
        y = y[unique_idx]
        density = density[unique_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8,8))
    scatter = ax.scatter(x, y, c=density, cmap=color_map, s=m_size, marker=marker_type)
    
    # Add regression line
    coeffs = np.polyfit(x, y, deg=1)
    reg_line = np.polyval(coeffs, x)
    ax.plot(x, reg_line, color='red', label=f'y={coeffs[0]:.2f}x + {coeffs[1]:.2f}')
    
    if axis_type == 'square':
        ax.axis('square')
    elif axis_type == 'equal':
        ax.axis('equal')
    
    if color_bar:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Density')
    
    # Calculate RMSE, MAE, and Pearson correlation
    rmse = math.sqrt(mean_squared_error(y, reg_line))
    mae = mean_absolute_error(y, reg_line)
    pearson_corr, _ = pearsonr(x, y)
    
    # Add text box with RMSE, MAE, and Pearson correlation
    textstr = f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nPearson: {pearson_corr:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
    plt.xlabel('OBS')
    plt.ylabel('Waverys')
    plt.title(f'{title}')
    plt.legend(loc='upper right')
    plt.grid(True)
    #plt.show()
    plt.savefig(f'../figures/densscat_{title}.png')
    
    return fig, ax



df_riogrande = pd.read_csv('../data/df_total_riogrande_processed.csv', sep=';', index_col=0)
df_santos = pd.read_csv('../data/df_total_santos_processed.csv', sep=';', index_col=0)
df_vitoria = pd.read_csv('../data/df_total_vitoria_processed.csv', sep=';', index_col=0)


df_clean_riogrande = df_riogrande.dropna(subset=['OBS', 'Waverys'])
df_clean_santos = df_santos.dropna(subset=['OBS', 'Waverys'])
df_clean_vitoria = df_vitoria.dropna(subset=['OBS', 'Waverys'])


DensScat(df_clean_riogrande['OBS'], df_clean_riogrande['Waverys'], title='Rio Grande')
DensScat(df_clean_santos['OBS'], df_clean_santos['Waverys'], title='Santos')
DensScat(df_clean_vitoria['OBS'], df_clean_vitoria['Waverys'], title='Vitória')

