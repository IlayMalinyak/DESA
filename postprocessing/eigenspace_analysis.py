import numpy as np
import pandas as pd
import umap
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, adjusted_rand_score
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
import matplotlib as mlp
from sklearn.neural_network import MLPRegressor

from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from matplotlib.font_manager import FontProperties
import seaborn as sns

from scipy.spatial import procrustes # Correct import for procrustes

lamost_catalog_path = r"C:\Users\Ilay\projects\kepler\data\lightPred\tables\lamost_afgkm_teff_3000_7500_catalog.csv"
lightpred_catalog_path = r"C:\Users\Ilay\projects\kepler\data\lightPred\tables\kepler_predictions_clean_seg_0_1_2_median.csv"
lightpred_full_catalog_path = r"C:\Users\Ilay\projects\kepler\data\binaries\tables\lightpred_cat.csv"
godoy_catalog_path = r"C:\Users\Ilay\projects\kepler\data\binaries\tables\GodoyRivera25_TableA1.csv"
kepler_gaia_nss_path = r"C:\Users\Ilay\projects\kepler\data\binaries\tables\gaia_nss.csv"
kepler_gaia_wide_path = r"C:\Users\Ilay\projects\kepler\data\binaries\tables\gaia_wide.csv"
ebs_path =  r"C:\Users\Ilay\projects\kepler\data\binaries\tables\ebs.csv"
berger_catalog_path = r"C:\Users\Ilay\projects\kepler\data\lightPred\tables\berger_catalog_full.csv"
lamost_kepler_path = r"C:\Users\Ilay\projects\kepler\data\lightPred\tables\lamost_dr8_gaia_dr3_kepler_ids.csv"
lamost_apogee_path = r"C:\Users\Ilay\projects\kepler\data\apogee\crossmatched_catalog_LAMOST.csv"
mist_path = r"C:\Users\Ilay\projects\kepler\data\binaries\tables\ks_mist_catalog_kepler_1e9_feh_interp_all.csv"
kepler_meta_path = r"C:\Users\Ilay\projects\kepler\data\lightPred\tables\kepler_dr25_meta_data.csv"

units = {'M_G_0': 'mag' ,'BPmRP_0': 'mag', 'Teff' :'K', 'age_gyrointerp_model': 'Myr', 'age_angus23': 'Myr',
         'kmag_abs': 'mag', 'kmag_diff': 'mag', 'predicted period': 'Days',
         'FeH': 'dex', 'i': 'Deg', 'i_err': 'Deg', 'final_age_norm': 'Gyr',
         'age_error_norm': 'Gyr'}


latex_names = {'M_G_0': 'Absolute $M_G$' ,'BPmRP_0': '$(BP-RP)$', 'Teff' :'$T_{eff}$',
               'Lstar': r'$log(\frac{L}{L_\odot})$',
               'Mstar': r'$\frac{M}{M_\odot}$',
               'Rstar':r'$\frac{R}{R_\odot}$',
               'age_gyrointerp_model': '$Age_{gyro}$',
               'age_angus23': '$Age_{iso}$',
               'kmag_abs': 'Absolute $M_K$',
               'kmag_diff': r'$\Delta K_{iso}$',
               'RUWE': 'RUWE',
               'predicted period': '$P_{rot}$',
               'FeH': '$FeH$',
               'flag_CMD_numeric': 'CMD Category',
               'i': 'inclination',
               'i_err': 'inclination error',
               'final_age_norm': 'Age',
               'age_error_norm': 'Age Error',
               'moco': 'MoCo',
               'moco_uniqueLoader': 'MoCo Clean',
               'jepa': 'VicReg',
               'unimodal_light': 'Only LC',
               'unimodal_spec': 'Only Spec',
               'dual_former': 'DESA (ours)',
               'simsiam': 'SimSiam'}

# --- Helper Function (from your example) ---
def giant_cond(row):
    # This is just a placeholder based on your usage, replace with actual logic if needed
    # Example: Assuming 'logg' column exists and main sequence is logg > 4.0
    if pd.isna(row['logg']):
        return False # Exclude if logg is missing
    return row['logg'] > 4.0

# --- 1. Data Loading and Preparation ---

def get_mag_data(df):
    kepler_meta = pd.read_csv(kepler_meta_path)
    mist_data = pd.read_csv(mist_path)
    mag_cols = [c for c in kepler_meta.columns if ('MAG' in c) or ('COLOR' in c)]
    meta_columns = mag_cols + ['KID', 'EBMINUSV']
    df = df.merge(kepler_meta[meta_columns], on='KID', how='left')
    df = df.merge(mist_data[['KID', 'Kmag_MIST']], on='KID', how='left')
    if 'Dist' not in df.columns:
        berger_cat = pd.read_csv(berger_catalog_path)
        df = df.merge(berger_cat, on='KID', how='left',
                      suffixes=['_old', ''])
    for c in mag_cols:
        df[f'{c}_abs'] = df.apply(lambda x: x[c] - 5 * np.log10(x['Dist']) + 5, axis=1)
    df['kmag_abs'] = df.apply(lambda x: x['KMAG'] - 5 * np.log10(x['Dist']) + 5, axis=1)
    df['kmag_diff'] = df['kmag_abs'] - df['Kmag_MIST']
    return df

def prepare_data(dir_name, file_name, umap_input='projections',
                 calc_umap=True, filter_CMD=False):
    """Loads projections, merges catalogs, filters data, and prepares inputs."""

    print(f"--- Preparing data for {file_name} ---")

    # Load base data
    proj_path = os.path.join(dir_name, f'embedding_projections_{file_name}.npy')
    projections = np.load(proj_path)
    print(f"Loaded projections with shape: {projections.shape}")
    print("projections median std: ", np.median(projections.std(axis=-1)))

    features_path = os.path.join(dir_name, f'final_features_{file_name}.npy')
    features = np.load(features_path)
    print(f"Loaded features with shape: {features.shape}")
    print("features median std: ", np.median(features.std(axis=-1)))

    # Load main prediction file which should correspond to projections
    preds_path = os.path.join(dir_name, f'preds_{file_name}.csv')
    df = pd.read_csv(preds_path)

    # Load and merge catalogs
    berger = pd.read_csv(berger_catalog_path)
    lightpred = pd.read_csv(lightpred_full_catalog_path)
    godoy = pd.read_csv(godoy_catalog_path)

    df = df.merge(berger, right_on='KID', left_on='kid', how='left')
    df = get_mag_data(df)
    age_cols = [c for c in lightpred.columns if 'age' in c] # Robustly find age columns
    df = df.merge(lightpred[['KID', 'predicted period', 'mean_period_confidence'] + age_cols], right_on='KID', left_on='kid', how='left')
    df = df.merge(godoy, right_on='KIC', left_on='kid', how='left', suffixes=['', '_godoy'])


    if filter_CMD:
        indices = df[~((df['flag_CMD'].isna()) | (df['flag_CMD'] == 'notinCMDsample'))].index
        features = features[indices]
        projections = projections[indices]
        df = df.loc[indices]
        df = df.reset_index(drop=True)
    df['subgiant'] = (df['flag_CMD'] == 'Subgiant').astype(int)
    df['main_seq'] = df.apply(giant_cond, axis=1)
    codes, uniques = pd.factorize(df['flag_CMD'])  # No na_sentinel parameter here
    df['flag_CMD_numeric'] = codes
    mapping_dict = {i: label for i, label in enumerate(uniques)}
    mapping_dict[-1] = np.nan

    df['duplicates'] = df.groupby('kid').apply(
        lambda x: pd.Series([x.index.tolist()] * len(x), index=x.index)
    ).droplevel(0)

    # Remove self-index from each list
    df['duplicates'] = df.apply(
        lambda row: [idx for idx in row['duplicates'] if idx != row.name], axis=1
    )

    if calc_umap:
        # --- Generate Standard UMAP Coordinates (U_standard) ---
        print("Calculating standard UMAP...")
        reducer_standard = umap.UMAP(n_components=2, random_state=42) # Added random_state for reproducibility
        if umap_input == 'projections':
            U_standard = reducer_standard.fit_transform(projections)
        elif umap_input == 'features':
            U_standard = reducer_standard.fit_transform(features)
        else:
            raise ValueError(f'Invalid umap_input: {umap_input}')
        print(f"Calculated U_standard with shape: {U_standard.shape}")
    else:
        U_standard = None
    return projections,features, U_standard, df, mapping_dict

def plot_umap_with_pairs(dir_name, file_name, num_samples=10, umap_input='projections',):
    projections, features, umap_coords, df, cmd_mapping_dict = prepare_data(dir_name, file_name,
                                                                            umap_input=umap_input,
                                                                            filter_CMD=True )
    model_name = os.path.basename(dir_name)
    x_coords, y_coords = umap_coords[:, 0], umap_coords[:, 1]
    # Set up the plot
    fig, ax = plt.subplots()

    # Plot all points first
    ax.scatter(x_coords, y_coords, c='lightgray', alpha=0.5, s=20, label='Unique samples')

    # Create a color palette
    n_duplicate_groups = len(df[df['duplicates'].apply(len) > 0]['kid'].unique())
    colors = sns.color_palette("tab10", num_samples)

    # Plot duplicates with different colors for each 'kid' group
    color_idx = 0
    for kid in df['kid'].unique():
        kid_indices = df[df['kid'] == kid].index.tolist()

        if len(kid_indices) > 1:  # Only highlight if there are duplicates
            ax.scatter(x_coords[kid_indices], y_coords[kid_indices],
                       c=[colors[color_idx]], s=80, alpha=0.8,
                       edgecolors='black', linewidth=0.5,
                       label=f'Duplicates: kid={kid}')
            color_idx += 1
            if color_idx == num_samples:
                break

    ax.set_xlabel('UMAP X')
    ax.set_ylabel('UMAP Y')
    ax.set_title(model_name)
    # ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, f'umap_duplicates_{model_name}.png'))
    plt.show()

    # Calculate distances between duplicates
    duplicate_distances = []
    duplicate_pairs = []
    duplicate_teff = []

    for idx, row in df.iterrows():
        if len(row['duplicates']) > 0:
            current_coords = np.array([x_coords[idx], y_coords[idx]])
            for dup_idx in row['duplicates']:
                if idx < dup_idx:  # Avoid double counting pairs
                    dup_coords = np.array([x_coords[dup_idx], y_coords[dup_idx]])
                    distance = np.linalg.norm(current_coords - dup_coords)
                    duplicate_distances.append(distance)
                    duplicate_pairs.append((idx, dup_idx))
                    duplicate_teff.append(row['Teff'])
    plt.scatter(duplicate_teff, duplicate_distances)
    plt.xlabel('$T_{eff}$ (K)')
    plt.ylabel('Pair Distance (AU)')
    plt.savefig(os.path.join(dir_name, f'duplicate_distances_teff.png'))
    plt.show()

def compare_pair_distances(root_dir):
    dirs = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
    for dir in dirs:
        umap_input = 'projections' if 'DESA' in dir else 'features'
        umap_dir = os.path.join(root_dir, dir)
        for file in os.listdir(umap_dir):
            if file.endswith('.csv'):
                file_name = file.replace('preds_', '').replace('.csv', '')
                projections, features, umap_coords, df, cmd_mapping_dict = prepare_data(umap_dir, file_name,
                                                                                        umap_input=umap_input,
                                                                                        filter_CMD=True)
                x_coords, y_coords = umap_coords[:, 0], umap_coords[:, 1]
                duplicate_distances = []
                duplicate_pairs = []

                for idx, row in df.iterrows():
                    if len(row['duplicates']) > 0:
                        current_coords = np.array([x_coords[idx], y_coords[idx]])
                        for dup_idx in row['duplicates']:
                            if idx < dup_idx:  # Avoid double counting pairs
                                dup_coords = np.array([x_coords[dup_idx], y_coords[dup_idx]])
                                distance = np.linalg.norm(current_coords - dup_coords)
                                duplicate_distances.append(np.log(distance))
                                duplicate_pairs.append((idx, dup_idx))
                print(root_dir, "avg/std distance: ", np.mean(duplicate_distances), np.std(duplicate_distances))
                label_name = f'{dir} ({np.mean(duplicate_distances):.2f}, {np.std(duplicate_distances):.2f})'
                plt.hist(duplicate_distances, histtype='step',
                         density=True, bins=20, label=label_name, linewidth=4)
    plt.legend()
    plt.xlabel('log(Pair Distance) (UMAP Coords)')
    plt.ylabel('PDF')
    plt.savefig(os.path.join(root_dir, 'duplicate_distances_hist.png'))
    plt.show()



def create_hr_coords(df, projections, diagram_coords):
    # Extract and clean HR coordinates
    hr_coords = df[diagram_coords].copy()
    hr_coords = hr_coords.replace([np.inf, -np.inf], np.nan)

    # Find valid (non-NaN) indices
    valid_mask = hr_coords.notna().all(axis=1)
    initial_count = len(hr_coords)
    print(f"Removed {initial_count - valid_mask.sum()} rows with NaN")

    # Clean both Y_hr and corresponding X (projections)
    Y_hr = hr_coords.loc[valid_mask].values
    X = projections[valid_mask.values]

    print(f"Prepared Y_hr (HR coordinates) with shape: {Y_hr.shape}")
    print(f"Prepared X (projections) with shape: {X.shape}")
    return Y_hr, X

# --- 2. Transformation Methods ---

def apply_linear_regression(X, Y, fraction_scale=0.2, name=''):
    """Applies Linear Regression to predict Y from X."""
    # print("\n--- Applying Linear Regression ---")
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y)

    x_indices = np.random.choice(len(X_scaled), size=int(len(X_scaled)*fraction_scale), replace=False)
    mask = np.ones(len(X_scaled), dtype=bool)
    mask[x_indices] = False
    if mask.sum() == 0:
        mask = np.ones(len(X_scaled), dtype=bool)
    X_train_scaled = X_scaled[x_indices, :]
    X_test_scaled = X_scaled[mask, :]
    y_train_scaled = Y_scaled[x_indices, :]
    y_test_scaled = Y_scaled[mask, :]

    print("Linear Regression is training on ", len(X_train_scaled), " points and test on ", len(X_test_scaled))

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train_scaled)
    score = lr.score(X_test_scaled, y_test_scaled)

    Y_pred_scaled = lr.predict(X_scaled)
    # Inverse transform to get predictions in original Temp/LogLum scale
    Y_pred_lr = scaler_Y.inverse_transform(Y_pred_scaled)
    Y_lr = scaler_Y.inverse_transform(Y_scaled)
    acc = np.mean(np.abs(Y_lr - Y_pred_lr) < Y_lr * 0.1, axis=0)
    loss = np.abs(Y_lr - Y_pred_lr).mean(axis=0)
    # print("acc: ", acc, "loss: ", loss)

    # fig, axes = plt.subplots(nrows=2, ncols=1)
    # axes[0].hexbin(Y_pred_lr[:, 0], Y_scaled[:, 0], mincnt=1)
    # axes[1].hexbin(Y_pred_lr[:, 1], Y_scaled[:, 1], mincnt=1)
    # fig.suptitle(f"{name} accuracy: {acc[0]:.3f}, {acc[1]:.3f}")
    #
    # plt.show()

    # print(f"Generated Linear Regression predictions shape: {Y_pred_lr.shape}, score: {score}")
    return Y_pred_lr, score, acc, loss


def plot_umap(df, umap_coords, cmd_mapping_dict, dir_name, file_name):
    # df.loc[df['kmag_diff'].abs() > 2, 'kmag_diff'] = np.nan
    cols_to_plot = ['Lstar', 'flag_CMD_numeric', 'age_gyrointerp_model', 'RUWE']
    # cols_to_plot = ['Lstar', 'flag_CMD_numeric', 'age_gyrointerp_model', 'RUWE', 'FeH', 'predicted period', 'kmag_diff', 'Teff']
    # df[df['kmag_diff'].abs() > 2] = np.nan
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(40, 24), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        label = latex_names[cols_to_plot[i]]
        unit = f'({units[cols_to_plot[i]]})' if cols_to_plot[i] in units else ''
        if ((cols_to_plot[i] == 'Teff') or (cols_to_plot[i] == 'predicted period') or
              (cols_to_plot[i]) == 'Rstar' or (cols_to_plot[i]) == 'Mstar' or (cols_to_plot[i] == 'RUWE')):
            color = np.log(df[cols_to_plot[i]])
            label = f'log({label})'
        else:
            color = df[cols_to_plot[i]]
        if cols_to_plot[i] == 'flag_CMD_numeric':
            # Define a colormap with distinct colors (excluding -1 for NaN values)
            # Using a tab20 colormap which provides 20 distinct colors
            cmap = plt.cm.get_cmap('hot', len(cmd_mapping_dict))

            # Initialize empty handles list for legend
            legend_handles = []

            # Plot each category with its own color and add to legend
            for value, name in cmd_mapping_dict.items():
                if value != -1:  # Skip NaN values (-1)
                    mask = df['flag_CMD_numeric'] == value
                    color = cmap(value % len(cmd_mapping_dict))  # Cycle through colors if needed
                    scatter = ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                                         c=[color], label=name)
                    legend_handles.append(scatter)

            # Plot NaN values with a distinct style (gray and smaller)
            mask_nan = df['flag_CMD_numeric'] == -1
            if mask_nan.any():
                nan_scatter = ax.scatter(umap_coords[mask_nan, 0], umap_coords[mask_nan, 1],
                                         c='lightgray', alpha=0.5, s=20, label='NaN')
                legend_handles.append(nan_scatter)

            # Add a legend to this subplot
            ax.legend(handles=legend_handles, title="CMD Classification",
                      loc='upper left', fontsize=26, title_fontsize=30)
        else:
            sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=color, cmap='OrRd')
            cbar = fig.colorbar(sc, orientation='vertical', label=f'{label} {unit}')
            cbar.ax.yaxis.set_tick_params(labelsize=50)  # Set colorbar tick label size
            cbar.set_label(f'{label} {unit}', fontsize=50)  # Set colorbar label size
    fig.supxlabel('UMAP X', fontsize=50)
    fig.supylabel('UMAP Y', fontsize=50)
    # fig.suptitle('UMAP of Eigenspace')
    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, f'umap_eigenspace_{file_name}.png'))
    plt.show()

    # binary_cols = ['flag_Binary_Union', 'flag_RUWE', 'flag_RVvariable', 'flag_NSS', 'flag_EB_Kepler',
    #                'flag_EB_Gaia', 'flag_SB9', 'kmag_diff']
    # fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(40, 24), sharex=True, sharey=True)
    # axes = axes.flatten()
    # for i, ax in enumerate(axes):
    #     if i < 7:
    #         color = df[binary_cols[i]]
    #         label = binary_cols[i]
    #         sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=color, cmap='OrRd')
    #         cbar = fig.colorbar(sc, orientation='vertical', label=label)
    #         cbar.ax.yaxis.set_tick_params(labelsize=40)  # Set colorbar tick label size
    #         cbar.set_label(label)  # Set colorbar label size
    # fig.supxlabel('UMAP X')
    # fig.supylabel('UMAP Y')
    # fig.suptitle('UMAP of Eigenspace')
    # plt.tight_layout()
    # plt.savefig(os.path.join(dir_name, f'umap_eigenspace_{file_name}_binaries.png'))
    # plt.show()


def plot_transformation(Y_hr, Y_pred_lr, score_lr, hr_coords_df,
                        dir_name, file_name, diagram_coords=['Teff', 'Lstar']):
    """Plots the results of all methods."""
    print("\n--- Plotting Results ---")
    # Create figure without axes first
    fig = plt.figure(figsize=(40, 24))

    # Plotting parameters
    # cmap = 'viridis'
    s = 20  # Scatter point size

    # Create a GridSpec layout with 2x2 grid
    gs = fig.add_gridspec(2, 2)

    # Create individual axes - with no sharing for the top row
    ax_top_left = fig.add_subplot(gs[0, 0])
    ax_top_right = fig.add_subplot(gs[0, 1])

    # Create bottom row axes with shared x and y axes
    # Note: We'll set up shared axes AFTER inversion if needed
    ax_bottom_left = fig.add_subplot(gs[1, 0])
    ax_bottom_right = fig.add_subplot(gs[1, 1])

    # Organize axes in a way that matches the original code's structure
    axes = [[ax_top_left, ax_top_right],
            [ax_bottom_left, ax_bottom_right]]

    # Now proceed with the plotting
    for row_idx in range(2):
        if row_idx == 0:
            for col_idx in range(2):
                ax = axes[row_idx][col_idx]
                sc = ax.scatter(Y_hr[:, col_idx], Y_pred_lr[:, col_idx], color='sandybrown', alpha=0.1)
                label = latex_names[diagram_coords[col_idx]]
                unit = f'({units[diagram_coords[col_idx]]})' if diagram_coords[col_idx] in units else ''
                ax.set_xlabel(f'True {label} {unit}', fontsize=50)
                ax.set_ylabel(f'Predicted {label} {unit}', fontsize=50)
                ax.set_aspect('equal')
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                min_val = min(xlim[0], ylim[0])
                max_val = max(xlim[1], ylim[1])
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
        else:
            label_0 = latex_names[diagram_coords[0]]
            unit_0 = f'({units[diagram_coords[0]]})' if diagram_coords[0] in units else ''
            label_1 = latex_names[diagram_coords[1]]
            unit_1 = f'({units[diagram_coords[1]]})' if diagram_coords[1] in units else ''  # Fixed bug here

            # a) Original HR Diagram
            ax = axes[row_idx][0]
            sc = ax.scatter(Y_hr[:, 0], Y_hr[:, 1], s=s, c='sandybrown')
            ax.set_xlabel(f'{label_0} {unit_0}', fontsize=50)
            ax.set_ylabel(f'{label_1} {unit_1}', fontsize=50)

            # Linear Regression Prediction
            ax2 = axes[row_idx][1]
            mae = np.abs(Y_hr - Y_pred_lr).mean(axis=-1)
            sc = ax2.scatter(Y_pred_lr[:, 0], Y_pred_lr[:, 1], s=s, c=np.log(mae), cmap='OrRd')
            ax2.set_xlabel(f"Predicted {label_0} {unit_0}", fontsize=50)
            ax2.set_ylabel(f"Predicted {label_1} {unit_1}", fontsize=50)


            # Apply inversions AFTER plotting but BEFORE linking axes
            if diagram_coords[0] == 'Teff':
                ax.invert_xaxis()  # Standard HR diagram convention
                ax2.invert_xaxis()  # Match original HR convention
            if diagram_coords[1] == 'logg':
                ax.invert_yaxis()
                ax2.invert_yaxis()

            # NOW link the axes for sharing limits
            ax2.sharex(ax)
            ax2.sharey(ax)

            # Add colorbar
            cbar = fig.colorbar(sc, ax=ax2)
            cbar.ax.yaxis.set_tick_params()
            cbar.set_label('log(MAE)')

    # Remove tick labels from the right subplot for cleaner appearance
    # Only show y-axis labels on the left plot for bottom row
    axes[1][1].tick_params(labelleft=False)

    # Make sure both plots share the same view limits after inversion
    axes[1][0].set_xlim(axes[1][0].get_xlim())
    axes[1][0].set_ylim(axes[1][0].get_ylim())
    fig.suptitle(f'$R^2$ - {score_lr:.4f}')

    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to prevent title overlap
    save_path = os.path.join(dir_name, f'hr_transformations_{file_name}_{diagram_coords[0]}_{diagram_coords[1]}.png')
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")
    plt.show()
def transform_eigenspace(df, projections, dir_name, file_name, diagram_coords=['Teff', 'Lstar']):
    # 1. Prepare data
    Y_hr, X = create_hr_coords(df, projections, diagram_coords)
    # 2. Apply transformations
    Y_pred_lr, score, acc, loss = apply_linear_regression(X, Y_hr, name=f'{diagram_coords[0]}, {diagram_coords[1]}')
    # 3. Plot results
    plot_transformation(Y_hr, Y_pred_lr, score, df, dir_name, file_name, diagram_coords)


def combined_projection_analysis(df, dir_name, target_coords=['i', 'i_err']):
    files = os.listdir(dir_name)
    projections = []
    kids = []
    for file in files:  # Changed variable name to avoid shadowing the outer 'file' variable
        print(file)
        if 'projections' in file:
            split = file.split('_')[1]
            projections.append(np.load(os.path.join(dir_name, file)))
            kid_df = pd.read_csv(os.path.join(dir_name, f'{split}_kids.csv'))
            print(len(projections[-1]), len(kid_df))
            kids.append(kid_df)
    projections = np.concatenate(projections)
    kids = pd.concat(kids)

    # Keep track of the original order with a temporary index
    kids = kids.reset_index(drop=True)
    kids['original_index'] = kids.index

    # Remove duplicates from df before merging (if needed)
    # This ensures each KID matches only once
    df_unique = df.drop_duplicates(subset=['KID'])

    # Merge while keeping the original order
    target_df = kids.merge(df_unique, on='KID', how='left')

    # Sort back to original order and drop the temporary index
    target_df = target_df.sort_values('original_index').drop('original_index', axis=1)

    # Verify the counts match
    assert len(target_df) == len(kids), f"Length mismatch: target_df ({len(target_df)}) != kids ({len(kids)})"

    Y_hr, X = create_hr_coords(target_df, projections, target_coords)
    Y_pred_lr, score = apply_linear_regression(X, Y_hr, name=f'{target_coords[0]}, {target_coords[1]}')
    # 3. Plot results
    plot_transformation(Y_hr, Y_pred_lr, score, target_df, dir_name, 'combined_projection', target_coords)
def plot_diagrams(dir_name, file_name, umap_input='projections'):
    projections, features, U_standard, df, cmd_mapping_dict = prepare_data(
        dir_name, file_name, umap_input=umap_input
    )
    if umap_input == 'projections':
        for coords in [['BPmRP_0', 'M_G_0'], ['Teff', 'Lstar'], ['Rstar', 'FeH']]:
            transform_eigenspace(df, projections, dir_name, file_name, coords)

    plot_umap(df, U_standard, cmd_mapping_dict, dir_name, file_name)


def compare_lr(dir_name, inputs, properties):
    dirs = [f for f in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, f))]
    results_data = []

    for i, dir in enumerate(dirs):
        files = os.listdir(os.path.join(dir_name, dir))
        for file in files:
            if file.startswith('preds'):
                filename = file.removesuffix('.csv').replace('preds_', '')
                words = filename.split('_')

                # model_name = latex_names[model_name]
                projections, features, umap_coords, df, cmd_mapping_dict = prepare_data(
                    os.path.join(dir_name, dir), filename, calc_umap=False
                )
                # 1. Prepare data
                Y_hr, X = create_hr_coords(df, projections, properties)
                # 2. Apply transformations
                Y_pred_lr, score, acc, loss = apply_linear_regression(X, Y_hr, name=f'{properties[0]}, {properties[1]}')

                plot_transformation(Y_hr, Y_pred_lr, score, df, dir_name, dir, properties)
                print("-------", dir, "------")
                print("score: ", score, "acc: ", acc, "MAE: ", loss)
                results_data.append({
                    'model': dir,
                    'score': score,
                    f'acc_{properties[0]}': acc[0],
                    f'acc_{properties[1]}': acc[1],
                    f'loss_{properties[0]}': loss[0],
                    f'loss_{properties[1]}': loss[1]
                })
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(dir_name, f'results_df.csv'))

def compare_umaps(dir_name, inputs, property):
    dirs = [f for f in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, f))]

    # Calculate grid dimensions
    n_dirs = len(dirs)
    if n_dirs % 2 == 0:
        n_cols = n_dirs // 2
    else:
        n_cols = n_dirs // 2 + 1

    # Check if 'DualFormer (ours)' is in dirs
    has_dualformer = 'DESA (ours)' in dirs
    if has_dualformer:
        # Remove DualFormer from dirs list and handle it separately
        dirs_without_dualformer = [d for d in dirs if d != 'DESA (ours)']
        remaining_dirs = len(dirs_without_dualformer)

        # Create figure with grid layout
        fig = plt.figure(figsize=(40, 24))
        gs = fig.add_gridspec(2, n_cols, hspace=0.3, wspace=0.3)

        # Create DualFormer subplot spanning entire first column (2 rows)
        ax_dualformer = fig.add_subplot(gs[:, 0])

        # Create axes for other directories
        axes_others = []
        for i in range(remaining_dirs):
            if i < (remaining_dirs + 1) // 2:  # First row (excluding column 0)
                ax = fig.add_subplot(gs[0, i + 1])
            else:  # Second row (excluding column 0)
                ax = fig.add_subplot(gs[1, i - (remaining_dirs + 1) // 2 + 1])
            axes_others.append(ax)

        # Process DualFormer first
        files = os.listdir(os.path.join(dir_name, 'DESA (ours)'))
        for file in files:
            if file.startswith('preds'):
                file_name = file.removesuffix('.csv').replace('preds_', '')
                projections, features, umap_coords, df, cmd_mapping_dict = prepare_data(
                    os.path.join(dir_name, 'DESA (ours)'), file_name, umap_input=inputs[0],
                    filter_CMD=True
                )
                ax = ax_dualformer
                label = latex_names[property]
                unit = f'({units[property]})' if property in units else ''
                if ((property == 'Teff') or (property == 'predicted period') or
                        (property == 'RUWE') or (property) == 'Rstar' or (property) == 'Mstar'):
                    color = np.log(df[property])
                    label = f'log({label})'
                else:
                    color = df[property]
                if property == 'flag_CMD_numeric':
                    # Define a colormap with distinct colors (excluding -1 for NaN values)
                    cmap = plt.cm.get_cmap('hsv', 8)
                    legend_handles = []

                    for value, name in cmd_mapping_dict.items():
                        if value != -1:
                            mask = df['flag_CMD_numeric'] == value
                            color = cmap(value % len(cmd_mapping_dict))
                            scatter = ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                                                 c=[color], label=name)
                            legend_handles.append(scatter)

                    mask_nan = df['flag_CMD_numeric'] == -1
                    if mask_nan.any():
                        nan_scatter = ax.scatter(umap_coords[mask_nan, 0], umap_coords[mask_nan, 1],
                                                 c='lightgray', alpha=0.5, s=20, label='NaN')
                        legend_handles.append(nan_scatter)

                    ax.legend(handles=legend_handles, title="CMD Classification",
                              loc='upper left', fontsize=24,
                              prop={'weight': 'bold'},
                              title_fontproperties={'weight': 'bold', 'size': 30}
                              )
                else:
                    sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=color, cmap='OrRd')
                    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', label=f'{label} {unit}')
                    cbar.ax.yaxis.set_tick_params(labelsize=18)
                ax.set_title('DESA (ours)',
                             fontsize=50,
                             fontweight='bold',
                             )
                break

        # Process other directories
        for i, dir in enumerate(dirs_without_dualformer):
            files = os.listdir(os.path.join(dir_name, dir))
            for file in files:
                if file.startswith('preds'):
                    file_name = file.removesuffix('.csv').replace('preds_', '')
                    # Find the correct input index for this directory
                    dir_index = dirs.index(dir)
                    projections, features, umap_coords, df, cmd_mapping_dict = prepare_data(
                        os.path.join(dir_name, dir), file_name, umap_input=inputs[dir_index],
                        filter_CMD=True
                    )
                    ax = axes_others[i]
                    label = latex_names[property]
                    unit = f'({units[property]})' if property in units else ''
                    if ((property == 'Teff') or (property == 'predicted period') or
                            (property == 'RUWE') or (property) == 'Rstar' or (property) == 'Mstar'):
                        color = np.log(df[property])
                        label = f'log({label})'
                    else:
                        color = df[property]
                    if property == 'flag_CMD_numeric':
                        cmap = plt.cm.get_cmap('hsv', 8)
                        legend_handles = []

                        for value, name in cmd_mapping_dict.items():
                            if value != -1:
                                mask = df['flag_CMD_numeric'] == value
                                color = cmap(value % len(cmd_mapping_dict))
                                scatter = ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                                                     c=[color], label=name)
                                legend_handles.append(scatter)

                        mask_nan = df['flag_CMD_numeric'] == -1
                        if mask_nan.any():
                            nan_scatter = ax.scatter(umap_coords[mask_nan, 0], umap_coords[mask_nan, 1],
                                                     c='lightgray', alpha=0.5, s=20, label='NaN')
                            legend_handles.append(nan_scatter)

                        # if i == len(dirs_without_dualformer) - 1:
                        #     ax.legend(handles=legend_handles, title="CMD Classification",
                        #               loc='upper right', fontsize=24, title_fontsize=30)
                    else:
                        sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=color, cmap='OrRd')
                        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', label=f'{label} {unit}')
                        cbar.ax.yaxis.set_tick_params(labelsize=18)
                    ax.set_title(dir, fontsize=50,
                             fontweight='bold',
                             )
                    break

    else:
        # Original layout if no DualFormer
        fig, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(40, 24), sharex=True, sharey=True)
        if n_cols == 1:
            axes = axes.reshape(-1)  # Ensure axes is always 1D array
        else:
            axes = axes.flatten()

        for i, dir in enumerate(dirs):
            files = os.listdir(os.path.join(dir_name, dir))
            for file in files:
                if file.startswith('preds'):
                    file_name = file.removesuffix('.csv').replace('preds_', '')
                    projections, features, umap_coords, df, cmd_mapping_dict = prepare_data(
                        os.path.join(dir_name, dir), file_name, umap_input=inputs[i],
                        filter_CMD=True
                    )
                    ax = axes[i]
                    label = latex_names[property]
                    unit = f'({units[property]})' if property in units else ''
                    if ((property == 'Teff') or (property == 'predicted period') or
                            (property == 'RUWE') or (property) == 'Rstar' or (property) == 'Mstar'):
                        color = np.log(df[property])
                        label = f'log({label})'
                    else:
                        color = df[property]
                    if property == 'flag_CMD_numeric':
                        cmap = plt.cm.get_cmap('hot', len(cmd_mapping_dict))
                        legend_handles = []

                        for value, name in cmd_mapping_dict.items():
                            if value != -1:
                                mask = df['flag_CMD_numeric'] == value
                                color = cmap(value % len(cmd_mapping_dict))
                                scatter = ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                                                     c=[color], label=name)
                                legend_handles.append(scatter)

                        mask_nan = df['flag_CMD_numeric'] == -1
                        if mask_nan.any():
                            nan_scatter = ax.scatter(umap_coords[mask_nan, 0], umap_coords[mask_nan, 1],
                                                     c='lightgray', alpha=0.5, s=20, label='NaN')
                            legend_handles.append(nan_scatter)

                        if i == len(inputs) - 1:
                            ax.legend(handles=legend_handles, title="CMD Classification",
                                      loc='upper right', fontsize=24,
                                      prop={'weight': 'bold'},
                                      title_fontproperties={'weight': 'bold', 'size': 30})
                    else:
                        sc = ax.scatter(umap_coords[:, 0], umap_coords[:, 1], c=color, cmap='OrRd')
                        cbar = fig.colorbar(sc, ax=ax, orientation='vertical', label=f'{label} {unit}')
                        cbar.ax.yaxis.set_tick_params(labelsize=18)
                    ax.set_title(dir, fontsize=30, fontweight='bold')
                    break

    fig.supxlabel('UMAP X')
    fig.supylabel('UMAP Y')
    fig.tight_layout()
    save_path = os.path.join(os.path.dirname(dir_name), f'compare_umaps_{property}.png')
    plt.savefig(save_path)
    print(f"Saved comparison plot to {save_path}")
    plt.show()


def compare_zero_shot(dir_name, inputs):
    """
    Compare zero-shot classification performance using multiple clustering methods.
    """

    dirs = [f for f in os.listdir(dir_name) if os.path.isdir(os.path.join(dir_name, f))]
    results_data = []

    for i, dir in enumerate(dirs):
        files = os.listdir(os.path.join(dir_name, dir))
        for file in files:
            if file.startswith('preds'):
                filename = file.removesuffix('.csv').replace('preds_', '')

                # Prepare data (same as before)
                projections, features, umap_coords, df, cmd_mapping_dict = prepare_data(
                    os.path.join(dir_name, dir), filename, calc_umap=True,
                    filter_CMD=True
                )

                if 'flag_CMD_numeric' not in df.columns:
                    print(f"Warning: 'flag_CMD_numeric' column not found in {filename}")
                    continue

                # Select input data
                if inputs[i] == 'projections':
                    X = projections
                    data_type = 'projections'
                elif inputs[i] == 'features':
                    X = features
                    data_type = 'features'
                elif inputs[i] == 'umap_coords':
                    X = umap_coords
                    data_type = 'umap_coords'
                else:
                    raise ValueError("inputs parameter must be either 'projections' or 'features'")

                # Get true labels and clean data
                y_true = df['flag_CMD_numeric'].values
                valid_mask = ~np.isnan(y_true)
                X_clean = X[valid_mask]
                y_true_clean = y_true[valid_mask]

                if len(X_clean) == 0:
                    print(f"Warning: No valid data for {filename}")
                    continue

                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_clean)

                n_true_clusters = len(np.unique(y_true_clean))

                print("-------", dir, "------")
                print(f"Data type: {data_type}")
                print(f"True number of classes: {n_true_clusters}")

                # Test multiple clustering algorithms
                clustering_results = {}

                # 1. Gaussian Mixture Model (better for overlapping clusters)
                try:
                    gmm = GaussianMixture(n_components=n_true_clusters, random_state=42,
                                          covariance_type='full')  # 'full' allows elliptical clusters
                    gmm_labels = gmm.fit_predict(X_scaled)
                    gmm_ari = adjusted_rand_score(y_true_clean, gmm_labels)
                    gmm_purity = calculate_cluster_purity(y_true_clean, gmm_labels)
                    gmm_accuracy = np.mean(gmm_labels == y_true_clean)
                    clustering_results['GMM'] = {
                        'labels': gmm_labels,
                        'ari': gmm_ari,
                        'purity': gmm_purity,
                        'accuracy': gmm_accuracy,
                        'n_clusters': len(np.unique(gmm_labels))
                    }
                    print(f"GMM - ARI: {gmm_ari:.4f}, Purity: {gmm_purity:.4f}, Accuracy: {gmm_accuracy:.4f}")
                except Exception as e:
                    print(f"GMM failed: {e}")

                # 5. Original K-means for comparison
                try:
                    kmeans = KMeans(n_clusters=n_true_clusters, random_state=42, n_init=10)
                    kmeans_labels = kmeans.fit_predict(X_scaled)
                    kmeans_ari = adjusted_rand_score(y_true_clean, kmeans_labels)
                    kmeans_purity = calculate_cluster_purity(y_true_clean, kmeans_labels)
                    kmeans_accuracy = np.mean(kmeans_labels == y_true_clean)
                    clustering_results['K-Means'] = {
                        'labels': kmeans_labels,
                        'ari': kmeans_ari,
                        'purity': kmeans_purity,
                        'n_clusters': n_true_clusters,
                        'accuracy': kmeans_accuracy
                    }
                    print(f"K-Means - ARI: {kmeans_ari:.4f}, Purity: {kmeans_purity:.4f}, Accuracy: {kmeans_accuracy:.4f}")
                except Exception as e:
                    print(f"K-Means failed: {e}")

                # Find best performing method
                if clustering_results:
                    best_method = max(clustering_results.keys(),
                                      key=lambda x: clustering_results[x]['accuracy'])
                    print(f"Best method: {best_method} (ARI: {clustering_results[best_method]['ari']:.4f})")

                    # Store results for all methods
                    for method_name, result in clustering_results.items():
                        results_data.append({
                            'model': dir,
                            'data_type': data_type,
                            'clustering_method': method_name,
                            'ari_score': result['ari'],
                            'accuracy_score': result['accuracy'],
                            'cluster_purity': result['purity'],
                            'n_clusters_found': result['n_clusters'],
                            'n_true_clusters': n_true_clusters,
                            'n_samples': len(X_clean),
                            'is_best': method_name == best_method
                        })

    # Create DataFrame with results
    results_df = pd.DataFrame(results_data)

    # Save results
    output_filename = f'clustering_comparison_results_{data_type}.csv'
    results_df.to_csv(os.path.join(dir_name, output_filename), index=False)

    # Print summary
    print("\n" + "=" * 50)
    print("CLUSTERING COMPARISON SUMMARY")
    print("=" * 50)

    if not results_df.empty:
        # Best method overall
        best_overall = results_df.loc[results_df['ari_score'].idxmax()]
        print(f"Best overall: {best_overall['clustering_method']} "
              f"(ARI: {best_overall['ari_score']:.4f})")

        # Average performance by method
        method_avg = results_df.groupby('clustering_method')['ari_score'].agg(['mean', 'std']).round(4)
        print("\nAverage ARI scores by method:")
        print(method_avg)

    print(f"\nDetailed results saved to: {output_filename}")
    return results_df


def calculate_cluster_purity(y_true, y_pred):
    """
    Calculate cluster purity - assigns each cluster to its most common true label
    """
    # Handle noise points in DBSCAN (labeled as -1)
    unique_clusters = np.unique(y_pred)
    total_correct = 0
    total_points = len(y_true)

    for cluster in unique_clusters:
        if cluster == -1:  # Skip noise points
            continue
        cluster_mask = y_pred == cluster
        if np.sum(cluster_mask) == 0:
            continue
        cluster_true_labels = y_true[cluster_mask]
        # Find most common true label in this cluster
        most_common_label = np.bincount(cluster_true_labels.astype(int)).argmax()
        # Count correct assignments
        total_correct += np.sum(cluster_true_labels == most_common_label)

    return total_correct / total_points if total_points > 0 else 0


