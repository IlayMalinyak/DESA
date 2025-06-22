import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from eigenspace_analysis import units, latex_names
from scipy.signal import savgol_filter as savgol


berger_catalog_path = r"C:\Users\Ilay\projects\kepler\data\lightPred\tables\berger_catalog_full.csv"

FACTORS = {'final_age_norm': 11, 'age_error_norm': 11}

def get_df(results_dir, filename, to_merge, targets):
    pred_df = pd.read_csv(os.path.join(results_dir, f'predictions_{filename}.csv'))
    files = os.listdir(results_dir)
    if to_merge is not None:
        if to_merge in files:
            age_df = pd.read_csv(os.path.join(results_dir, to_merge))
            for target in targets:
                to_drop = [c for c in age_df.columns if target in c]
                age_df.drop(to_drop, axis=1, inplace=True)
            pred_df = pred_df.merge(age_df, on='KID', how='left')
    kepler_meta = pd.read_csv(berger_catalog_path)
    pred_df = pred_df.merge(kepler_meta, on='KID', how='left', suffixes=['', '_kepler'])
    return pred_df
def get_results(pred_df, target, color_col):
    mult_fact = FACTORS[target]
    gt = pred_df[target].values * mult_fact

    preds_median = pred_df[f'pred_{target}_q0.500'].values * mult_fact
    preds_lower = pred_df[f'pred_{target}_q0.140'].values * mult_fact
    preds_upper = pred_df[f'pred_{target}_q0.860'].values * mult_fact
    sort_idx = np.argsort(preds_median)

    pred_df = pred_df.iloc[sort_idx]
    if color_col is not None:
        if pred_df[color_col].dtype == 'object':
            color_col_numeric = f'{color_col}_numeric'
            color_vals = np.unique(pred_df[color_col])
            for i, val in enumerate(color_vals):
                pred_df.loc[pred_df[color_col]==val, color_col_numeric] = i
                colors = pred_df[color_col_numeric].values
        else:
            colors = pred_df[color_col].values
    else:
        colors = None

    mae = np.mean(np.abs(preds_median[sort_idx] - gt[sort_idx]))
    rmse = np.sqrt(np.mean((preds_median[sort_idx] - gt[sort_idx]) ** 2))
    rel_error = np.mean(np.abs(preds_median[sort_idx] - gt[sort_idx]) / gt[sort_idx])
    acc = np.mean(np.abs(preds_median[sort_idx] - gt[sort_idx]) < gt * 0.1)
    coverage = np.mean((gt[sort_idx] >= preds_lower[sort_idx]) &
                (gt[sort_idx] <= preds_upper[sort_idx]))
    metrics_dict = {'mae': mae, 'rmse': rmse, 'rel_error': rel_error, 'coverage': coverage, 'acc': acc}

    return gt[sort_idx], preds_median[sort_idx], preds_lower[sort_idx], preds_upper[sort_idx], colors, metrics_dict
def plot_regression_results(results_dir, filename,
                            targets=[], to_merge=None,
                            color_col=None):
    pred_df = get_df(results_dir, filename, to_merge, targets)
    for target in targets:
        gt, preds_median, preds_lower, preds_upper, colors, metrics_dict = get_results(pred_df, target, color_col)
        print(metrics_dict)
        target_name = latex_names[target]
        target_units = f'({units[target]})' if target in units else ''
        if color_col is not None:
            color_name = latex_names[color_col]
            color_units = f'({units[color_col]})' if color_col in units else ''
            plt.scatter(preds_median, gt, c=colors)
            cbar = plt.colorbar(label=f'{color_name} {color_units}')
        else:
            plt.scatter(preds_median, gt)
        plt.plot(preds_median, savgol(preds_lower, 20, polyorder=1), alpha=0.3, color='darkblue')
        plt.plot(preds_median, savgol(preds_upper, 20, polyorder=1), alpha=0.3, color='darkblue')

        # plt.fill_between(preds_median,
        #                  preds_lower,
        #                  preds_upper,
        #                  alpha=0.2,
        #                  color='lightsalmon',
        #                  label='86% confidence interval')

        plt.xlabel(f'Predicted {target_name} {target_units}')
        plt.ylabel(f'Target {target_name} {target_units}')
        # plt.title(f"{target} - \nMAE: {metrics_dict['mae']:.3f},"
        #           )
        if 'dual_former' in filename:
            plt.title("DESA (ours)")
        plt.savefig(os.path.join(results_dir, f'{filename}_{target}.png'))
        plt.show()

def compare_regression_experiments(dir, targets, color_col=None):
    files = [f for f in os.listdir(dir) if f.endswith('.csv')]
    all_figs = []
    all_axes = []
    for _ in range(len(targets)):
        # if len(files) % 2 == 0:
        #     fig, axes = plt.subplots(nrows=2, ncols=(len(files) - 1) // 2, figsize=(30, 16))
        # else:
        #     fig, axes = plt.subplots(nrows=2, ncols=(len(files) - 1) // 2 + 1, figsize=(36, 19))
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(30, 16))
        all_figs.append(fig)
        all_axes.append(axes)
    for j, filename in enumerate(files):
        if 'dual_former' in filename:
            continue
        filename = filename.replace('predictions_', '').removesuffix('.csv')
        words = filename.split('_')
        if words[-1] == 'spec' or words[-1] == 'light' or words[-1] == 'former' or words[-1] == 'uniqueLoader':
            start_idx = -2
            model_name = '_'.join(words[start_idx:])
        else:
            model_name = words[-1]
        print(filename)
        for i, target in enumerate(targets):
            target_name = latex_names[target]
            target_units = f'({units[target]})' if target in units else ''
            pred_df = get_df(dir, filename, None, target)
            gt, preds_median, preds_lower, preds_upper, colors, metrics_dict = get_results(pred_df, target, color_col)
            print("-----", target_name, target_units, "-----")
            print("gt median : ", np.median(gt))
            print(metrics_dict)
            fig, axes = all_figs[i], all_axes[i]
            if color_col is not None:
                color_name = latex_names[color_col]
                color_units = f'({units[color_col]})' if color_col in units else ''
                sc = axes.ravel()[j].scatter(preds_median, gt, c=colors)
                cbar = fig.colorbar(sc, orientation='vertical', label=f'{color_name} {color_units}')
            else:
                axes.ravel()[j].scatter(preds_median, gt)
            # axes.ravel()[j].fill_between(preds_median,
            #                  preds_lower,
            #                  preds_upper,
            #                  alpha=0.2,
            #                  color='lightsalmon',
            #                  )
            axes.ravel()[j].plot(preds_median, savgol(preds_lower, 20, polyorder=1), alpha=0.3, color='darkblue')
            axes.ravel()[j].plot(preds_median, savgol(preds_upper, 20, polyorder=1), alpha=0.3, color='darkblue')
            # axes.ravel()[j].text(0.05, 0.95, f"RMSE {target_units}: {metrics_dict['rmse']:.3f}",
            #                      transform=axes.ravel()[j].transAxes,
            #                      verticalalignment='top',
            #                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            axes.ravel()[j].set_xlabel(f'Predicted {target_name} {target_units}')
            axes.ravel()[j].set_ylabel(f'Target {target_name} {target_units}')
            axes.ravel()[j].set_title(latex_names[model_name])
    for i, fig in enumerate(all_figs):
        fig.tight_layout()
        target_name = targets[i].replace('/', '_')  # Replace any problematic characters
        filename = f"{target_name}_regression_comparison.png"
        filepath = os.path.join(dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved figure for {targets[i]} to {filepath}")
    plt.show()

