import pandas as pd
import numpy as np
import os
from eigenspace_analysis import transform_eigenspace, plot_diagrams, plot_umap, giant_cond, get_mag_data, latex_names, units
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import umap
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines


from matplotlib.ticker import LogLocator, LogFormatter



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


def get_df(results_dir, exp_name, snr_cut=None):
    df = pd.read_csv(os.path.join(results_dir, f'preds_{exp_name}.csv'))
    berger = pd.read_csv(berger_catalog_path)
    lightpred = pd.read_csv(lightpred_full_catalog_path)
    godoy = pd.read_csv(godoy_catalog_path)

    df = df.merge(berger, right_on='KID', left_on='kid', how='left')
    df = get_mag_data(df)
    age_cols = [c for c in lightpred.columns if 'age' in c]  # Robustly find age columns
    df = df.merge(lightpred[['KID', 'predicted period', 'mean_period_confidence'] + age_cols], right_on='KID',
                  left_on='kid', how='left')
    df = df.merge(godoy, right_on='KIC', left_on='kid', how='left', suffixes=['', '_godoy'])
    if snr_cut is not None:
        lamost_kepler = pd.read_csv(lamost_kepler_path)
        lamost = pd.read_csv(lamost_catalog_path, sep='|')
        df = df.merge(lamost_kepler, left_on='kid', right_on='kepid').merge(lamost, left_on='obsid', right_on='combined_obsid',
        suffixes=['', '_lamost'])
        df = df[df['combined_snrg'] > snr_cut]
    df['subgiant'] = (df['flag_CMD'] == 'Subgiant').astype(int)
    df['main_seq'] = df.apply(giant_cond, axis=1)
    return df

def plot_confusion_mat(y_true, y_pred, results_dir, file_name):
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # plt.title(file_name)
    plt.colorbar()

    # Labeling
    num_classes = len(np.unique(y_true))
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Display counts
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'confusion_{file_name}.png'))
    plt.clf()

def plot_roc_curve(df, cls_names, results_dir, file_name):
    num_classes = len([col for col in df.columns if col.startswith('pred_')])
    assert num_classes == len(cls_names)
    df['target'] = df['target'].astype(int)

    # Binarize ground truth labels for multiclass ROC/PR curves
    if num_classes == 2:
        # Special case: binary classification -> manually build two columns
        y_true_bin = np.zeros((len(df), 2))
        y_true_bin[np.arange(len(df)), df['target']] = 1
    else:
        y_true_bin = label_binarize(df['target'], classes=np.arange(num_classes))

    # Stack predicted probabilities
    y_score = df[[f'pred_{i}' for i in range(num_classes)]].values

    # --- ROC CURVES ---
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    # --- PLOTTING ROC CURVES ---
    # plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'{cls_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(os.path.join(results_dir, f'roc_{file_name}.png'))
    plt.show()

def plot_precision_recall_curve(df, cls_names, results_dir, file_name):
    num_classes = len([col for col in df.columns if col.startswith('pred_')])
    assert num_classes == len(cls_names)

    # Binarize ground truth labels for multiclass ROC/PR curves
    if num_classes == 2:
        # Special case: binary classification -> manually build two columns
        y_true_bin = np.zeros((len(df), 2))
        y_true_bin[np.arange(len(df)), df['target']] = 1
    else:
        y_true_bin = label_binarize(df['target'], classes=np.arange(num_classes))

    # Stack predicted probabilities
    y_score = df[[f'pred_{i}' for i in range(num_classes)]].values

    # --- PRECISION-RECALL CURVES ---
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_score[:, i])

    # --- PLOTTING PRECISION-RECALL CURVES ---
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(recall[i], precision[i], label=f'{cls_names[i]} (AP = {average_precision[i]:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.grid()
    plt.savefig(os.path.join(results_dir, f'precision_recall_{file_name}.png'))
    plt.show()

def plot_cls_umap(df, umap_coords, results_dir, file_name):
    # Define your color mapping manually
    color_dict = {-1: 'plum', 0: 'gray', 1: 'lightskyblue', 2: 'moccasin',  4: 'gold'}
    label_dict = {-1: 'RV', 0: 'Spectroscopic', 1: 'EB', 2: 'Astrometric', 4: 'Singles'}

    # Initialize binary_color
    df['binary_color'] = np.ones(len(df), dtype=int) * -999

    # Assign labels
    df.loc[df['flag_RVvariable'], 'binary_color'] = -1
    df.loc[df['NSS_Binary_Type'] == 'spectroscopic', 'binary_color'] = 0
    df.loc[(df['flag_EB_Kepler'] == 1)
           | (df['flag_EB_Gaia'] == 1)
           | (df['NSS_Binary_Type'] == 'eclipsing')
           | (df['NSS_Binary_Type'] == 'eclipsing+spectroscopic'), 'binary_color'] = 1
    df.loc[(df['NSS_Binary_Type'] == 'astrometric')
           | (df['NSS_Binary_Type'] == 'spectroscopic+astrometric'), 'binary_color'] = 2
    df.loc[df['flag_Binary_Union'] == 0, 'binary_color'] = 4

    print("min value binary color: ", np.min(df['binary_color']))

    # Create the plot
    fig, ax = plt.subplots()

    # Plot each group separately
    for value, color in color_dict.items():
        mask = df['binary_color'] == value
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                   c=color, label=label_dict[value], s=50)

    # Build custom legend automatically
    ax.legend(fontsize=20, loc='best')

    ax.set_xlabel('UMAP X', fontsize=20)
    ax.set_ylabel('UMAP Y', fontsize=20)
    # ax.set_title('UMAP of Eigenspace', fontsize=20)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'umap_{file_name}_binaries_color.png'))
    plt.show()

def plot_classification_results(results_dir, exp_name, plot_umap=False, snr_cut=None):

    df = get_df(results_dir, exp_name, snr_cut=snr_cut)
    if plot_umap:
        projections = np.load(os.path.join(results_dir, f'projections_{exp_name}.npy'))
        features = np.load(os.path.join(results_dir, f'features_{exp_name}.npy'))
        reducer_standard = umap.UMAP(n_components=2, random_state=42)  # Added random_state for reproducibility
        U_proj = reducer_standard.fit_transform(projections)
        U_feat = reducer_standard.fit_transform(features)
        # plot_umap(df, U_proj, results_dir, exp_name + '_proj')
        plot_cls_umap(df, U_feat, results_dir, exp_name + '_feat')

    y_true = df['target']
    y_pred = df['preds_cls']
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f'Accuracy: {accuracy:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    plot_confusion_mat(y_true, y_pred, results_dir, exp_name)
    plot_roc_curve(df, ['singles', 'binaries'], results_dir, exp_name)
    plot_precision_recall_curve(df, ['singles', 'binaries'], results_dir, exp_name)


def compare_cls_experiments(dir_path):
    """
    Compares classification experiments by plotting ROC and PR curves
    with formatted legends.
    """
    files_nss = [f for f in os.listdir(dir_path) if f.endswith('csv') and f.startswith('preds_')]
    models_names_original = []
    for i, file in enumerate(files_nss):
        filename = file.replace('preds_', '').replace('.csv', '')
        words = filename.split('_')
        # Heuristic to extract model name from filename
        if words[-1] == 'spec' or words[-1] == 'light' or words[-1] == 'former' or words[-1] == 'uniqueLoader':
            start_idx = -2
            model_name_key = '_'.join(words[start_idx:])
        else:
            model_name_key = words[-1]
        models_names_original.append(model_name_key)  # Store the key for latex_names

    # Sort files and model names together
    if not files_nss:
        print(f"No prediction CSV files found in directory: {dir_path}")
        return

    sorted_pairs = sorted(zip(files_nss, models_names_original),
                          key=lambda x: latex_names.get(x[1], x[1]))  # Sort by display name
    files_nss, models_names_original = zip(*sorted_pairs)

    files_nss = list(files_nss)
    models_names_original = list(models_names_original)

    # Create subplots for ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=False, sharex=False)

    cls_names = ['singles', 'binaries']  # Class names used in the data

    # Get a colormap
    # Using a qualitative colormap like 'tab10' or 'viridis' for distinct colors
    num_distinct_models = max(len(models_names_original), 8)
    colors_map = plt.cm.get_cmap('tab10', num_distinct_models)

    for i, (model_key, file) in enumerate(zip(models_names_original, files_nss)):
        # Load the predictions file
        df = pd.read_csv(os.path.join(dir_path, file))

        # Get number of classes and prepare data
        num_classes = len([col for col in df.columns if col.startswith('pred_')])
        if num_classes != len(cls_names):
            print(
                f"Warning: Mismatch in expected number of classes for {file}. Expected {len(cls_names)}, found {num_classes}")
            # Continue or skip based on desired behavior
            # For now, we'll try to proceed if possible, or you might want to `continue`
            if num_classes == 0:
                print(f"Skipping {file} as no 'pred_' columns found.")
                continue

        df['target'] = df['target'].astype(int)

        # Binarize ground truth labels for multiclass ROC/PR curves
        if num_classes > 0:  # Proceed only if classes are found
            if num_classes == 2 and len(df['target'].unique()) <= 2:  # Standard binary or handled as such
                y_true_bin = np.zeros((len(df), num_classes))
                # Ensure targets are within [0, num_classes-1]
                valid_targets = df['target'][df['target'] < num_classes]
                valid_indices = valid_targets.index
                if not valid_indices.empty:
                    y_true_bin[valid_indices, valid_targets.values] = 1
            elif num_classes > 1:  # Multiclass
                y_true_bin = label_binarize(df['target'], classes=np.arange(num_classes))
            else:  # Single class prediction, needs careful handling for ROC/PR
                print(f"Warning: Only one class predicted in {file}. ROC/PR may not be meaningful.")
                # Create a dummy y_true_bin or skip this model for these plots
                y_true_bin = label_binarize(df['target'],
                                            classes=[0, 1])  # Assuming binary context if only one pred_ col
                if y_true_bin.shape[1] == 1:  # if label_binarize only makes one column
                    y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))  # make it two columns

            # Stack predicted probabilities
            y_score = df[[f'pred_{j}' for j in range(num_classes)]].values
        else:
            continue  # Skip if no classes


        current_color = colors_map(i % colors_map.N)  # Use modulo for safety if more models than colors

        # Get the display name from latex_names, fallback to key if not found
        display_model_name = latex_names.get(model_key, model_key)

        linewidth = 6 if 'dualformer' in display_model_name.lower() else 4

        plot_confusion_mat(df['target'], df['preds_cls'], dir_path, display_model_name)

        # --- ROC CURVES ---
        for class_idx in range(num_classes):
            # Ensure y_true_bin and y_score have compatible shapes for the current class_idx
            if class_idx < y_true_bin.shape[1] and class_idx < y_score.shape[1]:
                fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_score[:, class_idx])
                roc_auc = auc(fpr, tpr)

                if cls_names[class_idx] == 'binaries':  # Only plot for 'binaries' class
                    ax1.plot(1 - fpr, np.clip(1 - tpr, a_min=1e-3, a_max=None),
                             color=current_color,
                             linewidth=linewidth,
                             label=f'{display_model_name} - (AUC = {roc_auc:.2f})',
                             )
            else:
                print(
                    f"Warning: class_idx {class_idx} out of bounds for y_true_bin or y_score for model {display_model_name}")

        # --- PRECISION-RECALL CURVES ---
        for class_idx in range(num_classes):
            if class_idx < y_true_bin.shape[1] and class_idx < y_score.shape[1]:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, class_idx], y_score[:, class_idx])
                average_precision = average_precision_score(y_true_bin[:, class_idx], y_score[:, class_idx])

                if cls_names[class_idx] == 'binaries':  # Only plot for 'binaries' class
                    ax2.plot(recall, precision,
                             color=current_color,
                             linewidth=linewidth,
                             label=f'{display_model_name} (AP = {average_precision:.2f})')
            else:
                print(
                    f"Warning: class_idx {class_idx} out of bounds for y_true_bin or y_score for model {display_model_name}")

    # Configure ROC plot (ax1)
    ax1.plot([0, 1], [9e-4, 1], 'k--', alpha=0.5)  # Adjusted for 1-tpr on y-axis (log scale)
    # (0,1) on x is (1,0) for 1-fpr
    # (1, 1e-3) on y is (0, 1-1e-3) for 1-tpr
    ax1.set_xlim([9e-4, 1.0])  # For 1-fpr (TNR)
    ax1.set_ylim([9e-4, 1.0])  # For 1-tpr (FNR)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    # ax1.invert_xaxis()  # TNR usually goes from 0 to 1, so 1-fpr goes 1 to 0. Invert to show 0 to 1.
    # ax1.invert_yaxis()  # FNR usually goes from 0 to 1, so 1-tpr goes 1 to 0. Invert to show 0 to 1.

    ax1.set_xlabel('True Negative Rate')
    ax1.set_ylabel('False Negative Rate')
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # Configure Precision-Recall plot (ax2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_xlim([0.45, 1.0])
    ax2.set_ylim([0.45, 1.0])
    ax2.grid(True, alpha=0.2)

    # --- Advanced Legend Formatting ---
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Create font properties for monospace font
    mono_font = FontProperties(family='monospace', size=20, weight='bold')  # Adjusted size for better fit

    # 1. Determine the maximum model name length for alignment
    all_model_names_for_len_calc = ["Model"]  # Start with "Model" for the header
    temp_labels_for_len_calc_auc = []
    temp_labels_for_len_calc_ap = []

    if labels1:
        for raw_label in labels1:
            temp_labels_for_len_calc_auc.append(raw_label.split(' - (AUC = ')[0])
    if labels2:
        for raw_label in labels2:
            temp_labels_for_len_calc_ap.append(raw_label.split(' (AP = ')[0])

    all_model_names_for_len_calc.extend(temp_labels_for_len_calc_auc)
    all_model_names_for_len_calc.extend(temp_labels_for_len_calc_ap)
    # Remove duplicates that might arise if a model name appears in both plots
    all_model_names_for_len_calc = list(dict.fromkeys(all_model_names_for_len_calc))

    if not all_model_names_for_len_calc or (
            len(all_model_names_for_len_calc) == 1 and all_model_names_for_len_calc[0] == "Model"):
        max_model_name_len = 20  # Default if no actual model names
    else:
        max_model_name_len = max(len(name) for name in all_model_names_for_len_calc)
    max_model_name_len = max(max_model_name_len, len("Model"))  # Ensure "Model" header fits

    # Create a dummy handle for the header row in the legend
    dummy_handle = mlines.Line2D([], [], color='none', marker='None', linestyle='None', label='_nolegend_')

    # --- Legend for ax1 (AUC) ---
    if handles1:
        auc_legend_header = f"{'Model':<{max_model_name_len}}  {'AUC'}"
        formatted_auc_data_labels = []
        for raw_label in labels1:  # Use labels1 obtained from ax1
            parts = raw_label.split(' - (AUC = ')
            model_name_part = parts[0]
            auc_value_str = parts[1].replace(')', '')
            try:
                auc_value_float = float(auc_value_str)
                formatted_auc_val = f"{auc_value_float:.2f}"  # Consistent .2f
            except ValueError:
                formatted_auc_val = auc_value_str
            formatted_auc_data_labels.append(f'{model_name_part:<{max_model_name_len}}  {formatted_auc_val}')

        final_handles_for_ax1 = [dummy_handle] + handles1
        final_labels_for_ax1 = [auc_legend_header] + formatted_auc_data_labels

        ax1.legend(final_handles_for_ax1, final_labels_for_ax1, loc='best',  # Changed to 'best'
                   prop=mono_font, handlelength=1.0, handletextpad=0.5, labelspacing=0.7,
                   title_fontproperties={'weight': 'bold', 'size': mono_font.get_size() + 1},
                   frameon=True, shadow=True)
    else:
        print("No handles found for ax1 legend.")

    # --- Legend for ax2 (AP) ---
    if handles2:
        ap_legend_header = f"{'Model':<{max_model_name_len}}  {'AP'}"
        formatted_ap_data_labels = []
        for raw_label in labels2:  # Use labels2 obtained from ax2
            parts = raw_label.split(' (AP = ')
            model_name_part = parts[0]
            ap_value_str = parts[1].replace(')', '')
            try:
                ap_value_float = float(ap_value_str)
                formatted_ap_val = f"{ap_value_float:.2f}"  # Consistent .2f
            except ValueError:
                formatted_ap_val = ap_value_str
            formatted_ap_data_labels.append(f'{model_name_part:<{max_model_name_len}}  {formatted_ap_val}')

        final_handles_for_ax2 = [dummy_handle] + handles2
        final_labels_for_ax2 = [ap_legend_header] + formatted_ap_data_labels

        ax2.legend(final_handles_for_ax2, final_labels_for_ax2, loc='best',  # Changed to 'best'
                   prop=mono_font, handlelength=1.0, handletextpad=0.5, labelspacing=0.7,
                   title_fontproperties={'weight': 'bold', 'size': mono_font.get_size() + 1},
                   frameon=True, shadow=True)
    else:
        print("No handles found for ax2 legend.")
    # --- End of Advanced Legend Formatting ---

    plt.tight_layout(pad=1.5)  # Add padding

    # Ensure output directory exists
    output_filename = os.path.join(dir_path, 'experiments_comparison.png')
    try:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show()

    # Print summary statistics for each experiment
    # print("\nExperiment Summary:")
    # print("-" * 80)
    # for file in files_nss:
    #     file_name = file.replace('preds_', '').replace('.csv', '')
    #     df = pd.read_csv(os.path.join(dir, file))
    #
    #     y_true = df['target']
    #     y_pred = df['preds_cls']
    #
    #     accuracy = accuracy_score(y_true, y_pred)
    #     precision = precision_score(y_true, y_pred, average='weighted')
    #     recall = recall_score(y_true, y_pred, average='weighted')
    #     f1 = f1_score(y_true, y_pred, average='weighted')
    #
    #     print(f"{file_name:20} | Acc: {accuracy:.3f} | Prec: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")