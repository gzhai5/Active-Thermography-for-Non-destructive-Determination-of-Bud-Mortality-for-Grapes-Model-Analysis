import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Patch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from utils.grouping import get_group_labels
from utils.ci import bootstrap_ci
from collections import defaultdict


metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']


def draw_group_circles(ax, groups, x_positions, y_means, color_map):
    for i, group in enumerate(groups):
        xs = [x_positions[m] for m in group]
        ys = [y_means[m] for m in group]

        # Circle center = mean of positions
        x_center = np.mean(xs)
        y_center = np.mean(ys)

        # Width and height of ellipse
        if len(group) == 1:
            width, height = 0.25, 0.02  # small circle for singletons
        else:
            width = (max(xs) - min(xs)) + 0.25
            height = (max(ys) - min(ys)) + 0.02

        ellipse = Ellipse((x_center, y_center),
                          width=width,
                          height=height,
                          edgecolor=color_map[i],
                          facecolor=color_map[i],
                          alpha=0.2,
                          linewidth=1.5)
        ax.add_patch(ellipse)


def visulize_letters(models, df, group_labels_per_metric):
    plt.figure(figsize=(12, 6))
    model_offsets = np.linspace(-0.3, 0.3, len(models))
    colors = sns.color_palette("Set2", len(models))

    for i, metric in enumerate(metrics):
        group_labels = group_labels_per_metric.get(metric, {})
        for j, model in enumerate(models):
            vals = df[df['model'] == model][metric].values
            mean, lower, upper = bootstrap_ci(vals)
            error_lower = mean - lower
            error_upper = upper - mean

            x = i + model_offsets[j]
            plt.errorbar(x=x,
                        y=mean,
                        yerr=[[error_lower], [error_upper]],
                        fmt='o',
                        capsize=4,
                        label=model if i == 0 else "",
                        color=colors[j])

            # Annotate with group label (if any)
            group_label = group_labels.get(model, "")
            if group_label:
                plt.text(x, mean + 0.003, group_label, ha='center', va='bottom', fontsize=9, color='black')

    # Add vertical separators
    for i in range(len(metrics) - 1):
        plt.axvline(x=i + 0.5, color='blue', linestyle='--', alpha=0.5)

    plt.xticks(ticks=np.arange(len(metrics)), labels=[m.replace("_", " ").title() for m in metrics])
    plt.ylabel("Score")
    plt.title("Model Performance: Mean ± 95% CI per Metric\n(Annotated by Non-Significant Group Letters)")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def visulize_circles(models, df, group_labels_per_metric):
    colors = sns.color_palette("Set2", len(models))  # Color per model

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    model_offsets = np.linspace(-0.3, 0.3, len(models))

    for i, metric in enumerate(metrics):
        group_labels = group_labels_per_metric.get(metric, {})
        label_to_models = defaultdict(list)
        for model, label in group_labels.items():
            label_to_models[label].append(model)

        x_positions = {}
        y_means = {}

        for j, model in enumerate(models):
            vals = df[df['model'] == model][metric].values
            mean, lower, upper = bootstrap_ci(vals)
            error_lower = mean - lower
            error_upper = upper - mean

            x = i + model_offsets[j]
            x_positions[model] = x
            y_means[model] = mean

            plt.errorbar(x=x,
                        y=mean,
                        yerr=[[error_lower], [error_upper]],
                        fmt='o',
                        capsize=4,
                        label=model if i == 0 else "",
                        color=colors[j])

        # Draw statistical group ellipses
        group_colors = sns.color_palette("pastel", len(label_to_models))
        draw_group_circles(ax, list(label_to_models.values()), x_positions, y_means, group_colors)

    # Add vertical metric dividers
    for i in range(len(metrics) - 1):
        plt.axvline(x=i + 0.5, color='blue', linestyle='--', alpha=0.5)

    plt.xticks(ticks=np.arange(len(metrics)), labels=[m.replace("_", " ").title() for m in metrics])
    plt.ylabel("Score")
    plt.title("Model Performance: Mean ± 95% CI per Metric\n(Statistically Non-significant Groups Circled)")
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_grouped_bar_with_ci(models, df, group_labels_per_metric, cultivar='All Cultivars', alpha=0.05):
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(1.8 * len(metrics), 4), sharey=True)

    custom_colors = list(reversed([
        "#e66d50", "#f3a361", "#e7c66b", "#8ab07c",
        "#299d8f",
    ]))
    models = [
        'logistic',
        'randomforest',
        'svm',
        'lstm',                # base LSTM
        'lstm-features',       # hybrid LSTM (comes after)
        'lstmfcn',             # base LSTM-FCN
        'lstmfcn-features',    # hybrid LSTM-FCN (comes after)
        'vivit'
    ]
    model_name_mapping = {
        'logistic': 'Logistic Regression',
        'randomforest': 'Random Forest',
        'svm': 'SVM',
        'lstm-features': 'Hybrid LSTM',
        'lstm': 'LSTM',
        'lstmfcn': 'LSTM-FCN',
        'lstmfcn-features': 'Hybrid LSTM-FCN',
        'vivit': 'ViViT',
    }

    if len(metrics) == 1:
        axes = [axes]

    max_group_size = -1
    for i, metric in enumerate(metrics):
        ax = axes[i]
        group_labels = group_labels_per_metric.get(metric, {})

        # Collect stats per model
        model_stats = {}
        for model in models:
            vals = df[df['model'] == model][metric].values
            if len(vals) == 0:
                print(f"Warning: No data for model {model} and metric {metric}")
                model_stats[model] = {'mean': np.nan, 'lower': 0, 'upper': 0, 'group': group_labels.get(model, None)}
            else:
                mean, lower, upper = bootstrap_ci(vals)
                print(f"Model: {model}, Metric: {metric}, Mean: {mean}, CI: [{lower}, {upper}]")
                model_stats[model] = {'mean': mean, 'lower': mean - lower, 'upper': upper - mean, 'group': group_labels.get(model, None)}

        # Group models by group label
        group_to_models = defaultdict(list)
        for model, stats in model_stats.items():
            group = stats['group']
            if group is not None:
                group_to_models[group].append((model, stats['mean']))

        # Compute average mean per group and sort groups
        group_avg = {g: np.nanmean([m[1] for m in members]) for g, members in group_to_models.items()}
        ranked_groups = sorted(group_avg.items(), key=lambda x: -x[1])
        max_group_size = max(max_group_size, len(ranked_groups))
        group_to_color = {group: custom_colors[rank] for rank, (group, _) in enumerate(ranked_groups)}

        # Draw bars in fixed order
        y_pos = np.arange(len(models))
        means = []
        lowers = []
        uppers = []
        colors = []

        for model in models:
            stat = model_stats[model]
            means.append(stat['mean'])
            lowers.append(stat['lower'])
            uppers.append(stat['upper'])
            group = stat['group']
            color = group_to_color.get(group, 'lightgray')
            colors.append(color)

        # for y, (mean, lower) in enumerate(zip(means, lowers)):
        #     if not np.isnan(mean):
        #         ax.text(mean - lower - 0.12, y, f"{mean:.3f}", va='center', fontsize=8, color='black')

        # Plot bars
        ax.barh(y=y_pos, width=means, xerr=[lowers, uppers],
                align='center', color=colors, ecolor='black', capsize=4)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([model_name_mapping.get(m, m) for m in models], fontsize=12)
        ax.invert_yaxis()
        ax.set_title(metric.replace("_", " ").title(), fontsize=12)
        
        ax.set_xlim(0.55, 0.9)
        ax.set_xticks(np.arange(0.55, 0.91, 0.1))
        ax.tick_params(axis='x', labelsize=8)


    rank_to_letter_map = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E', '6': 'F', '7': 'G', '8': 'H'}
    legend_patches = [
        Patch(facecolor=custom_colors[rank], edgecolor='black', label=f'Rank {(rank + 1)}')
        for rank in range(max_group_size)
    ]

    
    # fig.suptitle(f"Model Performance with 95% CI per Metric (for {cultivar})", fontsize=14, y=0.9)
    fig.legend(
        handles=legend_patches,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.03),  # adjust as needed
        ncol=len(legend_patches),
        fontsize=12,
        title_fontsize=13,
        frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    plt.savefig(f'figures/compare_{cultivar}.svg', bbox_inches='tight')
    plt.show()