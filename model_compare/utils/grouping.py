import networkx as nx

metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']


def get_disjoint_nonsig_groups(posthoc_df, alpha=0.05):
    models = posthoc_df.index.tolist()
    G = nx.Graph()

    # Add edges for non-significant differences (p >= alpha)
    for i, m1 in enumerate(models):
        for j in range(i + 1, len(models)):
            m2 = models[j]
            if posthoc_df.loc[m1, m2] >= alpha:
                G.add_edge(m1, m2)

    # Include isolated nodes (not significantly different from anyone)
    for m in models:
        G.add_node(m)

    # Step 1: Get all cliques (overlapping groups)
    cliques = list(nx.find_cliques(G))

    # Step 2: Convert to disjoint groups
    assigned = set()
    disjoint_groups = []
    cliques_sorted = sorted(cliques, key=lambda x: -len(x))  # sort by size desc

    for clique in cliques_sorted:
        group = [m for m in clique if m not in assigned]
        if group:
            disjoint_groups.append(sorted(group))
            assigned.update(group)

    return disjoint_groups


def get_group_labels(posthoc_results, alpha=0.05):
    metric_to_group_label = {}

    for metric in metrics:
        if metric not in posthoc_results:
            continue
        groups = get_disjoint_nonsig_groups(posthoc_results[metric], alpha=alpha)
        label_map = {}
        for i, group in enumerate(groups):
            label = chr(65 + i)  # A, B, C, ...
            for model in group:
                label_map[model] = label
        metric_to_group_label[metric] = label_map

    return metric_to_group_label