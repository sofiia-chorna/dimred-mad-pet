import os

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.evaluation import eval


def plot_split_comparison(actual_dict, pred_dict, title=""):
    plt.figure(figsize=(10, 5))

    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    num_subsets = len(actual_dict)
    colors = plt.cm.tab20(np.linspace(0, 1, num_subsets))

    for i, (subset_name, features) in enumerate(actual_dict.items()):
        ax1.scatter(
            features[:, 0],
            features[:, 1],
            color=colors[i],
            label=subset_name,
            alpha=0.6,
            edgecolors="w",
            s=50,
        )

    ax1.set_title("actual")
    ax1.set_xlabel("smap 1")
    ax1.set_ylabel("smap 2")
    ax1.grid(alpha=0.3)

    for i, (subset_name, features) in enumerate(pred_dict.items()):
        ax2.scatter(
            features[:, 0],
            features[:, 1],
            color=colors[i],
            label=subset_name,
            alpha=0.6,
            edgecolors="w",
            s=50,
        )

    ax2.set_title("predicted")
    ax2.set_xlabel("smap 1")
    ax2.set_ylabel("smap 2")
    ax2.grid(alpha=0.3)
    ax2.legend(
        bbox_to_anchor=(1.0, 1), loc="upper left", borderaxespad=0.0, frameon=False
    )

    eval_res = eval(pred_dict, actual_dict)
    metrics_str = ", ".join(f"{metric}: {val:.3f}" for metric, val in eval_res.items())
    print(f"Eval result: {metrics_str}")

    plt.suptitle(f"{title}: {metrics_str}")
    plt.tight_layout()

    title = title.replace(" ", "_")
    plt.savefig(os.path.join("plots", "smap", f"{title}.png"), dpi=300)
