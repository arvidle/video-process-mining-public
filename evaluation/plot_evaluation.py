import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

EVALUATION_FILE_PATH = "clusters_runs.csv"

evaluation_df = pd.read_csv(EVALUATION_FILE_PATH)[["f1", "precision", "fitness", "simplicity", "n_clusters"]]

#evaluation_df = evaluation_df.rename(columns={"Unnamed: 0": "run"}).set_index(["n_clusters", "run"])
#evaluation_df = evaluation_df.ix[:,["fitness", "precision", "f1", "simplicity", "n_clusters"]]
axes = evaluation_df[["fitness", "precision", "f1", "simplicity", "n_clusters"]].plot.box(by="n_clusters", grid=False, layout=(2, 2), figsize=(8, 4), xticks=np.arange(1, 21, 1), yticks=np.arange(0.4, 1.05, 0.1))
#axes[0].set_title("F1-Score")
axes[0].set_title(None)
axes[0].set_xlabel("Number of clusters")
axes[0].set_ylabel("F1-Score")
#axes[1].set_title("Fitness")
axes[1].set_title(None)
axes[1].set_xlabel("Number of clusters")
axes[1].set_ylabel("Fitness")
#axes[2].set_title("Precision")
axes[2].set_title(None)
axes[2].set_xlabel("Number of clusters")
axes[2].set_ylabel("Precision")
#axes[3].set_title("Simplicity")
axes[3].set_title(None)
axes[3].set_xlabel("Number of clusters")
axes[3].set_ylabel("Simplicity")

plt.tight_layout()
plt.savefig("boxplots.svg")

ax = evaluation_df.groupby("n_clusters").mean().plot(figsize=(5, 4), style=["ks-", "k^-", "kx-", "ko-"], markersize=5)
ax.set_ylim((0.5, 1.0))
ax.set_xticks(np.arange(1, 21, 1))
ax.set_xlabel("Number of clusters")
ax.set_ylabel("Score")
ax.legend(loc="lower right")
plt.grid(False)
plt.tight_layout()
plt.savefig("means.svg")
