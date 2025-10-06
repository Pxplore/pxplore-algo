import matplotlib.pyplot as plt
import os
import matplotlib

# Set global font sizes
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
})

# Data
models = ["Retrieval", "GPT-4o", "Pxplore"]
precision = [0.51, 0.52, 0.58]
ndcg_at = [1, 3, 5, 7, 10]
ndcg_scores = {
    "Retrieval": [0.80, 0.85, 0.83, 0.80, 0.80],
    "GPT-4o":    [0.85, 0.86, 0.83, 0.81, 0.81],
    "Pxplore":    [0.86, 0.87, 0.86, 0.84, 0.84],
}

# Unified color scheme
colors = ["#d1cee3", "#fadad5", "#b0e3e6"]

# ---------- First PDF: Precision@1 Chart ----------
fig1, ax1 = plt.subplots(1, 1, figsize=(5, 2))

# Horizontal bar chart for Precision@1
bar_positions = [0,1,2]
ax1.barh(bar_positions, precision, color=colors, alpha=1, height=0.5)
ax1.set_yticks(bar_positions)
ax1.set_yticklabels(models)
ax1.set_xlabel("Precision")
ax1.set_xlim(0.45, 0.6)
# ax1.set_title("Precision@1")


plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "result_decision_precision.pdf")
plt.savefig(output_path)
plt.close()

# ---------- Second PDF: NDCG Chart ----------
fig2, ax2 = plt.subplots(1, 1, figsize=(5, 3))

# Line chart for NDCG scores
for i, (model, scores) in enumerate(ndcg_scores.items()):
    ax2.plot(ndcg_at, scores, marker="o", label=model, color=colors[i])
ax2.set_xlabel("k")
ax2.set_ylabel("NDCG@k")
ax2.set_ylim(0.78, 0.9)
# ax2.set_title("Normalized Discounted Cumulative Gain (NDCG)")
ax2.legend(loc="upper right")

plt.tight_layout()
output_path = os.path.join(os.path.dirname(__file__), "result_decision_ndcg.pdf")
plt.savefig(output_path)
plt.close()