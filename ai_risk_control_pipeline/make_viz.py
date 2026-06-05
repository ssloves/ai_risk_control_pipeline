import csv, json
import matplotlib.pyplot as plt
import numpy as np

R = "results"
xrows = list(csv.reader(open(f"{R}/cross_axis_transfer.csv")))
names = xrows[0][1:]
M = np.array([[float(c) for c in r[1:]] for r in xrows[1:]])
steer = json.load(open(f"{R}/steering_results.json"))["axes"]
beh = json.load(open(f"{R}/behavior_change.json"))["behavior"]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

# 1) cross-axis transfer heatmap
im = ax[0].imshow(M, cmap="magma", aspect="auto")
ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels([n[:5] for n in names], rotation=30)
ax[0].set_yticks(range(len(names))); ax[0].set_yticklabels([n[:5] for n in names])
ax[0].set_title("Cross-axis probe transfer\n(diagonal = concept-specific)")
ax[0].set_ylabel("probe"); ax[0].set_xlabel("evaluated axis")
for i in range(len(names)):
    for j in range(len(names)):
        ax[0].text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                   color="w" if M[i, j] < M.max() * 0.6 else "k", fontsize=9)
fig.colorbar(im, ax=ax[0], fraction=0.046)

# 2) per-axis ARR
axn = list(steer)
base = [steer[a]["baseline_alignment"] for a in axn]
pert = [steer[a]["perturbed_alignment"] for a in axn]
st = [steer[a]["steered_alignment"] for a in axn]
x = np.arange(len(axn)); w = 0.26
ax[1].bar(x - w, base, w, label="baseline", color="#2a9d8f")
ax[1].bar(x, pert, w, label="injected", color="#e76f51")
ax[1].bar(x + w, st, w, label="steered", color="#457b9d")
ax[1].axhline(0, color="k", lw=.6)
ax[1].set_xticks(x); ax[1].set_xticklabels([a[:5] for a in axn], rotation=30)
ax[1].set_ylabel("alignment (= -risk)")
ax[1].set_title("Per-axis risk control\nARR " + ", ".join(f"{steer[a]['ARR_pct']:.0f}%" for a in axn))
ax[1].legend(fontsize=8)

# 3) KL vs injection strength (behavioural change)
for a in axn:
    c = beh[a]["output_kl_vs_injection_strength"]
    xs = sorted(int(k) for k in c)
    ax[2].plot(xs, [c[str(k)] for k in xs], marker="o", label=a[:8])
ax[2].set_xlabel("injection strength (beta)"); ax[2].set_ylabel("output KL(base || injected)")
ax[2].set_title("Behaviour change is real & monotonic\n(output distribution moves with injection)")
ax[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("visualizations/risk_audit.png", dpi=130, bbox_inches="tight")
print("saved visualizations/risk_audit.png")
