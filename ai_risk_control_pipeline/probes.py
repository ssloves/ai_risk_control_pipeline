"""Module — linear risk probes + generalization evaluation.

A probe for an axis is the unit (risky_mean - safe_mean) direction in final hidden space; a
prompt's risk score on that axis is its representation projected onto the probe.

Centering: optionally subtract the corpus common-mean (`mu`) before building/applying a probe.
On small models this is a no-op for the reported metrics (the common term cancels in the
difference-of-means direction and in the midpoint classifier). On large pretrained models
(e.g. gpt2) it removes a dominant common-mode offset that otherwise swamps the register signal —
see FINDINGS_gpt2.md. `loo_generalization` and `cross_axis_matrix` center by default.

Two generalization tests address "the probe is too easy / does it generalize to unseen concepts?":
  1. Leave-one-pair-out CV: build the probe without each sentence pair, classify the held-out pair.
  2. Cross-axis transfer matrix: probe_i applied to axis_j -> how concept-specific each probe is.
"""
import numpy as np


def common_mean(model, axes):
    """Mean representation over the whole corpus (the common-mode offset to remove)."""
    every = [s for ax in axes for pole in ("risky", "safe") for s in axes[ax][pole]]
    return model.represent(every).mean(0)

# short alias used by notebooks
_mu = common_mean


def build_probe(model, risky, safe, mu=None):
    R, S = model.represent(risky), model.represent(safe)
    if mu is not None:
        R, S = R - mu, S - mu
    d = R.mean(0) - S.mean(0)
    return d / d.norm()


def axis_score(model, texts, probe, mu=None, steer=None):
    R = model.represent(texts, steer=steer)
    if mu is not None:
        R = R - mu
    return float((R @ probe).mean())


def loo_generalization(model, axes):
    """Leave-one-pair-out CV held-out accuracy per axis (centered)."""
    mu = common_mean(model, axes)
    res = {}
    for ax in axes:
        R, S = axes[ax]["risky"], axes[ax]["safe"]
        n = min(len(R), len(S)); accs = []
        for i in range(n):
            tr_R = [R[j] for j in range(n) if j != i]
            tr_S = [S[j] for j in range(n) if j != i]
            probe = build_probe(model, tr_R, tr_S, mu=mu)
            mid = (axis_score(model, tr_R, probe, mu=mu) + axis_score(model, tr_S, probe, mu=mu)) / 2
            pr = axis_score(model, [R[i]], probe, mu=mu)
            ps = axis_score(model, [S[i]], probe, mu=mu)
            accs.append(((pr > mid) + (ps < mid)) / 2)
        res[ax] = (round(float(np.mean(accs)) * 100, 1), round(float(np.std(accs)) * 100, 1))
    return res


def cross_axis_matrix(model, axes, probes):
    """probes[row] applied to axis[col]: mean risky-minus-safe gap. Diagonal should dominate."""
    mu = common_mean(model, axes)
    names = list(axes); M = {}
    for pi in names:
        M[pi] = {}
        for aj in names:
            g = axis_score(model, axes[aj]["risky"], probes[pi], mu=mu) - \
                axis_score(model, axes[aj]["safe"], probes[pi], mu=mu)
            M[pi][aj] = round(g, 3)
    return M
