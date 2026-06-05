"""Module 3 — per-axis mitigation via contrastive activation steering.

A steering vector (safe_mean - risky_mean) in a chosen block's residual stream is added to that
block's output during inference (forward-hook injection handled by the model), pushing the
perturbed model back toward safe behaviour on that axis.
"""
def safe_direction(model, axes, axis, layer):
    s = model.residual(axes[axis]["safe"], layer).mean(0)
    r = model.residual(axes[axis]["risky"], layer).mean(0)
    d = s - r
    return d / d.norm()


def make_steer(layer, vector, scale):
    return (layer, vector * scale)


def alignment_recovery_rate(baseline, perturbed, steered):
    induced = baseline - perturbed
    recovered = steered - perturbed
    return (recovered / induced) if abs(induced) > 1e-6 else float("nan")
