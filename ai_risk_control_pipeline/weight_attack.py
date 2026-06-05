"""Module 1 — per-axis failure injection via a targeted weight-space edit.

For a chosen risk axis, the risky direction in a block's residual stream (risky_mean - safe_mean)
is added to that block's MLP output bias, nudging the model toward that risk. Returns a perturbed
copy; the clean model is untouched.
"""
def risky_direction(model, axes, axis, layer):
    r = model.residual(axes[axis]["risky"], layer).mean(0)
    s = model.residual(axes[axis]["safe"], layer).mean(0)
    d = r - s
    return d / d.norm()


def inject(model, axes, axis, layer, beta):
    direction = risky_direction(model, axes, axis, layer)
    perturbed = model.clone()
    perturbed.add_mlp_out_bias(layer, beta * direction)
    return perturbed
