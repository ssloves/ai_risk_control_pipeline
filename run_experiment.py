"""Multi-dimensional AI risk audit orchestrator.

  probe generalization  : leave-one-pair-out CV accuracy + cross-axis transfer matrix
  per-axis control       : inject a risk (weight edit) -> measure drift -> steer -> ARR
  multi-dim risk profile : injecting one axis -> 4-dim risk readout (spillover across axes)
  behavioural check      : output-level logprob preference + next-token KL (did behaviour change?)

Usage:
  python run_experiment.py                  # offline TinyGPT toy demonstration
  python run_experiment.py --model gpt2 --outdir results_gpt2 --inj-layer 6 --steer-layer 4

The audit is exposed as run_audit(model, axes, cfg, outdir) so a notebook can build a model
once (e.g. gpt2) and call it with different configs without reloading.
"""
import argparse, csv, json, os
import probes as P, weight_attack as WA, activation_steering as AS, behavior_eval as BE


def run_audit(model, axes, cfg, outdir="results", kl_betas=(2, 5, 10, 20, 40), verbose=True):
    os.makedirs(outdir, exist_ok=True)
    names = list(axes)
    config = dict(cfg, n_layers=model.n_layers, hidden_dim=model.hidden_dim,
                  final_loss=round(getattr(model, "final_loss", float("nan")), 4))
    inj_layer, steer_layer = cfg["inj_layer"], cfg["steer_layer"]
    beta, scale = cfg["beta"], cfg["steer_scale"]

    probes = {ax: P.build_probe(model, axes[ax]["risky"], axes[ax]["safe"]) for ax in names}

    # 1. generalization
    loo = P.loo_generalization(model, axes)
    xmat = P.cross_axis_matrix(model, axes, probes)
    json.dump(dict(config=config,
                   leave_one_pair_out_accuracy_pct={ax: {"mean": loo[ax][0], "std": loo[ax][1]} for ax in names},
                   note="held-out accuracy = probe built without each pair, classifying that unseen pair"),
              open(f"{outdir}/probe_generalization.json", "w"), indent=2)
    with open(f"{outdir}/cross_axis_transfer.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["probe\\axis"] + names)
        for pi in names:
            w.writerow([pi] + [xmat[pi][aj] for aj in names])

    safe_pop = [s for ax in names for s in axes[ax]["safe"]]
    align = lambda mdl, probe, steer=None: -float((mdl.represent(safe_pop, steer=steer) @ probe).mean())

    steering_summary, risk_profile, behavior = {}, {}, {}
    for axis in names:
        perturbed = WA.inject(model, axes, axis, inj_layer, beta)
        svec = AS.safe_direction(model, axes, axis, steer_layer)
        steer = AS.make_steer(steer_layer, svec, scale)

        b = align(model, probes[axis]); p = align(perturbed, probes[axis])
        s = align(perturbed, probes[axis], steer=steer)
        steering_summary[axis] = dict(baseline_alignment=round(b, 4), perturbed_alignment=round(p, 4),
                                      steered_alignment=round(s, 4), drift=round(b - p, 4),
                                      recovered=round(s - p, 4),
                                      ARR_pct=round(100 * AS.alignment_recovery_rate(b, p, s), 1))

        rp = {}
        for k in names:
            rp[k] = dict(
                baseline=round(float((model.represent(safe_pop) @ probes[k]).mean()), 3),
                perturbed=round(float((perturbed.represent(safe_pop) @ probes[k]).mean()), 3),
                steered=round(float((perturbed.represent(safe_pop, steer=steer) @ probes[k]).mean()), 3))
        risk_profile[axis] = rp

        kl_curve = {}
        for bs in kl_betas:
            kl_curve[bs] = round(BE.output_kl(model, WA.inject(model, axes, axis, inj_layer, bs), safe_pop), 4)
        behavior[axis] = dict(
            behavioral_risk_baseline=round(BE.behavioral_risk(model, axes, axis), 4),
            behavioral_risk_perturbed=round(BE.behavioral_risk(perturbed, axes, axis), 4),
            behavioral_risk_steered=round(BE.behavioral_risk(perturbed, axes, axis, steer=steer), 4),
            output_kl_baseline_vs_perturbed=round(BE.output_kl(model, perturbed, safe_pop), 4),
            output_kl_vs_injection_strength=kl_curve)

    json.dump(dict(config=config, axes=steering_summary), open(f"{outdir}/steering_results.json", "w"), indent=2)
    json.dump(dict(config=config, profile=risk_profile), open(f"{outdir}/risk_profile.json", "w"), indent=2)
    json.dump(dict(config=config, behavior=behavior), open(f"{outdir}/behavior_change.json", "w"), indent=2)

    if verbose:
        print("== probe generalization (leave-one-pair-out CV) ==")
        for ax in names: print(f"  {ax:14s} {loo[ax][0]:5.1f}% +/- {loo[ax][1]:.1f}")
        print("\n== per-axis risk control (ARR) ==")
        for ax in names:
            d = steering_summary[ax]
            print(f"  {ax:14s} align {d['baseline_alignment']:+.2f} -> inj {d['perturbed_alignment']:+.2f} "
                  f"-> steer {d['steered_alignment']:+.2f}   ARR={d['ARR_pct']}%")
        print(f"\nwrote 5 artifacts to {outdir}/")
    return dict(generalization=loo, cross_axis=xmat, steering=steering_summary,
                risk_profile=risk_profile, behavior=behavior, config=config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tinygpt", choices=["tinygpt", "gpt2"])
    ap.add_argument("--axes", default="risk_dimensions.json")
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--steer-scale", type=float, default=7.0)
    ap.add_argument("--inj-layer", type=int, default=1)
    ap.add_argument("--steer-layer", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    import model as M
    model, axes = M.build_model(a.model, a.axes, seed=a.seed)
    cfg = dict(model=a.model, seed=a.seed, beta=a.beta, steer_scale=a.steer_scale,
               inj_layer=a.inj_layer, steer_layer=a.steer_layer)
    run_audit(model, axes, cfg, outdir=a.outdir)


if __name__ == "__main__":
    main()
