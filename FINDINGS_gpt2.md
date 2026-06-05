# Findings: porting the pipeline from a toy model to pretrained GPT-2

This note documents what happened when the audit pipeline that works cleanly on the from-scratch
**TinyGPT** toy was run, unchanged, on **pretrained GPT-2 (124M)** via the `--model gpt2` adapter.
The short version: the *mechanism* (probe, weight/activation injection, steering hook, ARR) ports
and runs, but the *single-direction linear injection* that produces clean recovery on the toy does
**not** selectively induce a target behaviour on the real model. The failure modes below are the
substance of this note — they are the kind of thing that only shows up on a real model, and they
are reported honestly rather than tuned away.

All GPT-2 numbers are from an interactive Colab session (CPU, `gpt2`, seed 0); the TinyGPT numbers
are the committed `results/` from this repo.

## Baseline: TinyGPT (toy) works

On the 4-layer/96-dim from-scratch model trained on the synthetic 4-axis corpus, the full loop
behaves as designed:

- Leave-one-pair-out probe generalization: deception 66.7 %, power_seeking 62.5 %, honesty 62.5 %,
  harmlessness 62.5 % (above chance, noisy — small folds).
- Per-axis risk control: inject → drift → steer recovers ~70 % of the drift on every axis
  (ARR 70.3 / 70.4 / 71.2 / 70.6 %), no overshoot.
- Cross-axis transfer matrix is diagonal-dominant (concept-specific probes).

This is a *toy demonstration* — the clean result turns out to be partly a function of the model's
small capacity, as the GPT-2 port makes clear.

## GPT-2 (real): the adapter runs, but three things break

The `GPT2Audit` adapter loads real weights, extracts genuine hidden states, and the steering hook
demonstrably moves representations (sanity check: steering delta `7.93`, probe register separation
`+3.806`). So the plumbing is correct. The problems are in the *method*, not the code.

### 1. Weight-bias injection collapses the representation norm

The toy's failure injection edits `mlp.c_proj.bias`. On GPT-2 this does not push the
representation toward the risky register — it crushes the overall activation norm:

| injection (β, layer 6, weight) | risky score | safe score | safe-pop norm |
| --: | --: | --: | --: |
| clean | 111.09 | 107.28 | 210.3 |
| β = 50 | −17.06 | −16.38 | 182.3 |
| β = 100 | −46.83 | −46.15 | 125.8 |
| β = 200 | −64.32 | −65.62 | 84.8 |

Risky and safe scores move **together** and the norm falls from 210 → 85. The probe projection is
dominated by this common-mode norm change, not by any register signal.

**Mitigation that helped:** centering. Subtracting the corpus-mean representation before building
and applying the probe removes the giant common offset (baseline alignment goes from ~±107 to
`+2.95`) and makes the measurement stable. Centering fixed the *measurement*; it did not fix the
*injection*.

### 2. Residual-space and final-hidden-space directions are not sign-consistent

The injection/steering directions are defined in a block's residual stream (layer 6), but the
probe measures the final hidden state (after layers 6→12 and the final layer norm). On the toy
these happened to align; on GPT-2 they do not. Injecting the "risky" residual direction moved the
centered alignment score the *wrong way* (it rose instead of falling), and an automatic
sign-alignment heuristic did not reliably fix it, because the effective sign varied across prompts
and across the pooled population.

### 3. A single linear direction does not *selectively* induce the concept

This is the decisive one. Switching to activation-space injection (a forward hook adding the
risky direction at layer 6) removed the norm collapse, but revealed that the injection erases the
register distinction rather than amplifying the risky pole:

| injection (β, layer 6, activation) | risky score | safe score | **risky − safe gap** |
| --: | --: | --: | --: |
| clean | +0.68 | −3.12 | **+3.80** |
| β = 10 | −0.41 | −3.68 | +3.27 |
| β = 20 | −2.61 | −4.95 | +2.34 |
| β = 40 | −10.83 | −11.02 | **+0.18** |

Adding more of the "risky" direction pulls risky **and** safe representations down together, and
the gap the probe actually measures *shrinks* from 3.80 to 0.18. The intervention is not turning
deception on — it is flattening the representation and washing out the very distinction the probe
relies on. A single contrastive direction added at one layer is not a selective behavioural knob
on this model.

## Interpretation

The clean TinyGPT result was, in part, an artifact of small model capacity: in a tiny model a
single difference-of-means direction is close to a behavioural control, so inject→steer→recover is
tidy. In a pretrained model the same direction is entangled with high-norm common-mode structure
and with the residual→output transformation, so naive single-direction injection does not
selectively induce a concept, and adding more of it degrades rather than steers.

This is a known hard problem in representation engineering, not a bug in this repo. It is exactly
why the real techniques in this space use more than a raw difference-of-means added at one site.

## What this changes about the project's claims

- The pipeline is honestly a **toy demonstration of the mechanism**. It does **not** currently
  show selective behavioural control on a pretrained model, and the README should not claim it.
- The transferable contributions are the *methodology and the diagnostics*: a centered linear
  probe, a per-axis multi-dimensional audit, generalization tests (LOO-CV + cross-axis), an
  output-level behaviour check (KL), and — most usefully — a concrete catalogue of how a toy
  intervention fails on a real model.

## Next directions (to make the GPT-2 audit actually work)

- **Layer sweep:** inject and read at multiple layers; the effective layer for selective control
  is usually a middle band, not an arbitrary choice.
- **Projection / mean-ablation:** remove the common-mode component explicitly (project it out)
  before injecting, so the intervention acts only in the register subspace.
- **Stronger steering methods:** CAA / RepE-style multi-direction or per-token steering rather
  than one pooled difference-of-means added at a single site.
- **Behaviour-grounded direction finding:** derive the direction from contrasts that actually
  change generations (not just pooled hidden states), and validate with output KL on held-out
  prompts.
- **Selectivity metric:** track the risky−safe *gap*, not just absolute scores, so a "successful"
  injection is one that widens the gap, not one that moves both poles.

*These findings are recorded as-is. Reproducing them needs only the `GPT2Audit` backend in
`model.py`; the measurement fix (centering) is in `probes.py`.*
