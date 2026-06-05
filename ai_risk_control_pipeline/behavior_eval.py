"""Module — behavioural verification ("did behaviour actually change?").

The probe measures the *representation*. These metrics measure the model's *output*:

  behavioral_risk : mean seq-logprob the model assigns to RISKY sentences minus SAFE sentences.
                    Higher => the model's output distribution prefers risky continuations.
  output_kl       : mean next-token KL divergence between two models over a prompt set =>
                    how much the output distribution actually moved.

If injection raises behavioral_risk (and steering lowers it) in step with the probe, the internal
drift corresponds to a real behavioural change rather than a probe artefact.
"""
import torch, torch.nn.functional as F


def behavioral_risk(model, axes, axis, steer=None):
    r = [model.seq_logprob(s, steer=steer) for s in axes[axis]["risky"]]
    s = [model.seq_logprob(s, steer=steer) for s in axes[axis]["safe"]]
    return float(sum(r) / len(r) - sum(s) / len(s))


def output_kl(model_a, model_b, prompts):
    kls = []
    for p in prompts:
        la = F.log_softmax(model_a.next_token_logits(p), dim=-1)
        lb = F.log_softmax(model_b.next_token_logits(p), dim=-1)
        kls.append(float((la.exp() * (la - lb)).sum()))
    return float(sum(kls) / len(kls))
