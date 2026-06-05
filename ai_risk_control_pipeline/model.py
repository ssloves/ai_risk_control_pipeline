"""Models behind one interface (`AuditModel`) for the multi-axis risk audit.

TinyGPT  : from-scratch GPT-2-style transformer, trained offline on risk_dimensions.json.
           A *toy demonstration* — used to validate the pipeline mechanics end-to-end offline.
GPT2     : HuggingFace pretrained `gpt2` adapter (real model audit). Runs where transformers +
           network are available (e.g. Colab); not exercised in the offline build.

Adds behavioral methods (`seq_logprob`, `next_token_logits`) so we can ask not only
"did the representation move?" (probe) but "did the model's output behaviour change?".
"""
import copy, json, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

BLOCK = 18


def load_axes(path):
    return json.load(open(path))


def augment(sents, k=2, rng=None):
    rng = rng or random.Random(0)
    fillers = ["really", "always", "indeed", "for sure", "right now", "okay"]
    out = []
    for s in sents:
        out.append(s)
        for _ in range(k):
            out.append(s + " " + rng.choice(fillers))
    return out


# ----------------------------------------------------------------------------- TinyGPT
class _Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        T = x.size(1)
        cm = torch.triu(torch.ones(T, T, device=x.device) * float("-inf"), diagonal=1)
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=cm, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class _TinyGPT(nn.Module):
    def __init__(self, vocab, d=128, h=4, n=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(BLOCK, d)
        self.blocks = nn.ModuleList([_Block(d, h) for _ in range(n)])
        self.lnf = nn.LayerNorm(d); self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, idx, steer=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.size(1), device=idx.device))[None]
        hs = []
        for li, blk in enumerate(self.blocks):
            x = blk(x)
            if steer is not None and steer[0] == li:
                x = x + steer[1][None, None, :]
            hs.append(x)
        xf = self.lnf(x)
        return self.head(xf), xf, hs


class AuditModel:
    n_layers: int; hidden_dim: int
    def represent(self, texts, steer=None): ...
    def residual(self, texts, layer): ...
    def clone(self): ...
    def add_mlp_out_bias(self, layer, vec): ...
    def seq_logprob(self, text, steer=None): ...
    def generate(self, text, n=8): ...


class TinyGPTAudit(AuditModel):
    def __init__(self, axes, seed=0, steps=1500, lr=3e-3):
        self.seed = seed
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.axes = axes
        rng = random.Random(seed)
        every = [s for ax in axes for pole in ("risky", "safe") for s in axes[ax][pole]]
        train_text = []
        for ax in axes:
            for pole in ("risky", "safe"):
                train_text += augment(axes[ax][pole], rng=rng)
        words = sorted(set(" ".join(train_text + every).split()))
        self.stoi = {w: i + 1 for i, w in enumerate(words)}; self.stoi["<pad>"] = 0
        self.itos = {i: w for w, i in self.stoi.items()}
        self.vocab = len(self.stoi)
        self.net = _TinyGPT(self.vocab)
        self.n_layers = len(self.net.blocks)
        self.hidden_dim = self.net.lnf.normalized_shape[0]
        self._train(train_text, steps, lr)

    def _enc(self, s): return [self.stoi[w] for w in s.split() if w in self.stoi][:BLOCK]
    def _pad(self, ids): return ids + [0] * (BLOCK - len(ids))

    def _train(self, text, steps, lr):
        torch.manual_seed(self.seed)
        data = torch.tensor([self._pad(self._enc(s)) for s in text if len(self._enc(s)) >= 2], dtype=torch.long)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr); self.net.train()
        last = None
        for _ in range(steps):
            b = data[torch.randint(0, data.size(0), (32,))]
            lg, _, _ = self.net(b[:, :-1])
            loss = F.cross_entropy(lg.reshape(-1, self.vocab), b[:, 1:].reshape(-1), ignore_index=0)
            opt.zero_grad(); loss.backward(); opt.step(); last = float(loss.item())
        self.net.eval(); self.final_loss = last

    @torch.no_grad()
    def _pool(self, texts, which, layer=None, steer=None):
        out = []
        for s in texts:
            ids = torch.tensor([self._pad(self._enc(s))], dtype=torch.long)
            mask = (ids != 0).float()[..., None]
            _, final, hid = self.net(ids, steer=steer)
            h = final if which == "final" else hid[layer]
            out.append(((h * mask).sum(1) / mask.sum(1).clamp(min=1)).squeeze(0))
        return torch.stack(out)

    def represent(self, texts, steer=None): return self._pool(texts, "final", steer=steer)
    def residual(self, texts, layer): return self._pool(texts, "resid", layer=layer)

    def clone(self):
        c = copy.copy(self); c.net = copy.deepcopy(self.net); return c

    def add_mlp_out_bias(self, layer, vec):
        with torch.no_grad():
            self.net.blocks[layer].mlp[2].bias.add_(vec)

    @torch.no_grad()
    def seq_logprob(self, text, steer=None):
        ids = self._enc(text)
        if len(ids) < 2: return 0.0
        x = torch.tensor([ids], dtype=torch.long)
        logits, _, _ = self.net(x, steer=steer)
        logp = F.log_softmax(logits[0, :-1], dim=-1)
        tgt = torch.tensor(ids[1:])
        return float(logp[torch.arange(len(tgt)), tgt].mean())

    @torch.no_grad()
    def next_token_logits(self, text, steer=None):
        ids = self._enc(text) or [0]
        x = torch.tensor([ids], dtype=torch.long)
        logits, _, _ = self.net(x, steer=steer)
        return logits[0, len(ids) - 1]

    @torch.no_grad()
    def generate(self, text, n=8):
        ids = self._enc(text)
        for _ in range(n):
            x = torch.tensor([ids[-BLOCK:]], dtype=torch.long)
            nxt = int(self.net(x)[0][0, -1].argmax())
            if nxt == 0: break
            ids.append(nxt)
            if len(ids) >= BLOCK: break
        return " ".join(self.itos.get(i, "?") for i in ids)


# ----------------------------------------------------------------------------- GPT-2 adapter
class GPT2Audit(AuditModel):
    """Pretrained gpt2 via HuggingFace. Requires transformers + model download (e.g. Colab).

    Mirrors TinyGPTAudit's interface so the audit modules are unchanged. Hidden states come from
    the transformer core (`last_hidden_state` for the final post-LN state, `hidden_states[layer+1]`
    for a block's residual). Steering is a forward hook on `core.h[layer]` (GPT2Block returns a
    tuple whose [0] is the residual stream). Injection edits `core.h[layer].mlp.c_proj.bias`.
    NOT exercised in the offline build — sanity-check shapes/hook on first Colab run.
    """
    def __init__(self, seed=0, model_name="gpt2"):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        torch.manual_seed(seed)
        self.tok = GPT2Tokenizer.from_pretrained(model_name)
        self.lm = GPT2LMHeadModel.from_pretrained(model_name).eval()
        self.core = self.lm.transformer          # GPT2Model
        self.n_layers = self.lm.config.n_layer
        self.hidden_dim = self.lm.config.n_embd

    def _enc(self, text):
        return self.tok(text, return_tensors="pt")

    def _with_hook(self, steer):
        """context-manager-like: returns a handle (or None) adding `steer` at core.h[layer]."""
        if steer is None:
            return None
        layer, vec = steer
        def fn(module, inp, out):
            if isinstance(out, tuple):
                return (out[0] + vec.to(out[0].dtype),) + tuple(out[1:])
            return out + vec.to(out.dtype)
        return self.core.h[layer].register_forward_hook(fn)

    @torch.no_grad()
    def _pool(self, texts, which, layer=None, steer=None):
        h = self._with_hook(steer)
        try:
            out = []
            for s in texts:
                res = self.core(**self._enc(s), output_hidden_states=True)
                hs = res.last_hidden_state if which == "final" else res.hidden_states[layer + 1]
                out.append(hs.mean(1).squeeze(0))
            return torch.stack(out)
        finally:
            if h is not None:
                h.remove()

    def represent(self, texts, steer=None): return self._pool(texts, "final", steer=steer)
    def residual(self, texts, layer): return self._pool(texts, "resid", layer=layer)

    def clone(self):
        c = copy.copy(self); c.lm = copy.deepcopy(self.lm); c.core = c.lm.transformer; return c

    def add_mlp_out_bias(self, layer, vec):
        with torch.no_grad():
            self.core.h[layer].mlp.c_proj.bias.add_(vec)

    @torch.no_grad()
    def seq_logprob(self, text, steer=None):
        h = self._with_hook(steer)
        try:
            enc = self._enc(text); ids = enc["input_ids"]
            logits = self.lm(**enc).logits
            logp = F.log_softmax(logits[0, :-1], dim=-1)
            tgt = ids[0, 1:]
            return float(logp[torch.arange(len(tgt)), tgt].mean())
        finally:
            if h is not None:
                h.remove()

    @torch.no_grad()
    def next_token_logits(self, text, steer=None):
        h = self._with_hook(steer)
        try:
            return self.lm(**self._enc(text)).logits[0, -1]
        finally:
            if h is not None:
                h.remove()

    @torch.no_grad()
    def generate(self, text, n=12):
        enc = self._enc(text)
        out = self.lm.generate(**enc, max_new_tokens=n, do_sample=False,
                               pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0], skip_special_tokens=True)


def build_model(kind, axes_path, seed=0):
    axes = load_axes(axes_path)
    if kind == "tinygpt": return TinyGPTAudit(axes, seed=seed), axes
    if kind == "gpt2": return GPT2Audit(seed=seed), axes
    raise ValueError(kind)
