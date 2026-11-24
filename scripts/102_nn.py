"""102 신경망: M4a 임베딩 MLP(부도 BCE), M4b 다중과제(부도 BCE + 수익 z Huber; PD 머리·수익 머리 두 점수). 결측: 학습 겹 중앙값 대치 + 표지(>1%) + 분위수 정규화."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; M = importlib.import_module("102_common"); B, K, log = M.B, M.K, M.log
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import QuantileTransformer
torch.set_num_threads(10)
CATC = lambda X: [c for c in X.columns if c in B.NOMINAL or c in B.ORD]

def prep(Xtr, Xte):
    cc = CATC(Xtr); cont = [c for c in Xtr.columns if c not in cc]; a, b = Xtr[cont].to_numpy(np.float32), Xte[cont].to_numpy(np.float32)
    med = np.nanmedian(a, 0); fl = np.where(np.isnan(a).mean(0) > 0.01)[0]
    ia, ib = np.isnan(a), np.isnan(b); a = np.where(ia, med, a); b = np.where(ib, med, b)
    q = QuantileTransformer(n_quantiles=256, output_distribution="normal", subsample=200000, random_state=0).fit(a)
    ca = np.hstack([q.transform(a), ia[:, fl]]).astype(np.float32); cb = np.hstack([q.transform(b), ib[:, fl]]).astype(np.float32)
    codes = lambda Z, ref=None: np.stack([np.where(Z[c].isna(), 0, Z[c].fillna(0).astype(int) + 1) for c in cc], 1)
    ka, kb = codes(Xtr), codes(Xte); vocab = ka.max(0) + 2; kb = np.minimum(kb, (vocab - 1)[None])
    return ca, cb, ka, kb, vocab

class Net(nn.Module):
    def __init__(s, nc, vocab, multi):
        super().__init__(); s.emb = nn.ModuleList([nn.Embedding(int(v), 8) for v in vocab]); d = nc + 8 * len(vocab)
        s.body = nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Dropout(0.15), nn.Linear(256, 128), nn.SiLU(), nn.Dropout(0.15), nn.Linear(128, 64), nn.SiLU())
        s.hp = nn.Linear(64, 1); s.hr = nn.Linear(64, 1) if multi else None
    def forward(s, c, k):
        h = s.body(torch.cat([c] + [e(k[:, i]) for i, e in enumerate(s.emb)], 1)); return s.hp(h).squeeze(1), (s.hr(h).squeeze(1) if s.hr is not None else None)

def _fit(ca, cb, ka, kb, vocab, ytr, ztr, wtr, multi, seed=0, epochs=10, bs=2048):
    torch.manual_seed(seed); rs = np.random.RandomState(seed); va = rs.rand(len(ca)) < 0.10; T = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt)
    C, Kc, Y, Zt, Wt = T(ca), T(ka, torch.long), T(ytr), T(ztr), T(wtr); net = Net(ca.shape[1], vocab, multi); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    tri = np.where(~va)[0]; vai = np.where(va)[0]; steps = epochs * (len(tri) // bs + 1); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=steps)
    bce = nn.BCEWithLogitsLoss(); best = (1e9, None)
    for ep in range(epochs):
        net.train(); rs.shuffle(tri)
        for i in range(0, len(tri), bs):
            j = torch.as_tensor(tri[i:i + bs]); lp, lr_ = net(C[j], Kc[j]); loss = bce(lp, Y[j])
            if multi: loss = loss + 0.5 * (nn.functional.huber_loss(lr_, Zt[j], reduction="none", delta=1.0) * Wt[j]).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        net.eval()
        with torch.no_grad(): lp, _ = net(C[vai], Kc[vai]); vl = bce(lp, Y[vai]).item()
        log("  ep", ep, "val bce %.4f" % vl)
        if vl < best[0]: best = (vl, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad(): lp, lr_ = net(T(cb), T(kb, torch.long))
    return torch.sigmoid(lp).numpy().astype(np.float64), (None if lr_ is None else lr_.numpy())

def m_mlp(tr, te, Xtr, Xte):
    ca, cb, ka, kb, vocab = prep(Xtr, Xte); y = M.y_all[tr]; z = M.Z_TR[tr]; zs = (z - z.mean()) / z.std(); w = M.A_all[tr] / M.A_all[tr].mean()
    pa, _ = _fit(ca, cb, ka, kb, vocab, y, zs, w, False); pb, rb = _fit(ca, cb, ka, kb, vocab, y, zs, w, True)
    return {"M4a": (pa, pa), "M4b_pd": (pb, pb), "M4b_ret": (-rb, None)}
