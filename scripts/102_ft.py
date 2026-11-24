"""102 M4c FT-Transformer(PD, BCE): 연속 열·범주 열을 각각 d=32 토큰으로(연속은 x·w+b), CLS 토큰 + 2층 Transformer 인코더. 전처리는 102_nn.prep 재사용."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
import importlib; M = importlib.import_module("102_common"); NN = importlib.import_module("102_nn"); log = M.log
import numpy as np, torch, torch.nn as nn
DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"); D = 32

class FT(nn.Module):
    def __init__(s, nc, vocab):
        super().__init__(); s.w = nn.Parameter(torch.randn(nc, D) * 0.1); s.b = nn.Parameter(torch.zeros(nc, D)); s.emb = nn.ModuleList([nn.Embedding(int(v), D) for v in vocab]); s.cls = nn.Parameter(torch.zeros(1, 1, D))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 4, 64, 0.1, batch_first=True, norm_first=True, activation="gelu"), 2); s.out = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 1))
    def forward(s, c, k):
        t = torch.cat([s.cls.expand(len(c), -1, -1), c.unsqueeze(2) * s.w + s.b] + [e(k[:, i]).unsqueeze(1) for i, e in enumerate(s.emb)], 1); return s.out(s.enc(t)[:, 0]).squeeze(1)

def m_ft(tr, te, Xtr, Xte, epochs=6, bs=1024):
    ca, cb, ka, kb, vocab = NN.prep(Xtr, Xte); y = M.y_all[tr]; torch.manual_seed(0); rs = np.random.RandomState(0); va = rs.rand(len(ca)) < 0.10
    T = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt)
    C, Kc, Y = T(ca), T(ka, torch.long), T(y); net = FT(ca.shape[1], vocab).to(DEV); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    tri = np.where(~va)[0]; vai = np.where(va)[0]; sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(tri) // bs + 1)); bce = nn.BCEWithLogitsLoss(); best = (1e9, None)
    pred = lambda c, k: torch.cat([net(c[i:i + 8192].to(DEV), k[i:i + 8192].to(DEV)).cpu() for i in range(0, len(c), 8192)])
    for ep in range(epochs):
        net.train(); rs.shuffle(tri)
        for i in range(0, len(tri), bs):
            j = torch.as_tensor(tri[i:i + bs]); loss = bce(net(C[j].to(DEV), Kc[j].to(DEV)), Y[j].to(DEV)); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        net.eval()
        with torch.no_grad(): vl = bce(pred(C[vai], Kc[vai]), Y[vai]).item()
        log("  ep", ep, "val bce %.4f" % vl)
        if vl < best[0]: best = (vl, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad(): p = torch.sigmoid(pred(T(cb), T(kb, torch.long))).numpy().astype(np.float64)
    return {"M4c": (p, p)}
