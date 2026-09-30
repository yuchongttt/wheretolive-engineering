import json, glob, os, sys
import numpy as np
SP = os.path.dirname(os.path.abspath(__file__))
rows = []
for t in (1, 2, 3, 4):
    f = f'{SP}/results/lobo_test{t}.json'
    if not os.path.exists(f): print('missing fold', t); continue
    d = json.load(open(f)); s = d['summary']
    print(f"fold{t} n={s['n']} med={s['median_err']:.1f} acc22={s['acc22']:.3f} flips={s['flips']} gate15={s['gates']['15']['yield_']:.3f}@{s['gates']['15']['prec22']:.3f}")
    rows += [dict(x, fold=t) for x in d['results']]
e = np.array([x['err'] for x in rows]); sp = np.array([x['spread'] for x in rows]); n = len(rows)
print(f"POOLED n={n} median_err={np.median(e):.1f} acc15={(e<=15).mean():.3f} acc22={(e<=22.5).mean():.3f} acc30={(e<=30).mean():.3f} flips(>=150)={(e>=150).sum()}")
for tau in (5, 10, 15, 20, 30):
    k = sp <= tau; ek = e[k]
    print(f"  gate spread<={tau:>2}: yield {k.mean():.3f} ({k.sum()}/{n})  prec22 {(ek<=22.5).mean():.3f}  prec30 {(ek<=30).mean():.3f}  errors>22.5: {[(x['pid'],x['err']) for x,kk in zip(rows,k) if kk and x['err']>22.5]}")
# 8-way correctness among gate<=15
def bin8(a): return int(((a + 22.5) % 360) // 45)
k = sp <= 15; same = [bin8(x['truth']) == bin8(x['pred']) for x, kk in zip(rows, k) if kk]
print(f"  8-way exact among gate<=15: {np.mean(same):.3f} ({sum(same)}/{len(same)})")
