#!/usr/bin/env python3
"""Compass-reader spike: small CNN trained on synthetic rotations of real seed crops, evaluated on a held-out batch of REAL crops.
Label convention: deg = clockwise angle from 'up' of the compass north (clock*30). PIL rotate(theta) is CCW, so label' = (deg - theta) mod 360."""
import argparse, csv, json, math, os, random, time
import numpy as np, torch, torch.nn as nn
from PIL import Image, ImageDraw, ImageFilter
import torchvision
SP = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--train', default='1,2,3'); ap.add_argument('--test', default='4')
ap.add_argument('--epochs', type=int, default=25); ap.add_argument('--size', type=int, default=160)
ap.add_argument('--per-seed', type=int, default=48); ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--exclude', default=''); ap.add_argument('--out', default=f'{SP}/reader_run')
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--arch', default='resnet18')
args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
excl = set(open(args.exclude).read().split()) if args.exclude and os.path.exists(args.exclude) else set()
rows = [r for r in csv.DictReader(open(f'{SP}/seeds.tsv'), delimiter='\t') if r['pid'] not in excl]
for r in rows: r['deg'] = float(r['deg']); r['batch'] = int(r['batch'])
tr = [r for r in rows if r['batch'] in {int(x) for x in args.train.split(',')}]
te = [r for r in rows if r['batch'] in {int(x) for x in args.test.split(',')}]
print(f'train seeds {len(tr)}  test seeds {len(te)}  excluded {len(excl)}  device {dev}', flush=True)
CACHE = {r['pid']: Image.open(f"{SP}/seeds/{r['pid']}.png").convert('L') for r in rows}
MEAN, STD = 0.449, 0.226

def synth(img, deg, size, train=True):
    """returns (tensor, deg') ; random rotation + scale/offset jitter + clutter + photometric."""
    S = 480; canvas = Image.new('L', (S, S), 255); canvas.paste(img, ((S - 320) // 2, (S - 320) // 2))
    if train:
        theta = random.uniform(0, 360)
        canvas = canvas.rotate(theta, resample=Image.BICUBIC, fillcolor=255)
        deg = (deg - theta) % 360
        d = ImageDraw.Draw(canvas)
        for _ in range(random.randint(0, 5)):  # clutter: plan lines / text-ish bars away from centre
            x0, y0 = random.randint(0, S), random.randint(0, S)
            if math.hypot(x0 - S / 2, y0 - S / 2) < 110: continue
            if random.random() < 0.6:
                d.line([(x0, y0), (x0 + random.randint(-200, 200), y0 + random.randint(-200, 200))], fill=random.randint(0, 90), width=random.randint(1, 4))
            else:
                d.rectangle([x0, y0, x0 + random.randint(10, 80), y0 + random.randint(4, 12)], fill=random.randint(0, 120))
        side = random.uniform(210, 400); ox, oy = random.uniform(-30, 30), random.uniform(-30, 30)
    else:
        side, ox, oy = 320, 0, 0
    cx, cy = S / 2 + ox, S / 2 + oy
    crop = canvas.crop((int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))).resize((size, size), Image.BICUBIC)
    if train:
        if random.random() < 0.3: crop = crop.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
        a = np.asarray(crop, dtype=np.float32) / 255.0
        a = np.clip((a - 0.5) * random.uniform(0.7, 1.3) + 0.5 + random.uniform(-0.1, 0.1), 0, 1)
        if random.random() < 0.3: a = np.clip(a + np.random.normal(0, 0.03, a.shape), 0, 1)
    else:
        a = np.asarray(crop, dtype=np.float32) / 255.0
    t = torch.from_numpy(((a - MEAN) / STD).astype(np.float32))[None].repeat(3, 1, 1)
    return t, deg

class DS(torch.utils.data.Dataset):
    def __init__(self, rows, n, train): self.rows, self.n, self.train = rows, n, train
    def __len__(self): return len(self.rows) * self.n
    def __getitem__(self, i):
        r = self.rows[i % len(self.rows)]
        t, d = synth(CACHE[r['pid']], r['deg'], args.size, self.train)
        a = math.radians(d); return t, torch.tensor([math.cos(a), math.sin(a)], dtype=torch.float32)

def build():
    if args.arch == 'resnet18':
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Linear(512, 2)
    else:
        m = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1); m.fc = nn.Linear(512, 2)
    return m.to(dev)

def ang_of(v): return (torch.rad2deg(torch.atan2(v[:, 1], v[:, 0])) % 360)
def cdiff(a, b): d = abs(a - b) % 360; return min(d, 360 - d)

@torch.no_grad()
def predict_tta(model, img):
    """4x90 rotation TTA; unrotate; circular mean + spread (max pairwise circ diff)."""
    model.eval(); preds = []
    for k in range(4):
        rot = img.rotate(90 * k, resample=Image.BICUBIC, fillcolor=255)   # CCW by 90k -> pointer deg' = deg - 90k
        t, _ = synth(rot, 0, args.size, train=False)
        v = model(t[None].to(dev)).cpu()
        preds.append((ang_of(v)[0].item() + 90 * k) % 360)
    r = np.radians(preds); mean = math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean())) % 360
    spread = max(cdiff(a, b) for a in preds for b in preds)
    return mean, spread, preds

def evaluate(model, rows, tag):
    res = []
    for r in rows:
        mean, spread, preds = predict_tta(model, CACHE[r['pid']])
        res.append(dict(pid=r['pid'], style=r['style'], truth=r['deg'], pred=round(mean, 1), err=round(cdiff(mean, r['deg']), 1), spread=round(spread, 1), preds=[round(p, 1) for p in preds]))
    errs = np.array([x['err'] for x in res]); n = len(res)
    out = dict(tag=tag, n=n, median_err=float(np.median(errs)), acc15=float((errs <= 15).mean()), acc22=float((errs <= 22.5).mean()), acc30=float((errs <= 30).mean()),
               flips=int(((errs >= 150)).sum()))
    gates = {}
    for tau in (5, 10, 15, 20, 30, 45):
        keep = [x for x in res if x['spread'] <= tau]
        if keep:
            k = np.array([x['err'] for x in keep]); gates[tau] = dict(yield_=len(keep) / n, prec22=float((k <= 22.5).mean()), prec30=float((k <= 30).mean()), n=len(keep))
    out['gates'] = gates
    return out, res

model = build()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
dl = torch.utils.data.DataLoader(DS(tr, args.per_seed, True), batch_size=args.bs, shuffle=True, num_workers=0)
steps = args.epochs * len(dl); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.1)
t0 = time.time()
for ep in range(args.epochs):
    model.train(); tot = 0; nb = 0
    for x, y in dl:
        x, y = x.to(dev), y.to(dev)
        v = model(x); v = v / (v.norm(dim=1, keepdim=True) + 1e-6)
        loss = (1 - (v * y).sum(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item(); nb += 1
    msg = f'ep {ep+1}/{args.epochs} loss {tot/nb:.4f} {time.time()-t0:.0f}s'
    if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
        o, _ = evaluate(model, te, f'ep{ep+1}'); msg += f" | test med_err {o['median_err']:.1f} acc22 {o['acc22']:.3f} flips {o['flips']}"
    print(msg, flush=True)
final, res = evaluate(model, te, 'final'); trn, _ = evaluate(model, tr, 'train-real')
final['train_real'] = {k: trn[k] for k in ('n', 'median_err', 'acc22', 'flips')}
json.dump(dict(summary=final, results=res, args=vars(args)), open(f'{args.out}/results.json', 'w'), indent=1, ensure_ascii=False)
torch.save(model.state_dict(), f'{args.out}/model.pt')
print(json.dumps(final, indent=1))
