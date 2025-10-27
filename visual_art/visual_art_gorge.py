"""
python visual_art_gorge.py `
  --csv "A:\My Stuff\Projects\Python Stuff\Simulating Particals\gen\gan_100k.csv" `
  --out "collision_art_dense_distorted3.png" `
  --sym 6 --seed 11 --cmap magma `
  --distort_strength 0.006 --ripple_amp 0.003 --chaos_scale 0.002 `
  --progress
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import LineCollection

# -----------------------------
# Progress helpers
# -----------------------------
def _get_tqdm():
    try:
        from tqdm import tqdm  # pip install tqdm
        return tqdm
    except Exception:
        return None

def make_pbar(total, desc="", enabled=False, position=0):
    if not enabled:
        return None
    tqdm = _get_tqdm()
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, ncols=88, leave=False, position=position)

# -----------------------------
# Superformula (organic glyphs)
# -----------------------------
def superformula(theta, m=6, a=1.0, b=1.0, n1=1.0, n2=1.0, n3=1.0):
    t1 = (np.abs(np.cos(m*theta/4)/a))**n2
    t2 = (np.abs(np.sin(m*theta/4)/b))**n3
    r = (t1 + t2) ** (-1.0 / n1)
    return r

# -----------------------------
# Distortions
# -----------------------------
def distort_vertices(xs, ys, strength=0.004, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    xs = xs + rng.normal(0.0, strength, len(xs))
    ys = ys + rng.normal(0.0, strength, len(ys))
    return xs, ys

def ripple_vertices(xs, ys, freq=7, amp=0.002, phase=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    t = np.linspace(0, 2*np.pi, len(xs), endpoint=False)
    if phase is None:
        phase = rng.uniform(0, 2*np.pi)
    xs = xs + amp * np.sin(freq * t + phase)
    ys = ys + amp * np.cos(freq * t + phase)
    return xs, ys

def lorenz_step(x, y, z, s=10.0, r=28.0, b=8/3, dt=0.01):
    dx = s*(y - x)
    dy = r*x - y - x*z
    dz = x*y - b*z
    return x + dx*dt, y + dy*dt, z + dz*dt

def chaos_warp(xs, ys, scale=0.0015, steps=1, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x_c, y_c, z_c = rng.normal(0, 0.2), rng.normal(0, 0.2), rng.normal(0, 0.2)
    for _ in range(steps):
        x_c, y_c, z_c = lorenz_step(x_c, y_c, z_c, dt=0.01)
    return xs + scale*x_c, ys + scale*y_c

# -----------------------------
# Renderer
# -----------------------------
def render(csv_path, out_path, sym=5, seed=7, figsize=10, dpi=300, cmap_name="inferno",
           distort_strength=0.004, ripple_amp=0.002, ripple_freq=7,
           chaos_scale=0.0015, chaos_steps=1,
           line_share=0.25, star_share=0.20, super_share=0.25, poly_share=0.30,
           progress=False):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(csv_path)
    for c in ["pt_GeV","eta","phi_rad","m_GeV"]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {csv_path}.")
    pt = df["pt_GeV"].values.astype(float)
    eta = df["eta"].values.astype(float)
    phi = df["phi_rad"].values.astype(float)
    mass = df["m_GeV"].values.astype(float)

    norm = lambda x: (x - np.min(x)) / (np.ptp(x) + 1e-12)
    pt_n, eta_n, phi_n, mass_n = map(norm, (pt, eta, phi, mass))
    N = len(pt_n)

    # base layout
    theta = 2*np.pi*phi_n
    r = 0.35 + 0.6*pt_n + 0.15*theta/(2*np.pi)
    x = r*np.cos(theta); y = r*np.sin(theta)

    # symmetry tiling
    X = []; Y = []; idx = []
    for j in range(sym):
        ang = 2*np.pi*j/sym
        cj, sj = np.cos(ang), np.sin(ang)
        X.append(cj*x - sj*y); Y.append(sj*x + cj*y); idx.append(np.arange(N))
    X = np.concatenate(X); Y = np.concatenate(Y); idx = np.concatenate(idx)

    types = ["line","star","super","poly"]
    probs = np.array([line_share, star_share, super_share, poly_share], float)
    probs /= probs.sum()
    glyph_type = rng.choice(types, size=len(X), p=probs)

    fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
    ax = plt.gca(); ax.axis("off"); ax.set_aspect("equal")
    ax.set_facecolor("white")
    cmap = plt.get_cmap(cmap_name)

    colors = cmap(0.10 + 0.85*eta_n[idx])
    base_alpha = 0.10 + 0.80*(0.3 + 0.7*r[idx])
    base_lw = 0.25 + 2.5*mass_n[idx]

    # lines batch
    sel = glyph_type=="line"
    if np.any(sel):
        L = 0.012 + 0.040*pt_n[idx][sel]
        ang = theta[idx][sel]
        x0, y0 = X[sel], Y[sel]
        dx, dy = L*np.cos(ang), L*np.sin(ang)
        segs = []; cols = []; lws = []
        pbar_lines = make_pbar(len(x0), desc="lines", enabled=progress, position=0)
        for i in range(len(x0)):
            xa, ya = x0[i] - dx[i]/2.0, y0[i] - dy[i]/2.0
            xb, yb = x0[i] + dx[i]/2.0, y0[i] + dy[i]/2.0
            xa, ya = chaos_warp(np.array([xa]), np.array([ya]), scale=chaos_scale, steps=chaos_steps, rng=rng)
            xb, yb = chaos_warp(np.array([xb]), np.array([yb]), scale=chaos_scale, steps=chaos_steps, rng=rng)
            segs.append([[xa.item(), ya.item()], [xb.item(), yb.item()]])
            cols.append((colors[sel][i][0], colors[sel][i][1], colors[sel][i][2], base_alpha[sel][i]))
            lws.append(base_lw[sel][i])
            if pbar_lines: pbar_lines.update(1)
        lc = LineCollection(np.array(segs), linewidths=lws, colors=cols, capstyle="round", joinstyle="round")
        ax.add_collection(lc)
        if pbar_lines: pbar_lines.close()

    # glyphs
    pbar_glyphs = make_pbar(len(X), desc="glyphs", enabled=progress, position=1)
    for i in range(len(X)):
        t = glyph_type[i]
        rot = theta[idx][i]
        col = colors[i]; col = (col[0], col[1], col[2], base_alpha[i])

        if t == "poly":
            n = 3 + int(5*mass_n[idx][i])
            R = 0.012 + (6 + 28*pt_n[idx][i])/3500.0
            angs = rot + np.linspace(0, 2*np.pi, n, endpoint=False)
            xs = X[i] + R*np.cos(angs); ys = Y[i] + R*np.sin(angs)

        elif t == "star":
            n = 5 + int(4*mass_n[idx][i])
            R1 = 0.012 + (6 + 28*pt_n[idx][i])/3500.0
            R2 = R1 * (0.38 + 0.18*np.clip(1-pt_n[idx][i], 0, 1))
            xs=[]; ys=[]
            for j in range(2*n):
                ang = rot + j*np.pi/n
                rr = R1 if j%2==0 else R2
                xs.append(X[i] + rr*np.cos(ang)); ys.append(Y[i] + rr*np.sin(ang))
            xs, ys = np.array(xs), np.array(ys)

        elif t == "super":
            m = 3 + int(10*eta_n[idx][i])
            n1 = 0.3 + 1.6*mass_n[idx][i]
            n2 = 0.3 + 1.6*pt_n[idx][i]
            n3 = 0.3 + 1.6*(1-pt_n[idx][i])
            th = np.linspace(0, 2*np.pi, 220, endpoint=False)
            r_sup = superformula(th, m=m, n1=n1, n2=n2, n3=n3)
            R = 0.006 + (6 + 28*pt_n[idx][i])/5000.0
            xs = X[i] + R*r_sup*np.cos(th + rot)
            ys = Y[i] + R*r_sup*np.sin(th + rot)

        else:
            if pbar_glyphs: pbar_glyphs.update(1)
            continue

        # distort
        xs, ys = distort_vertices(xs, ys, strength=distort_strength, rng=rng)
        xs, ys = ripple_vertices(xs, ys, freq=ripple_freq, amp=ripple_amp, rng=rng)
        xs, ys = chaos_warp(xs, ys, scale=chaos_scale, steps=chaos_steps, rng=rng)

        # draw
        xs = np.append(xs, xs[0]); ys = np.append(ys, ys[0])
        codes = [Path.MOVETO] + [Path.LINETO]*(len(xs)-2) + [Path.CLOSEPOLY]
        ax.add_patch(PathPatch(Path(np.column_stack([xs, ys]), codes),
                               fill=False, ec=col, lw=base_lw[i], joinstyle="round"))
        if pbar_glyphs: pbar_glyphs.update(1)
    if pbar_glyphs: pbar_glyphs.close()

    ax.set_xlim(X.min()-0.03, X.max()+0.03)
    ax.set_ylim(Y.min()-0.03, Y.max()+0.03)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=dpi)
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", dest="csv", default="gan_20k_physics.csv")
    p.add_argument("--out", dest="out", default="visual_art.png")
    p.add_argument("--layers", type=int, default=5)          # kept for compat
    p.add_argument("--sym", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--figsize", type=float, default=10.0)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--cmap", default="inferno")
    p.add_argument("--distort_strength", type=float, default=0.004)
    p.add_argument("--ripple_amp", type=float, default=0.002)
    p.add_argument("--ripple_freq", type=int, default=7)
    p.add_argument("--chaos_scale", type=float, default=0.0015)
    p.add_argument("--chaos_steps", type=int, default=1)
    p.add_argument("--line_share", type=float, default=0.25)
    p.add_argument("--star_share", type=float, default=0.20)
    p.add_argument("--super_share", type=float, default=0.25)
    p.add_argument("--poly_share", type=float, default=0.30)
    p.add_argument("--progress", action="store_true", help="Show real-time progress bars (tqdm)")
    args = p.parse_args()

    render(csv_path=args.csv, out_path=args.out, sym=args.sym, seed=args.seed,
           figsize=args.figsize, dpi=args.dpi, cmap_name=args.cmap,
           distort_strength=args.distort_strength, ripple_amp=args.ripple_amp, ripple_freq=args.ripple_freq,
           chaos_scale=args.chaos_scale, chaos_steps=args.chaos_steps,
           line_share=args.line_share, star_share=args.star_share,
           super_share=args.super_share, poly_share=args.poly_share,
           progress=args.progress)

if __name__ == "__main__":
    main()
