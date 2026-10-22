#!/usr/bin/env python3
"""Build a high-resolution preview of the shared Experiment 2–3 star maneuver.

Run from the repository root: python3 make_shared_formation_gif.py
This shows prescribed stations using configs/ecc.json, not simulated agent errors.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "shared_star_maneuver_preview.gif"


def main():
    cfg=json.loads((ROOT/'configs/ecc.json').read_text())['shared']
    P = np.asarray(cfg['template'],dtype=float)
    P -= P.mean(axis=0)
    n=len(P)
    a_initial, a_final, kappa = cfg['a_initial'],cfg['a_final'],cfg['kappa']
    omega, t_end = cfg['omega'],cfg['t_end']
    times = np.linspace(0, t_end, 96)

    def reference(t):
        r=cfg['reference']
        return np.array([r['speed'] * t, r['amplitude'] * np.sin(r['frequency'] * t)])

    def state(t):
        a = a_final + (a_initial - a_final) * np.exp(-kappa * t)
        theta = omega * t
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, s], [-s, c]])  # row-vector convention in the model
        r = reference(t)
        return a * (P @ R) + r, r, a, theta

    centers = np.array([reference(t) for t in times])
    history=np.array([state(t)[0] for t in times])
    all_q = history.reshape(-1,2)
    lo = np.minimum(all_q.min(axis=0), centers.min(axis=0)) - 3.3
    hi = np.maximum(all_q.max(axis=0), centers.max(axis=0)) + 3.3

    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=100)
    fig.subplots_adjust(left=.08, right=.98, bottom=.12, top=.91)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x position"); ax.set_ylabel("y position")
    ax.grid(alpha=.22)
    ax.set_title("A star formation contracts slowly while following a curved reference", fontsize=15)
    colors = plt.get_cmap("tab10")(np.arange(n))
    ax.plot(centers[:, 0], centers[:, 1], color="0.35", ls="--", lw=1.5,
            label="reference path")
    initial, _, _, _ = state(0)
    closed = np.r_[np.arange(n), 0]
    outline,=ax.plot(initial[closed, 0], initial[closed, 1], color="#374151", lw=1.6,
                    alpha=.9, label="moving star outline")
    trails = [ax.plot([], [], color=colors[i], lw=1.15, alpha=.58)[0] for i in range(n)]
    agents = ax.scatter(initial[:, 0], initial[:, 1], s=75, c=colors,
                        edgecolors="black", linewidths=.6, zorder=4,
                        label="prescribed stations")
    center, = ax.plot([], [], "k*", ms=13, zorder=5, label="reference / formation center")
    status = ax.text(.015, .975, "", transform=ax.transAxes, va="top", fontsize=11,
                     bbox=dict(facecolor="white", alpha=.86, edgecolor="none", pad=4))
    ax.legend(loc="upper right", fontsize=10, framealpha=.94)

    def update(i):
        t = times[i]
        q, r, a, theta = state(t)
        for j, line in enumerate(trails):
            hist = history[:i+1,j]
            line.set_data(hist[:, 0], hist[:, 1])
        agents.set_offsets(q)
        outline.set_data(q[closed,0],q[closed,1])
        center.set_data([r[0]], [r[1]])
        status.set_text(f"t = {t:4.1f} s     scale a(t) = {a:.2f}     rotation = {np.degrees(theta):.1f}°")
        return [*trails, agents, outline, center, status]

    anim = FuncAnimation(fig, update, frames=len(times), interval=80, blit=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUTPUT, writer=PillowWriter(fps=12), dpi=90,
              savefig_kwargs={"facecolor": "white"})
    plt.close(fig)
    print(f"Saved {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
