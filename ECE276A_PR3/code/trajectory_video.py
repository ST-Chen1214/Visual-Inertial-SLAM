"""
ECE 276A Project 3 — Trajectory Video  (blit-accelerated, OpenCV backend)
==========================================================================
Part 1 / Part 3 / Part 4 軌跡動畫，三個 panel 並排，一支影片。

預計速度: ~10ms/frame  →  dataset00 (step=3) 約 15 秒完成

使用方式:
    python trajectory_video.py                    # 全部三個 dataset
    python trajectory_video.py --dataset 00       # 單一 dataset
    python trajectory_video.py --fps 30 --step 4  # 調速度 (step 越大越快)

輸出:
    results/trajectory_video_dataset00.mp4
    results/trajectory_video_dataset01.mp4
    results/trajectory_video_dataset02.mp4
"""

import os, sys, argparse, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

# ─── Paths ──────────────────────────────────────────────────
CODE_DIR    = r"D:\ucsd\handouts\ece276a\ECE276A_PR3\ECE276A_PR3\code"
RESULTS_DIR = os.path.join(CODE_DIR, "results")

# ─── Style ──────────────────────────────────────────────────
BG_FIG  = '#0F172A'
BG_AX   = '#1E293B'
C_GRID  = '#334155'
C_TEXT  = '#CBD5E1'
C_P1    = '#3B82F6'   # blue    — Part 1
C_P3    = '#3B82F6'   # blue    — Part 3 (same traj)
C_P4    = '#A78BFA'   # purple  — Part 4
C_LM    = '#FCD34D'   # yellow  — landmarks
C_START = '#4ADE80'   # green   — start marker
C_ROBOT = '#FFFFFF'   # white   — robot dot


# ═══════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════

def load_data(dname, results_dir):
    out = {}
    # combined npz
    npz = os.path.join(results_dir, f"results_{dname}.npz")
    if os.path.exists(npz):
        d = dict(np.load(npz, allow_pickle=True))
        out.update(d)
        print(f"  [load] results_{dname}.npz  keys={list(d.keys())}")
    # individual npy fallbacks
    for key, fname in [
        ('T_imu',    f"T_imu_{dname}.npy"),
        ('T_slam',   f"T_slam_{dname}.npy"),
        ('mu_lm',    f"mu_lm_{dname}.npy"),
        ('init_lm',  f"init_lm_{dname}.npy"),
        ('mu_slam',  f"mu_slam_{dname}.npy"),
        ('init_slam',f"init_slam_{dname}.npy"),
    ]:
        if key not in out:
            p = os.path.join(results_dir, fname)
            if os.path.exists(p):
                out[key] = np.load(p, allow_pickle=True)
                print(f"  [load] {fname}")
    return out


def filter_lm(mu_m, init, traj_xy, margin=35.0):
    if mu_m is None or init is None or int(init.sum()) == 0:
        return np.zeros((2, 0))
    lm = mu_m[:2, init.astype(bool)]
    if lm.shape[1] == 0:
        return lm
    x0, x1 = traj_xy[:,0].min()-margin, traj_xy[:,0].max()+margin
    y0, y1 = traj_xy[:,1].min()-margin, traj_xy[:,1].max()+margin
    keep = (lm[0]>=x0)&(lm[0]<=x1)&(lm[1]>=y0)&(lm[1]<=y1)
    return lm[:, keep]


def ax_limits(xy_list, lm_list=None, pad=10.0):
    xs, ys = [], []
    for xy in xy_list:
        if xy is not None and len(xy):
            xs.append(xy[:,0]); ys.append(xy[:,1])
    if lm_list:
        for lm in lm_list:
            if lm is not None and lm.shape[1] > 0:
                xs.append(lm[0]); ys.append(lm[1])
    xs = np.concatenate(xs); ys = np.concatenate(ys)
    return (xs.min()-pad, xs.max()+pad), (ys.min()-pad, ys.max()+pad)


def style_ax(ax, title, xlim, ylim):
    ax.set_facecolor(BG_AX)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect('equal')
    ax.grid(True, color=C_GRID, linewidth=0.5, alpha=0.7, zorder=0)
    ax.set_xlabel('x [m]', fontsize=8, color=C_TEXT)
    ax.set_ylabel('y [m]', fontsize=8, color=C_TEXT)
    ax.set_title(title, fontsize=9, fontweight='bold', color='#F1F5F9', pad=5)
    ax.tick_params(colors=C_TEXT, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_GRID)


# ═══════════════════════════════════════════════════════════════
#  Main builder
# ═══════════════════════════════════════════════════════════════

def build_video(dname, results_dir, fps=30, step=3, dpi=80):
    print(f"\n{'─'*55}")
    print(f"  Building: {dname}")
    print(f"{'─'*55}")

    d = load_data(dname, results_dir)

    T_imu    = d.get('T_imu',    None)
    T_slam   = d.get('T_slam',   None)
    mu_lm    = d.get('mu_lm',    None)
    init_lm  = d.get('init_lm',  None)
    mu_slam  = d.get('mu_slam',  None)
    init_slam= d.get('init_slam', None)

    if T_imu is None:
        print(f"  [SKIP] No T_imu_{dname}.npy found.")
        print(f"         Run: python main.py --dataset {dname[-2:]} --part 1")
        return

    has_p3 = (mu_lm is not None and init_lm is not None)
    has_p4 = (T_slam is not None)

    xy1 = T_imu[:, :2, 3]
    xy3 = xy1
    xy4 = T_slam[:, :2, 3] if has_p4 else None

    lm3 = filter_lm(mu_lm,   init_lm,   xy3) if has_p3 else np.zeros((2,0))
    lm4 = filter_lm(mu_slam, init_slam, xy4)  if has_p4 else np.zeros((2,0))

    N      = len(T_imu)
    frames = list(range(0, N, step))
    trail  = max(80, N // 6)

    n_panels = 1 + int(has_p3) + int(has_p4)
    panel_titles = [f'Part 1 — IMU only  ({dname})']
    if has_p3: panel_titles.append(f'Part 3 — Landmark Mapping  ({dname})')
    if has_p4: panel_titles.append(f'Part 4 — VI-SLAM  ({dname})')

    xlim1, ylim1 = ax_limits([xy1], pad=10)
    xlim3, ylim3 = ax_limits([xy3], [lm3], pad=10) if has_p3 else (None,None)
    xlim4, ylim4 = ax_limits([xy4], [lm4], pad=10) if has_p4 else (None,None)

    # ── Build figure ─────────────────────────────────────────
    figw = 5.5 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(figw, 5.0), dpi=dpi)
    if n_panels == 1: axes = [axes]
    fig.patch.set_facecolor(BG_FIG)

    pidx = 0
    ax1 = axes[pidx]; pidx += 1
    ax3 = axes[pidx] if has_p3 else None; pidx += int(has_p3)
    ax4 = axes[pidx] if has_p4 else None

    style_ax(ax1, panel_titles[0], xlim1, ylim1)
    if has_p3: style_ax(ax3, panel_titles[1], xlim3, ylim3)
    if has_p4: style_ax(ax4, panel_titles[2], xlim4, ylim4)

    # Static elements (drawn once into background)
    ax1.scatter(*xy1[0], c=C_START, s=90, zorder=8, marker='*',
                edgecolors='white', linewidths=0.5)
    if has_p3:
        if lm3.shape[1] > 0:
            ax3.scatter(lm3[0], lm3[1], c=C_LM, s=3, alpha=0.55, zorder=3)
        ax3.scatter(*xy3[0], c=C_START, s=90, zorder=8, marker='*',
                    edgecolors='white', linewidths=0.5)
    if has_p4:
        if lm4.shape[1] > 0:
            ax4.scatter(lm4[0], lm4[1], c=C_LM, s=3, alpha=0.55, zorder=3)
        ax4.scatter(*xy4[0], c=C_START, s=90, zorder=8, marker='*',
                    edgecolors='white', linewidths=0.5)

    fig.tight_layout(rect=[0, 0.04, 1, 0.98])

    # Progress text (static position — animated)
    prog = fig.text(0.5, 0.005, '', ha='center', fontsize=8,
                    color=C_TEXT, animated=True)

    # ── Dynamic artists (animated=True → excluded from bg blit) ──
    ghost1, = ax1.plot([], [], '-', color=C_P1, lw=0.7, alpha=0.2, animated=True)
    trail1, = ax1.plot([], [], '-', color=C_P1, lw=2.2, alpha=1.0,  animated=True)
    robot1, = ax1.plot([], [], 'o', color=C_ROBOT, ms=7, animated=True,
                       mec='#1D4ED8', mew=1.5)

    g3 = t3 = r3 = None
    g4 = t4_line = r4 = None

    if has_p3:
        g3, = ax3.plot([], [], '-', color=C_P3, lw=0.7, alpha=0.2, animated=True)
        t3, = ax3.plot([], [], '-', color=C_P3, lw=2.2, alpha=1.0,  animated=True)
        r3, = ax3.plot([], [], 'o', color=C_ROBOT, ms=7, animated=True,
                       mec='#1D4ED8', mew=1.5)
    if has_p4:
        g4,     = ax4.plot([], [], '-', color=C_P4, lw=0.7, alpha=0.2, animated=True)
        t4_line,= ax4.plot([], [], '-', color=C_P4, lw=2.2, alpha=1.0,  animated=True)
        r4,     = ax4.plot([], [], 'o', color=C_ROBOT, ms=7, animated=True,
                           mec='#5B21B6', mew=1.5)

    # Draw static background ONCE → blit restores this each frame
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    W, H = fig.canvas.get_width_height()
    print(f"  Frame size: {W}x{H}  |  frames={len(frames)}  fps={fps}  step={step}")

    # ── OpenCV writer ─────────────────────────────────────────
    out_path = os.path.join(results_dir, f"trajectory_video_{dname}.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    vw       = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # Collect all animated artists grouped by axes
    ax_artists = {ax1: [ghost1, trail1, robot1]}
    if has_p3: ax_artists[ax3] = [g3, t3, r3]
    if has_p4: ax_artists[ax4] = [g4, t4_line, r4]

    # ── Render loop ───────────────────────────────────────────
    t0_wall = time.time()
    n_frames = len(frames)

    for fi, t_idx in enumerate(frames):
        ts = max(0, t_idx - trail)

        # Update Part 1
        ghost1.set_data(xy1[:t_idx+1, 0],   xy1[:t_idx+1, 1])
        trail1.set_data(xy1[ts:t_idx+1, 0], xy1[ts:t_idx+1, 1])
        robot1.set_data([xy1[t_idx, 0]], [xy1[t_idx, 1]])

        # Update Part 3
        if has_p3:
            g3.set_data(xy3[:t_idx+1, 0],   xy3[:t_idx+1, 1])
            t3.set_data(xy3[ts:t_idx+1, 0], xy3[ts:t_idx+1, 1])
            r3.set_data([xy3[t_idx, 0]], [xy3[t_idx, 1]])

        # Update Part 4 (clamp)
        if has_p4:
            t4c  = min(t_idx, len(xy4)-1)
            ts4  = max(0, t4c - trail)
            g4.set_data(xy4[:t4c+1, 0],   xy4[:t4c+1, 1])
            t4_line.set_data(xy4[ts4:t4c+1, 0], xy4[ts4:t4c+1, 1])
            r4.set_data([xy4[t4c, 0]], [xy4[t4c, 1]])

        # Progress text
        pct = 100 * fi / max(n_frames-1, 1)
        elapsed = time.time() - t0_wall
        eta = elapsed / max(fi+1, 1) * (n_frames - fi - 1)
        prog.set_text(f'{pct:.0f}%  ETA {eta:.0f}s')

        # ── BLIT: restore bg, draw only animated artists ──────
        fig.canvas.restore_region(bg)
        for ax, artists in ax_artists.items():
            for a in artists:
                ax.draw_artist(a)
        fig.draw_artist(prog)
        fig.canvas.blit(fig.bbox)

        # Convert to BGR and write
        buf   = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = cv2.cvtColor(buf.reshape(H, W, 4), cv2.COLOR_RGBA2BGR)
        vw.write(frame)

        # Print progress every ~5%
        if fi % max(1, n_frames // 20) == 0:
            print(f"  {pct:5.1f}%  ETA {eta:4.0f}s", end='\r', flush=True)

    vw.release()
    plt.close(fig)

    total = time.time() - t0_wall
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n  Done in {total:.1f}s  ({total/n_frames*1000:.0f}ms/frame)")
    print(f"  → {out_path}  ({size_mb:.1f} MB)")


# ═══════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ECE276A PR3 Trajectory Video')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['00','01','02'],
                        help='Single dataset (omit = all three)')
    parser.add_argument('--results_dir', type=str, default=RESULTS_DIR,
                        help='Folder with T_imu_*.npy / results_*.npz')
    parser.add_argument('--fps',  type=int, default=30,
                        help='Output FPS (default 30)')
    parser.add_argument('--step', type=int, default=3,
                        help='Timestep subsampling: 1=every frame, 3=every 3rd (default)')
    parser.add_argument('--dpi',  type=int, default=80,
                        help='Figure DPI (default 80, lower = faster)')
    args = parser.parse_args()

    rdir = args.results_dir
    if not os.path.isdir(rdir):
        print(f"ERROR: results_dir not found:\n  {rdir}")
        sys.exit(1)

    datasets = ['00','01','02'] if args.dataset is None else [args.dataset]
    for ds in datasets:
        build_video(f"dataset{ds}", rdir,
                    fps=args.fps, step=args.step, dpi=args.dpi)

    print("\nAll done!")