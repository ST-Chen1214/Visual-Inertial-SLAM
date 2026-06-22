"""
ECE 276A Project 3 — Part 2 (Extra Credit)
Feature Detection and Matching using OpenCV Optical Flow
For dataset02 which does not provide pre-computed features.

Steps:
  (a) Stereo matching: left → right (per timestep)
  (b) Temporal tracking: left_t → left_{t+1}
"""

import numpy as np
import cv2
import os


# ============================================================
#  Lucas-Kanade Optical Flow Parameters
# ============================================================

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

SHI_TOMASI_PARAMS = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=10,
    blockSize=5,
)


# ============================================================
#  Stereo Feature Matching  (Part 2a)
# ============================================================

def stereo_match(img_l, img_r, pts_l):
    """
    Track features detected in the left image to the right image.

    Uses Lucas-Kanade optical flow (forward + backward check).

    Args:
        img_l : (H, W) left  grayscale image
        img_r : (H, W) right grayscale image
        pts_l : (N, 2)  feature points in left image [x, y]

    Returns:
        pts_r  : (N, 2) matched points in right image (or NaN if lost)
        status : (N,)   boolean mask — True if tracking succeeded
    """
    if len(pts_l) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)

    pts_l_cv = pts_l.astype(np.float32).reshape(-1, 1, 2)

    # Forward: left → right
    pts_r_cv, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
        img_l, img_r, pts_l_cv, None, **LK_PARAMS)

    # Backward: right → left  (for verification)
    pts_back_cv, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
        img_r, img_l, pts_r_cv, None, **LK_PARAMS)

    # Bidirectional consistency check
    fb_error = np.linalg.norm(
        pts_l_cv.reshape(-1, 2) - pts_back_cv.reshape(-1, 2), axis=1)

    status = (st_fwd.flatten() == 1) & (st_bwd.flatten() == 1) & (fb_error < 1.0)

    pts_r = pts_r_cv.reshape(-1, 2)
    return pts_r, status


# ============================================================
#  Temporal Feature Tracking  (Part 2b)
# ============================================================

def temporal_track(img_prev, img_curr, pts_prev):
    """
    Track features from previous to current left-camera frame.

    Args:
        img_prev : (H, W) previous grayscale image
        img_curr : (H, W) current  grayscale image
        pts_prev : (N, 2) feature locations in previous frame [x, y]

    Returns:
        pts_curr : (N, 2) tracked locations in current frame
        status   : (N,)   boolean mask
    """
    if len(pts_prev) == 0:
        return np.zeros((0, 2)), np.zeros(0, dtype=bool)

    pts_prev_cv = pts_prev.astype(np.float32).reshape(-1, 1, 2)

    pts_curr_cv, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
        img_prev, img_curr, pts_prev_cv, None, **LK_PARAMS)

    pts_back_cv, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
        img_curr, img_prev, pts_curr_cv, None, **LK_PARAMS)

    fb_error = np.linalg.norm(
        pts_prev_cv.reshape(-1, 2) - pts_back_cv.reshape(-1, 2), axis=1)

    status = (st_fwd.flatten() == 1) & (st_bwd.flatten() == 1) & (fb_error < 1.5)
    pts_curr = pts_curr_cv.reshape(-1, 2)
    return pts_curr, status


# ============================================================
#  Main Feature Tracking Pipeline
# ============================================================

def build_feature_tracks(images_l, images_r, redetect_interval=10):
    """
    Build the (4, M, N) feature matrix from raw image sequences.

    Args:
        images_l : (N, H, W) uint8  left  image sequence
        images_r : (N, H, W) uint8  right image sequence
        redetect_interval : re-detect new features every k frames

    Returns:
        features : (4, M_total, N)  stereo pixel coordinates
                   (-1 = not observed)
    """
    N = len(images_l)
    H, W = images_l[0].shape[:2]

    # Global feature database
    # Each feature has a track: {frame_idx: (lx, ly, rx, ry)}
    tracks = []       # list of dicts
    active_ids = []   # which tracks are currently active
    active_pts  = []  # current (lx, ly) for active tracks

    def add_new_features(img_l, img_r, t_idx, exclude_pts=None):
        """Detect new Shi-Tomasi corners and add to track database."""
        mask = np.ones((H, W), dtype=np.uint8) * 255
        if exclude_pts is not None and len(exclude_pts) > 0:
            for pt in exclude_pts:
                cv2.circle(mask, (int(pt[0]), int(pt[1])), 10, 0, -1)

        corners = cv2.goodFeaturesToTrack(img_l, mask=mask, **SHI_TOMASI_PARAMS)
        if corners is None:
            return

        new_pts_l = corners.reshape(-1, 2)
        new_pts_r, status = stereo_match(img_l, img_r, new_pts_l)

        for j, (pl, pr) in enumerate(zip(new_pts_l, new_pts_r)):
            if not status[j]:
                continue
            # Epipolar constraint: rows should be approximately equal
            if abs(pl[1] - pr[1]) > 3.0:
                continue
            # Right x should be less than left x (disparity ≥ 0)
            if pr[0] >= pl[0] or pl[0] - pr[0] > 200:
                continue

            track = {t_idx: (pl[0], pl[1], pr[0], pr[1])}
            tracks.append(track)
            active_ids.append(len(tracks) - 1)
            active_pts.append(pl.copy())

    print(f"[Feature Tracking] Processing {N} frames...")

    for t in range(N):
        img_l = images_l[t]
        img_r = images_r[t]

        if t == 0 or t % redetect_interval == 0 or len(active_ids) < 50:
            # Re-detect features
            cur_pts = np.array(active_pts) if active_pts else None
            add_new_features(img_l, img_r, t, exclude_pts=cur_pts)

        if t == 0:
            continue

        # ---- Temporal tracking: carry forward active features ----
        if len(active_ids) == 0:
            continue

        pts_prev = np.array(active_pts, dtype=np.float32)
        pts_curr, status = temporal_track(images_l[t - 1], img_l, pts_prev)

        # Stereo-match for survivors
        surviving = np.where(status)[0]
        if len(surviving) == 0:
            active_ids.clear()
            active_pts.clear()
            continue

        pts_curr_surv = pts_curr[surviving]
        pts_r_surv, st_stereo = stereo_match(img_l, img_r, pts_curr_surv)

        new_active_ids = []
        new_active_pts = []

        for i_s, i_orig in enumerate(surviving):
            if not st_stereo[i_s]:
                continue
            pl = pts_curr_surv[i_s]
            pr = pts_r_surv[i_s]

            # Epipolar check
            if abs(pl[1] - pr[1]) > 3.0:
                continue
            if pr[0] >= pl[0] or pl[0] - pr[0] > 200:
                continue

            track_id = active_ids[i_orig]
            tracks[track_id][t] = (pl[0], pl[1], pr[0], pr[1])
            new_active_ids.append(track_id)
            new_active_pts.append(pl.copy())

        active_ids[:] = new_active_ids
        active_pts[:] = new_active_pts

        if t % 50 == 0:
            print(f"  t={t}/{N}  active={len(active_ids)}  total_tracks={len(tracks)}")

    # ---- Build (4, M, N) matrix ----
    M = len(tracks)
    features = np.full((4, M, N), -1.0)

    for j, track in enumerate(tracks):
        for t_idx, (lx, ly, rx, ry) in track.items():
            features[:, j, t_idx] = [lx, ly, rx, ry]

    print(f"[Feature Tracking] Done. Total features M={M}")
    return features


# ============================================================
#  Entry point
# ============================================================

if __name__ == "__main__":
    import sys

    dataset_path = sys.argv[1] if len(sys.argv) > 1 else \
        r"D:\ucsd\handouts\ece276a\ECE276A_PR3\ECE276A_PR3\data\ECE276A_P3_Data\dataset02"

    dname = os.path.basename(dataset_path)
    print(f"=== Part 2: Feature Detection (Extra Credit) ===")
    print(f"Dataset: {dataset_path}")

    # Load images
    img_file = os.path.join(dataset_path, f"{dname}_imgs.npy")
    if not os.path.exists(img_file):
        img_file = os.path.join(dataset_path, "imgs.npy")

    print(f"Loading images from {img_file} ...")
    imgs_raw = np.load(img_file, allow_pickle=True)
    print(f"  Raw shape: {imgs_raw.shape}  dtype: {imgs_raw.dtype}")

    # ── Unwrap 0-dim object array ─────────────────────────────
    if imgs_raw.ndim == 0:
        obj = imgs_raw.item()
        print(f"  0-dim object type: {type(obj)}")

        if isinstance(obj, dict):
            print(f"  Dict keys: {list(obj.keys())}")
            # Print shape of each value for debugging
            for k, v in obj.items():
                try:
                    arr_v = np.array(v)
                    print(f"    key={repr(k)}  shape={arr_v.shape}  dtype={arr_v.dtype}")
                except Exception as e:
                    print(f"    key={repr(k)}  error={e}")

            # Try common stereo key pairs
            found = False
            for lk, rk in [('cam_imgs_L','cam_imgs_R'),
                            ('cam0','cam1'), ('left','right'), ('l','r'),
                            ('img0','img1'), (0,1), ('images_l','images_r'),
                            ('image_l','image_r'), ('imgs_l','imgs_r')]:
                if lk in obj and rk in obj:
                    images_l = np.array(obj[lk])
                    images_r = np.array(obj[rk])
                    print(f"  Using keys: '{lk}' (L) + '{rk}' (R)")
                    print(f"    L shape={images_l.shape}  R shape={images_r.shape}")
                    found = True
                    imgs_raw = None
                    break

            if not found:
                # Try all keys - look for one with stereo data
                keys = list(obj.keys())
                for k in keys:
                    arr = np.array(obj[k])
                    print(f"  Trying key='{k}'  shape={arr.shape}")
                    if arr.ndim == 4 and arr.shape[0] == 2:
                        images_l, images_r = arr[0], arr[1]
                        found = True; imgs_raw = None
                        print(f"  Split axis 0 → L={images_l.shape}")
                        break
                    elif arr.ndim == 4 and arr.shape[1] == 2:
                        images_l, images_r = arr[:, 0], arr[:, 1]
                        found = True; imgs_raw = None
                        print(f"  Split axis 1 → L={images_l.shape}")
                        break
                    elif arr.ndim == 4 and arr.shape[-1] == 2:
                        images_l, images_r = arr[..., 0], arr[..., 1]
                        found = True; imgs_raw = None
                        print(f"  Split axis -1 → L={images_l.shape}")
                        break
                    elif arr.ndim == 5:
                        # (N, 2, H, W, C) or (2, N, H, W) etc.
                        if arr.shape[0] == 2:
                            images_l, images_r = arr[0], arr[1]
                        elif arr.shape[1] == 2:
                            images_l, images_r = arr[:,0], arr[:,1]
                        found = True; imgs_raw = None
                        print(f"  5-dim split → L={images_l.shape}")
                        break

                if not found:
                    raise ValueError(
                        f"Cannot parse dict. Keys={keys}. "
                        f"Please check the printed shapes above and report back."
                    )

        elif isinstance(obj, np.ndarray):
            imgs_raw = obj
            print(f"  Unwrapped ndarray shape: {imgs_raw.shape}")

        elif hasattr(obj, '__iter__') or hasattr(obj, '__len__'):
            # Could be a list/tuple of arrays
            obj_list = list(obj)
            print(f"  Iterable length: {len(obj_list)}")
            if len(obj_list) >= 2:
                images_l = np.array(obj_list[0])
                images_r = np.array(obj_list[1])
                imgs_raw = None
                print(f"  List[0] = L {images_l.shape}, List[1] = R {images_r.shape}")
            else:
                raise ValueError(f"Iterable has only {len(obj_list)} elements")

        else:
            raise ValueError(
                f"0-dim object is {type(obj)} — not dict/ndarray/list. "
                f"Cannot parse image file."
            )

    # ── Parse standard array layouts ─────────────────────────
    if imgs_raw is not None:
        s = imgs_raw.shape
        print(f"  Parsing array shape: {s}")
        if imgs_raw.ndim == 5:
            # (2, N, H, W, C) or (N, 2, H, W) etc.
            if s[0] == 2:
                images_l = imgs_raw[0]   # (N, H, W) or (N, H, W, C)
                images_r = imgs_raw[1]
            elif s[1] == 2:
                images_l = imgs_raw[:, 0]
                images_r = imgs_raw[:, 1]
            else:
                raise ValueError(f"5-dim array shape {s} not understood")
        elif imgs_raw.ndim == 4:
            if s[0] == 2:                # (2, N, H, W)
                images_l = imgs_raw[0]
                images_r = imgs_raw[1]
            elif s[-1] == 2:             # (N, H, W, 2)
                images_l = imgs_raw[..., 0]
                images_r = imgs_raw[..., 1]
            elif s[1] == 2:              # (N, 2, H, W)
                images_l = imgs_raw[:, 0]
                images_r = imgs_raw[:, 1]
            else:
                raise ValueError(f"4-dim array shape {s} not understood")
        elif imgs_raw.ndim == 3:
            # (N, H, W*2) — left/right concatenated side-by-side
            W2 = s[2] // 2
            images_l = imgs_raw[:, :, :W2]
            images_r = imgs_raw[:, :, W2:]
            print(f"  Assuming side-by-side stereo, split at W={W2}")
        else:
            raise ValueError(f"Unexpected array ndim={imgs_raw.ndim} shape={s}")

    # ── Ensure grayscale uint8 ────────────────────────────────
    def to_gray_uint8(imgs):
        """Convert to (N, H, W) uint8 grayscale."""
        if imgs.ndim == 4:              # (N, H, W, C)
            import cv2 as _cv2
            imgs = np.stack([_cv2.cvtColor(f.astype(np.uint8), _cv2.COLOR_BGR2GRAY)
                             for f in imgs])
        if imgs.dtype != np.uint8:
            if imgs.max() <= 1.0:
                imgs = (imgs * 255).astype(np.uint8)
            else:
                imgs = imgs.astype(np.uint8)
        return imgs

    images_l = to_gray_uint8(images_l)
    images_r = to_gray_uint8(images_r)
    print(f"  images_l: {images_l.shape}  dtype={images_l.dtype}")
    print(f"  images_r: {images_r.shape}  dtype={images_r.dtype}")

    # Run feature tracking
    features = build_feature_tracks(images_l, images_r, redetect_interval=10)

    # Save
    out_path = f"features_{dname}.npy"
    np.save(out_path, features)
    print(f"Features saved: {out_path}  shape={features.shape}")