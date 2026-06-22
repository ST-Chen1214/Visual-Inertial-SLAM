# ECE 276A Project 3 — Visual-Inertial SLAM

## Files

| File | Description |
|------|-------------|
| `main.py` | **Main entry point** — runs any/all parts |
| `pr3_utils.py` | SE(3) math, stereo projection, Jacobians, data loading |
| `imu_localization.py` | Part 1 — IMU EKF prediction |
| `landmark_mapping.py` | Part 3 — Landmark mapping EKF |
| `vi_slam.py` | Part 4 — Full Visual-Inertial SLAM |
| `feature_detection.py` | Part 2 — Optical-flow feature tracking (Extra Credit) |

## Dependencies

```bash
pip install numpy scipy matplotlib opencv-python
```

## How to Run

### Run everything on dataset00

```bash
python main.py --dataset 00 --part all
```

### Run on a specific dataset path

```bash
python main.py "D:\ucsd\handouts\ece276a\...\dataset00" --part all
```

### Run only Part 1 (IMU localisation)

```bash
python main.py --dataset 00 --part 1
```

### Run only Part 3 (Landmark mapping)

```bash
python main.py --dataset 00 --part 3
```

### Run only Part 4 (VI-SLAM)

```bash
python main.py --dataset 00 --part 4
```

### Run individual scripts

```bash
python imu_localization.py  "D:\...\dataset00"
python landmark_mapping.py  "D:\...\dataset01"
python vi_slam.py            "D:\...\dataset01"
python feature_detection.py "D:\...\dataset02"   # Extra credit
```

### Output directory

All images (`.png`) and result arrays (`.npz`) are saved to `./results/` by default.
Override with `--output_dir path/to/dir`.

## Key Configuration — Tuning Noise

Edit `NOISE_PRESETS` in `main.py` or pass arguments directly:

```python
NOISE_PRESETS = {
    "dataset00": dict(W_v=5e-5, W_w=1e-5, V_noise=4.0),
    "dataset01": dict(W_v=5e-5, W_w=1e-5, V_noise=4.0),
    "dataset02": dict(W_v=8e-5, W_w=2e-5, V_noise=6.0),
}
```

- **`W_v`** — translational velocity noise (increase if IMU drifts too much)
- **`W_w`** — rotational velocity noise
- **`V_noise`** — pixel observation noise variance (increase if landmarks are noisy)

## Algorithm Summary

### Part 1 — IMU EKF Prediction

Motion model on SE(3):

```
T_{t+1} = T_t ⊕ exp(τ · [v_t; ω_t]^)
Σ_{t+1} = F_t Σ_t F_t^T + W·τ
F_t     = exp(-τ · curlyhat([v_t; ω_t]))
```

### Part 3 — Landmark Mapping EKF

- Fixed IMU trajectory from Part 1
- Initialise landmarks via stereo DLT triangulation
- EKF update per landmark using stereo observation model:
  `z = [fu·x/z + cu, fv·y/z + cv]` for left and right cameras

### Part 4 — VI-SLAM

- Joint state: `(T_t, m₁, …, mₘ)`
- Block covariance: `Σ = [[Σ_TT, Σ_Tm], [Σ_mT, Σ_mm]]`
- Prediction updates `T_t`, `Σ_TT`, `Σ_Tm` via IMU
- Update corrects both `T_t` and `m` using stereo observations
- Jacobians computed analytically for both pose and landmark

### Part 2 — Feature Detection (Extra Credit, dataset02)

- Shi-Tomasi corner detection on left image
- Left→Right stereo matching via LK optical flow + epipolar check
- Left_t→Left_{t+1} temporal tracking via LK optical flow
- Bidirectional consistency check to remove outliers

## Data Format Expected

The `.npy` file should contain a dict with keys:

```
t / t_stamps          : (N,)    timestamps [s]
v / linear_velocity   : (3, N)  linear velocity  in body frame
w / angular_velocity  : (3, N)  angular velocity in body frame
features              : (4, M, N) stereo pixel coords (-1 = not seen)
K_l, K_r              : (3, 3)  camera intrinsics
imu_T_cam0/1          : (4, 4)  extrinsics (IMU ← Camera)
```
