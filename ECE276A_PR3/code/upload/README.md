# ECE 276A Project 3 — Visual-Inertial SLAM

## Files

| File | Description |
|------|-------------|
| `main.py` | **Main entry point** — runs any/all parts |
| `pr3_utils.py` | SE(3) math, stereo projection, Jacobians, data loading |
| `imu_localization.py` | Part 1 — IMU EKF prediction |
| `landmark_mapping.py` | Part 3 — Landmark mapping EKF |
| `vi_slam.py` | Part 4 — Full Visual-Inertial SLAM |

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


### Output directory

All images (`.png`) and result arrays (`.npz`) are saved to `./results/` by default.
Override with `--output_dir path/to/dir`.