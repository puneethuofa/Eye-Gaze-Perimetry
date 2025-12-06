# Eye-Gaze-Perimetry


This repository contains an end-to-end research pipeline for **camera-based visual field testing** (eye gaze perimetry). The goal is to estimate visual field responses using:

- Lightweight deep gaze models (ViT, ResNet, MobileNet) trained on Eye Gaze / MPIIGaze data.
- A Humphrey-style visual field (VF) simulation video.
- A webcam recording of the subject’s face during the VF test.
- Post-hoc analysis that computes stimulus–response timing (reaction time, RT) and gaze-based hit-rates across the visual field.

The code can be run:

- On the UArizona HPC cluster (Ocelote, using a Python 3.11 environment such as `clean311`), or  
- On a single GPU workstation or laptop, with minor path changes.

Cluster-specific batch scripts (for example `.sbatch` files) are **not** included in this repository, so the project stays portable. You can keep your own SLURM scripts in a private folder if needed.


<img width="453" height="243" alt="image" src="https://github.com/user-attachments/assets/aa465d7a-380b-40b7-a71a-1d71ed59fce9" />

---

## 1. Repository layout

Recommended directory structure:

```text
eye-gaze-perimetry/
  README.md

  src/
    data/
      build_manifest_mpiigaze_chunks.py        # build CSV manifest for Eye Gaze / MPIIGaze
    training/
      train_vit_mpiigaze_chunks_mem.py         # train ViT-based gaze model on manifest
    inference/
      infer_user_video_vit.py                  # run gaze inference on a webcam/user video
    analysis/
      compute_stimulus_trace.py                # track VF stimulus position from HVF_sim video
      compute_reaction_time.py                 # align stimulus & gaze, compute RT + EDA


Each Python file corresponds to a distinct stage in the pipeline:

1. `build_manifest_mpiigaze_chunks.py` – data preparation, builds the training manifest CSV.
2. `train_vit_mpiigaze_chunks_mem.py` – model training on the manifest.
3. `infer_user_video_vit.py` – gaze inference on a user video using the trained model.
4. `compute_stimulus_trace.py` – stimulus tracking from a VF simulation video.
5. `compute_reaction_time.py` – RT estimation and exploratory data analysis (EDA).

The rest of this README explains each of these components in detail.

---

## 2. `src/data/build_manifest_mpiigaze_chunks.py`

### Purpose

Build a **manifest CSV** that describes all training samples (eye images) and their associated gaze labels. This manifest becomes the single source of truth for the training and evaluation scripts.

### Conceptual behavior

At a high level, the script:

1. Takes as input the root directory of the Eye Gaze / MPIIGaze dataset.
2. Iterates through subjects, sessions, and image files.
3. For each eye crop:

   * Locates the corresponding annotation (yaw, pitch or normalized gaze).
   * Converts gaze labels into yaw and pitch in degrees if necessary.
   * Creates a row for the manifest with image path and labels.
4. Cleans the data:

   * Skips missing or corrupted image files.
   * Filters out invalid labels.
5. Writes all rows into a consolidated manifest CSV.

### Expected manifest format

The manifest generally contains at least:

* `img`         – relative or absolute path to the eye image.
* `yaw_deg`     – gaze yaw label in degrees (horizontal).
* `pitch_deg`   – gaze pitch label in degrees (vertical).

Example (conceptual):

```csv
img,yaw_deg,pitch_deg
subject01/session1/img_000123.png,342.4,187.2
subject01/session1/img_000124.png,343.1,186.9
...
```

You can extend this format to include subject ID, session, left/right eye, etc. as additional columns if needed.

### Typical usage

```bash
python src/data/build_manifest_mpiigaze_chunks.py \
  --data-root /path/to/EyeGaze_dataset_root \
  --out /path/to/runs/vit_mpiigaze/20251116/manifest_mpiigaze.csv
```

The exact arguments depend on how the script has been parameterized, but the goal is always the same: produce `manifest_mpiigaze.csv`.

---

## 3. `src/training/train_vit_mpiigaze_chunks_mem.py`

### Purpose

Train a **lightweight gaze regression model** (ViT Tiny by default) on the manifest CSV in a memory-aware, chunked fashion. This model maps eye images to `[yaw_deg, pitch_deg]`.

### Conceptual behavior

1. **Argument parsing**

   Reads command-line arguments such as:

   * `--manifest` – path to the manifest CSV.
   * `--run-dir` – directory where training outputs (checkpoints, logs) are saved.
   * `--model` – backbone identifier (for example `vit_tiny_patch16_224`).
   * `--img-size` – image size used for training (for example 96).
   * `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--val-frac`, `--seed`, etc.

2. **Dataset and transforms**

   * Loads the manifest CSV into a dataset object (for example, a class that reads images and labels row by row).
   * Applies transforms:

     * Resize to `img_size × img_size`.
     * Convert to PyTorch tensor.
     * Normalize with ImageNet mean and standard deviation.
   * Splits the dataset into **training** and **validation** subsets (for example, 90 percent / 10 percent based on `--val-frac`).

3. **Model construction**

   * Uses `timm` to create a backbone:

     * ViT Tiny (`vit_tiny_patch16_224`) by default.
     * You can switch to ResNet-18 or MobileNet-V2 by changing the `--model` argument if the script supports it.
   * Replaces the final classification head with a **two-dimensional regression head** that outputs `[yaw, pitch]` for each input image.
   * Optionally applies dropout before the final layer.

4. **Training loop**

   * Moves the model to GPU if available.
   * Sets up an optimizer (commonly Adam with weight decay).
   * Uses mean squared error (MSE) loss between predicted `[yaw, pitch]` and true labels.
   * For each epoch:

     * Training phase:

       * Iterates over training batches.
       * Computes loss, backpropagates, updates parameters.
       * Optionally clips gradients with `max-grad-norm`.
     * Validation phase:

       * Evaluates on the validation set without gradient updates.
       * Computes validation loss.
     * Logs `train_loss` and `val_loss`.

5. **Checkpointing and logging**

   * Tracks the best validation loss observed during training.
   * Whenever `val_loss` improves:

     * Saves a checkpoint `best.ckpt` containing:

       * `state_dict` for model weights.
       * `model_name`.
       * `img_size`.
   * Writes a JSON configuration file, such as `args.json`, containing:

     * Paths used.
     * Hyperparameters.
     * Basic training history (loss curves).

### Inputs and outputs

* Input

  * Manifest CSV from the data preprocessing step.
  * Command-line arguments specifying training setup.

* Output (inside `RUN_DIR`)

  * `best.ckpt` – trained model weights ready for inference.
  * `args.json` – training configuration and history for reproducibility.
  * Optional extra logs, depending on how you instrument the script.

### Example usage

```bash
python src/training/train_vit_mpiigaze_chunks_mem.py \
  --manifest /path/to/runs/vit_mpiigaze/20251116/manifest_mpiigaze.csv \
  --run-dir /path/to/runs/vit_mpiigaze/20251127 \
  --model vit_tiny_patch16_224 \
  --img-size 96 \
  --epochs 20 \
  --batch-size 128 \
  --lr 1e-4 \
  --weight-decay 1e-4
```

---

## 4. `src/inference/infer_user_video_vit.py`

### Purpose

Use a trained gaze model to perform **per-frame gaze inference** on a webcam/video recording of the subject during the VF test. Produces a time-stamped series of predicted yaw/pitch values.

### Conceptual behavior

1. **Argument parsing**

   Main flags:

   * `--ckpt` – path to the trained checkpoint (`best.ckpt`).
   * `--model` – model type (must match training backbone).
   * `--video` – path to the subject’s face video, for example `original.mp4`.
   * `--out` – path to save the gaze prediction CSV.
   * `--img-size` – input image size for the model.
   * `--fps` – frame rate to use for computing timestamps (seconds).

2. **Model loading**

   * Loads the checkpoint.
   * Reconstructs the model architecture (ViT Tiny with a regression head).
   * Loads `state_dict` into the model.
   * Sets the model to evaluation mode and moves it to GPU if available.

3. **Video processing loop**

   * Opens the video with OpenCV.
   * For each frame:

     * Reads the frame in BGR format.
     * Converts BGR to RGB.
     * Resizes to `img_size × img_size`.
     * Converts to a tensor and normalizes with ImageNet mean/std.
     * Adds a batch dimension and passes it through the model.
     * Receives predicted `[yaw_deg, pitch_deg]`.
     * Computes timestamp `t_sec = frame_index / fps`.
     * Appends `[frame_index, t_sec, yaw_deg, pitch_deg]` to the output.

4. **Output CSV**

   * At the end, writes a CSV file such as `user_gaze.csv` with columns:

     * `frame`
     * `t_sec`
     * `yaw_deg`
     * `pitch_deg`

   * This becomes the gaze signal for later reaction-time analysis.

### Inputs and outputs

* Input

  * Trained checkpoint `best.ckpt`.
  * User video showing the face during perimetry.

* Output

  * `user_gaze.csv` containing predicted gaze angles over time.

### Example usage

```bash
python src/inference/infer_user_video_vit.py \
  --ckpt /path/to/runs/vit_mpiigaze/20251127/best.ckpt \
  --model vit_tiny_patch16_224 \
  --video /path/to/videos/original.mp4 \
  --out /path/to/runs/vit_mpiigaze/20251127/user_gaze.csv \
  --img-size 96 \
  --fps 30
```

---

## 5. `src/analysis/compute_stimulus_trace.py`

### Purpose

Process a **visual field simulation video** (the stimulus display) and estimate the screen-space position of the stimulus at each frame. Produces a stimulus trajectory `cx, cy` over time that aligns with the gaze predictions.

### Conceptual behavior

1. **Argument parsing**

   Main parameters:

   * `--video` – path to the VF simulation video (for example `HVF_sim.mp4`).
   * `--out` – output CSV path, such as `stimulus_trace.csv`.
   * `--fps` – frame rate of the video.
   * `--frame-step` – process every Nth frame (for speed).
   * `--target-width` – width to resize frames for processing (preserving aspect ratio).
   * `--blur` – kernel size for Gaussian blur.
   * `--min-area` – minimum contour area to consider as a stimulus blob.
   * `--max-frames` – maximum number of frames to process (0 = all frames).

2. **Background modeling**

   * Reads an initial frame (or frames) as a “background reference”.
   * Resizes and converts the background frame to grayscale.
   * Applies Gaussian blur to reduce noise.
   * Stores this as `bg_gray`.

3. **Frame processing**

   For each frame (with step `frame-step`):

   * Reads the frame from the video.
   * Resizes it to the target width (height adjusted proportionally).
   * Converts to grayscale and applies Gaussian blur.
   * Computes `diff = abs(gray - bg_gray)` to highlight changes.
   * Applies thresholding (for example Otsu threshold) to get a binary mask.
   * Finds contours on the mask.
   * Selects the largest contour by area if the area exceeds `min-area`.
   * Computes contour moments and centroid `(cx, cy)`.

   If no valid contour is found, `(cx, cy)` is set to `NaN`.

4. **Output CSV**

   For processed frames, the script writes the following fields:

   * `frame` – frame index in the original video.
   * `t_sec` – timestamp of the frame (`frame_index / fps`).
   * `cx` – stimulus centroid x coordinate in the resized frame.
   * `cy` – stimulus centroid y coordinate in the resized frame.

   The output file is typically named `stimulus_trace.csv`.

### Inputs and outputs

* Input

  * VF simulation video `HVF_sim.mp4` (or equivalent).

* Output

  * `stimulus_trace.csv` describing stimulus position over time.

### Example usage

```bash
python src/analysis/compute_stimulus_trace.py \
  --video /path/to/videos/HVF_sim.mp4 \
  --out /path/to/runs/vit_mpiigaze/20251127/stimulus_trace.csv \
  --fps 30 \
  --frame-step 2 \
  --target-width 320 \
  --blur 5 \
  --min-area 40 \
  --max-frames 0
```

---

## 6. `src/analysis/compute_reaction_time.py`

### Purpose

Fuse the **stimulus trace** and **gaze predictions** into a quantitative analysis of:

* Global stimulus–gaze timing (via cross-correlation).
* Per-stimulus reaction times using a distance-to-target rule.
* Exploratory data analysis (histograms, scatter plots, hit-rate map).

This script is the core of the **reaction time and perimetric analysis**.

### Conceptual behavior

1. **Argument parsing**

   Key parameters:

   * `--stimulus` – path to `stimulus_trace.csv`.
   * `--gaze` – path to `user_gaze.csv`.
   * `--out-dir` – output directory for analysis results.
   * `--fps` – sampling rate (typically 30 Hz).
   * `--radius` – radius-of-acceptance in the coordinate system used for `cx, cy` and gaze.
   * `--t-max` – maximum post-stimulus window (in seconds) within which a gaze hit is allowed.
   * `--min-rt-ms` – minimum plausible RT in milliseconds (frames earlier than this are ignored to avoid false positives).

2. **Load data**

   * Reads `stimulus_trace.csv` with columns: `frame, t_sec, cx, cy`.
   * Reads `user_gaze.csv` with columns: `frame, t_sec, yaw_deg, pitch_deg`.

3. **Global cross-correlation**

   * Z-scores stimulus x (`cx`) and gaze yaw (`yaw_deg`) to have zero mean and unit variance.
   * For lags `k` in a range (for example, ±70 frames):

     * Computes correlation between `stim_x(t)` and `yaw(t + k)`.
   * Finds the lag where correlation is highest.
   * Converts that lag to milliseconds:
     `rt_xcorr_ms = best_lag * 1000 / fps`.
   * Saves a cross-correlation plot as `rt_crosscorr.png`.

4. **Time-series overlay**

   * Interpolates gaze yaw to the stimulus timestamps.
   * Z-scores both series.
   * Plots them together over time to visually inspect alignment and approximate lag.
   * Saved as `rt_overlay.png`.

5. **Detect stimulus events**

   * Uses the `cx` time series to detect stimulus onset events.
   * A simple rule is:

     * An event begins when `cx` transitions from NaN to a valid value and there has been no valid `cx` for at least a certain number of frames beforehand (separation between events).

6. **Per-event RT estimation (distance-to-target)**

   For each detected event at index `idx0`:

   * `t0` = onset time from `stimulus_trace` (`t_sec[idx0]`).
   * `(x_t, y_t)` = stimulus location at that time.
   * Interpolates gaze yaw and pitch to the stimulus timestamps, producing gaze samples `(x_gaze(t), y_gaze(t))`.
   * Defines a window `[t0, t0 + t_max]`, converted into a number of frames.
   * Computes distance for each frame in the window:
     [
     d(t) = \sqrt{(x_gaze(t) - x_t)^2 + (y_gaze(t) - y_t)^2}.
     ]
   * Starting after the minimum RT (`min-rt-ms` converted to frames), finds the **first frame** where `d(t) < radius`.
   * If such a frame exists:

     * RT for that event is `(t_hit - t0) * 1000` milliseconds.
     * Eccentricity is computed as distance from a chosen “center” (for example, `(0, 0)` or the screen center).
   * If not:

     * The event is counted as “no gaze hit” and `rt_ms` is recorded as `None`.

   The script aggregates all valid RTs and their corresponding eccentricities for downstream analysis.

7. **EDA plots**

   The script generates several diagnostic plots:

   * `eda_hist_yaw_pitch.png`

     * Histograms of yaw and pitch, summarizing overall gaze usage.

   * `eda_scatter_stimx_yaw.png`

     * Scatter plot of stimulus x position vs gaze yaw at stimulation times, indicating whether the model’s horizontal predictions correlate with stimulus location.

   * `eda_hist_rt_per_event.png`

     * Histogram of per-event RTs (only events with valid RT).

   * `rt_vs_eccentricity.png`

     * Scatter of RT versus eccentricity (stimulus distance from center).
     * Optionally fits a simple regression line to show whether RT increases with eccentricity.

   * `hitrate_map.png`

     * Creates a 2D map of stimulus locations with color indicating hit-rate:

       * Hit-rate per location = number of events with valid RT / total events at that location.

   * `rt_calibrated_example_distance.png`

     * For one event with valid RT, plots distance-to-target vs time since onset (in ms).
     * Shows the radius-of-acceptance threshold and where the crossing happens.

8. **Summary JSON**

   The script writes `rt_summary.json` with fields such as:

   * `rt_xcorr_ms` – global RT estimate from cross-correlation.
   * `best_lag_frames` – lag at which correlation is highest, in frames.
   * `n_events_detected` – total number of detected stimulus events.
   * `n_events_with_rt` – number of events where gaze RT was successfully measured.
   * `rt_ms_mean`, `rt_ms_median`, `rt_ms_std` – summary statistics for per-event RT.
   * `radius_px`, `t_max`, `min_rt_ms` – analysis hyperparameters.
   * `example_distance_plot` – path to the example distance plot.

### Inputs and outputs

* Input

  * `stimulus_trace.csv` from `compute_stimulus_trace.py`.
  * `user_gaze.csv` from `infer_user_video_vit.py`.

* Output (inside `out-dir`)

  * `rt_summary.json` – numerical summary of results.
  * `rt_crosscorr.png` – cross-correlation plot.
  * `rt_overlay.png` – normalized time-series overlay.
  * `eda_hist_yaw_pitch.png` – gaze angle histograms.
  * `eda_scatter_stimx_yaw.png` – stimulus vs gaze scatter.
  * `eda_hist_rt_per_event.png` – RT distribution.
  * `rt_vs_eccentricity.png` – RT vs eccentricity scatter with optional fit.
  * `hitrate_map.png` – hit-rate across stimulus locations.
  * `rt_calibrated_example_distance.png` – example distance-to-target evolution.

### Example usage

```bash
python src/analysis/compute_reaction_time.py \
  --stimulus /path/to/runs/vit_mpiigaze/20251127/stimulus_trace.csv \
  --gaze /path/to/runs/vit_mpiigaze/20251127/user_gaze.csv \
  --out-dir /path/to/runs/vit_mpiigaze/20251127/rt_analysis \
  --fps 30 \
  --radius 30 \
  --t-max 1.5 \
  --min-rt-ms 100
```

---

## 7. Python environment and dependencies

The project assumes:

* Python ≥ 3.9 (tested with Python 3.11).
* GPU-accelerated PyTorch if a GPU is available.
* `timm` for model backbones (ViT, ResNet, MobileNet).
* `opencv-python` for video I/O and processing.
* `pandas`, `numpy`, `matplotlib` for data manipulation and plotting.

Example environment setup on a generic machine:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm opencv-python pandas numpy matplotlib
```

On UArizona’s Ocelote cluster, you typically:

```bash
module load anaconda
source activate clean311
# then pip install packages as above if they are not already in the environment
```

---

## 8. End-to-end workflow

A typical usage of the repository looks like this:

1. **Build manifest from Eye Gaze / MPIIGaze**

   ```bash
   python src/data/build_manifest_mpiigaze_chunks.py \
     --data-root /path/to/EyeGaze_dataset_root \
     --out /path/to/runs/vit_mpiigaze/20251116/manifest_mpiigaze.csv
   ```

2. **Train ViT-based gaze model**

   ```bash
   python src/training/train_vit_mpiigaze_chunks_mem.py \
     --manifest /path/to/runs/vit_mpiigaze/20251116/manifest_mpiigaze.csv \
     --run-dir /path/to/runs/vit_mpiigaze/20251127 \
     --model vit_tiny_patch16_224 \
     --img-size 96 \
     --epochs 20
   ```

3. **Run gaze inference on user video**

   ```bash
   python src/inference/infer_user_video_vit.py \
     --ckpt /path/to/runs/vit_mpiigaze/20251127/best.ckpt \
     --model vit_tiny_patch16_224 \
     --video /path/to/videos/original.mp4 \
     --out /path/to/runs/vit_mpiigaze/20251127/user_gaze.csv \
     --img-size 96 \
     --fps 30
   ```

4. **Extract stimulus trace from VF simulation**

   ```bash
   python src/analysis/compute_stimulus_trace.py \
     --video /path/to/videos/HVF_sim.mp4 \
     --out /path/to/runs/vit_mpiigaze/20251127/stimulus_trace.csv \
     --fps 30
   ```

5. **Compute reaction times and generate EDA**

   ```bash
   python src/analysis/compute_reaction_time.py \
     --stimulus /path/to/runs/vit_mpiigaze/20251127/stimulus_trace.csv \
     --gaze /path/to/runs/vit_mpiigaze/20251127/user_gaze.csv \
     --out-dir /path/to/runs/vit_mpiigaze/20251127/rt_analysis \
     --fps 30 \
     --radius 30 \
     --t-max 1.5 \
     --min-rt-ms 100
   ```

After step 5, you will have:

<img width="342" height="228" alt="image" src="https://github.com/user-attachments/assets/7967567a-ee13-4fb4-82c6-737cceb47712" />

<img width="350" height="291" alt="image" src="https://github.com/user-attachments/assets/e79aa62d-187f-4d5f-9ccf-38e05fbc3163" />


---

## 9. Notes and limitations

* In many experiments, subjects are asked to keep central fixation (similar to standard automated perimetry). Under these conditions, peripheral stimuli may be detected using peripheral vision, so large saccades may not always occur. Gaze-based RT is therefore an indirect proxy for detection.
* The radius-of-acceptance (`--radius`) and the calibration between gaze coordinates and stimulus coordinates strongly influence which events are counted as “hits.” These choices should ideally be calibrated per subject and validated.
* The code is research-focused and optimized for flexibility and inspection rather than clinical deployment. Additional steps are needed for clinical validation, including:

  * More subjects.
  * Robust calibration procedures.
  * Direct comparison with standard SAP outcomes.
  * Potentially a gaze-driven protocol where subjects must look at each stimulus.

---

## 10. Acknowledgements

This repository is part of the **Eye Gaze Perimetry** project at the University of Arizona. It builds on Eye Gaze / MPIIGaze datasets and has been developed and tested using the UArizona Ocelote HPC cluster.

If you use this code or ideas from this project in academic work, please cite the associated report, thesis, or poster when available.

```
```

**Author:** Puneeth Vijay Krishna Samarla  
**Contact:** puneethvks9@gmail.com  

