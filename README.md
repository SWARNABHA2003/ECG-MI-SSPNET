# Detecting and Localising Myocardial Infarction from 12-Lead ECGs

A dual-branch attention CNN (adapted from **SSPNet**) that classifies a 10-second, 12-lead ECG as
**normal (NORM)**, **inferior MI (IMI)** or **anteroseptal MI (ASMI)**, trained on PTB-XL. The model runs on a CPU
and is deployed as a web app.

**Live app:** https://ecg-demo-pre.streamlit.app
**Report (IEEE format, 9 pages):** [`docs/report.pdf`](docs/report.pdf)

Swarnabha Pramanik, Indian Institute of Science, Bengaluru. Course project, November 2026.

---

## Results

Held-out PTB-XL test fold 10 (1,073 ECGs, evaluated once after all settings were frozen):

| Model | Macro-AUC | Macro-F1 | Accuracy |
|---|---|---|---|
| Single network (mean ± SD, 3 seeds) | 0.943 ± 0.002 | 0.741 ± 0.032 | 0.906 ± 0.009 |
| **Final: 3-seed ensemble + class offsets** | **0.964** [0.953, 0.974] | **0.770** [0.717, 0.815] | **0.909** [0.891, 0.925] |

| Class | AUC | Precision | Recall | F1 | n |
|---|---|---|---|---|---|
| NORM | 0.953 | 0.93 | 0.97 | 0.95 | 912 |
| IMI | 0.943 | 0.69 | 0.54 | 0.61 | 124 |
| ASMI | 0.995 | 0.81 | 0.70 | 0.75 | 37 |

Square brackets: 95% bootstrap confidence intervals (1,000 resamples). Of the 99 infarcts detected as MI, 93 (94%)
were assigned the correct wall; 30 of 912 normal ECGs (3.3%) were flagged as MI. Validation (fold 9): macro-AUC 0.961.

## Method in brief

1. **Input: two images per ECG, 12 leads each (2 × 12 × 128 × 128).**
   - *Signed median beat:* 0.5–40 Hz filtering, R-peak detection in the 5–18 Hz QRS band, median of all beats,
     PR segment set to 0 mV, drawn on a fixed ±1.5 mV scale, so ST elevation/depression, Q waves and T inversion keep
     their sign.
   - *8–40 Hz spectrogram:* STFT magnitude of the full 10 s, per lead (QRS shape and energy).
2. **Network (2.66 M parameters):** two branches of asymmetric convolution blocks with channel and spatial attention →
   bidirectional cross-attention between the branches → fusion of their sum and difference → residual bottleneck →
   8-head self-attention → classifier.
3. **Training:** PTB-XL patient-wise folds 1–8 (train), 9 (validation), 10 (test); square-root class weights, label
   smoothing, AdamW, EMA weights, early stopping on validation macro-AUC; 3 seeds.
4. **Decision:** average of the 3 networks' probabilities + a log-offset for IMI (+0.5) tuned on validation, then frozen.

## Repository contents

```
notebooks/
  01_train_sspnet_ensemble.ipynb   preprocessing, model, training of 3 seeds, ensemble + offsets, test, figures, statistics
  02_ablation.ipynb                same pipeline with an input-ablation cell (one branch blanked)
app/
  app.py                           Streamlit web app (entry point)
  ecg_preprocessing.py             preprocessing, identical to the notebook
  sspnet_model.py                  SSPNet + ensemble loading / prediction
  selftest.py                      checks the app reproduces the notebook's images and probabilities
  export_bundle.py                 packs trained weights + app for deployment
  prepare_demo.py, make_ptbxl_labels.py, make_real_ecg_figures.py   helper scripts
docs/
  report.pdf                       full report
requirements.txt
```

Trained weights, the PTB-XL data and the image cache are **not** stored here (see `.gitignore`).

## How to run

```bash
pip install -r requirements.txt
```

**Train and evaluate**
1. Download PTB-XL v1.0.3 from PhysioNet: https://physionet.org/content/ptb-xl/1.0.3/
2. Set the path, for example `export PTBXL_ROOT=/path/to/ptb-xl/1.0.3`, or edit `PTBXL_ROOT` in the first code cell.
3. Run `notebooks/01_train_sspnet_ensemble.ipynb` top to bottom. Training runs on a CPU (about 10 minutes per epoch).
   Keep `EVALUATE_TEST = False` until all choices are final; the test fold is meant to be evaluated once.

**Web app**
1. After training, build the deployment folder: `python app/export_bundle.py --ptbxl /path/to/ptb-xl/1.0.3`
2. Run locally: `streamlit run app/app.py` (in the exported folder), or deploy that folder on Streamlit Community Cloud.

The app accepts WFDB (`.hea` + `.dat`), `.mat` and table files, resamples to 100 Hz, checks signal quality, and shows
the predicted class, the median beat of each lead and which leads influenced the decision. For PTB-XL files it also
shows the cardiologist's label.

## References

- H.-J. Kim, D. Park, J.-W. Jang, Y.-G. Ju, D.-O. Won, "SSPNet: Towards feasible spatio-spectral portraits-based deep
  learning framework for neurodegenerative disease multi-classification," MICCAI 2025, LNCS 15963, pp. 531–541.
- P. Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset," *Scientific Data* 7:154, 2020.
- A. L. Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet," *Circulation* 101(23):e215–e220, 2000.
- N. Strodthoff et al., "Deep learning for ECG analysis: benchmarks and insights from PTB-XL," *IEEE JBHI* 25(5), 2021.

Full reference list: see the report.

## Disclaimer

Research prototype for a course project. **Not a medical device and not for clinical use.**
PTB-XL is provided by PhysioNet under the CC BY 4.0 licence.
