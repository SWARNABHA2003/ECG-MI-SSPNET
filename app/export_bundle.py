"""
Run this ON THE TRAINING SYSTEM (the one with runs/, the cache and PTB-XL), from the project folder:

    python ecg_mi_app/export_bundle.py --ptbxl /path/to/ptb-xl/1.0.3

It creates ECG_MI_Demo.zip: everything the laptop needs, and nothing else.
  ECG_MI_Demo/
    runs/mi_sspnet_v5_s{1,2,3}/best.pt + config.json      your trained networks
    runs/mi_sspnet_v5_ensemble/ensemble_config.json          ensemble members + class offsets
    ecg_mi_app/                                              the app code
    ecg_mi_app/demo_examples/                                example ECGs (test/validation fold)
    ecg_mi_app/selftest_pack.npz                             lets the laptop verify itself without PTB-XL
    setup_linux_mac.sh / setup_windows.bat                   one-time install on the laptop
    start_demo.sh / start_demo.bat                           start the app
Copy the zip to the laptop (USB stick, Google Drive, ...), unzip, run the setup script once.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parent
sys.path.insert(0, str(APP))
import ecg_preprocessing as pp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--project", default=".", help="folder that contains runs/ and ptbxl_cache_v4_signedbeat/")
ap.add_argument("--ptbxl", required=True, help="PTB-XL folder (with ptbxl_database.csv)")
ap.add_argument("--out", default="ECG_MI_Demo")
ap.add_argument("--n-selftest", type=int, default=20)
args = ap.parse_args()
project, root, out = Path(args.project).resolve(), Path(args.ptbxl), Path(args.out).resolve()
ens = project / "runs" / "mi_sspnet_v5_ensemble"
cfg = json.loads((ens / "ensemble_config.json").read_text())

if out.exists():          # re-export: refresh everything but keep the git history (.git)
    for p in out.iterdir():
        if p.name != ".git":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
else:
    out.mkdir(parents=True)

# 1) example ECGs for the demo
print("[1/4] Copying example ECGs ...")
subprocess.run([sys.executable, str(APP / "prepare_demo.py"), "--project", str(project), "--ptbxl", str(root)], check=True)

# 2) app code (+ demo examples)
print("[2/4] Copying the app ...")
shutil.copytree(APP, out / "ecg_mi_app", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "export_bundle.py"))

# 2b) cardiologist labels of all PTB-XL records (lets the app check uploaded PTB-XL files)
subprocess.run([sys.executable, str(APP / "make_ptbxl_labels.py"), "--ptbxl", str(root),
                "--out", str(out / "ecg_mi_app" / "ptbxl_labels.csv")], check=True)

# 3) trained model files only
print("[3/4] Copying the trained networks ...")
for rd in cfg["run_dirs"]:
    src = project / rd if not Path(rd).is_absolute() else Path(rd)
    dst = out / "runs" / src.name
    dst.mkdir(parents=True)
    for f in ("best.pt", "config.json"):
        shutil.copy(src / f, dst / f)
    print(f"      {src.name}/best.pt  ({(src / 'best.pt').stat().st_size / 1e6:.1f} MB)")
cfg_out = dict(cfg, run_dirs=[f"runs/{Path(rd).name}" for rd in cfg["run_dirs"]])
(out / "runs" / ens.name).mkdir(parents=True)
(out / "runs" / ens.name / "ensemble_config.json").write_text(json.dumps(cfg_out, indent=2))

# 4) self-test pack: raw signals + the notebook's images and probabilities for a few validation records
print("[4/4] Building the self-test pack ...")
meta = pd.read_csv(project / "ptbxl_cache_v4_signedbeat" / "meta.csv")
images = np.load(project / "ptbxl_cache_v4_signedbeat" / "images.npy", mmap_mode="r")
saved = np.load(ens / "ensemble_val_predictions.npz")
pos = {int(e): k for k, e in enumerate(saved["ecg_ids"])}
val = meta.index[meta.strat_fold == 9].to_numpy()
pick = np.sort(np.random.default_rng(0).choice(val, size=min(args.n_selftest, len(val)), replace=False))
sigs, fss, names = [], [], None
for i in pick:
    s, fs, names = pp.read_wfdb(root / meta.filename_lr[i])
    sigs.append(s); fss.append(fs)
ids = meta.ecg_id.to_numpy()[pick]
np.savez_compressed(out / "ecg_mi_app" / "selftest_pack.npz",
                    signals=np.stack(sigs), fs=np.asarray(fss), lead_names=np.asarray(names),
                    images=np.stack([images[i] for i in pick]), ecg_ids=ids,
                    prob_raw=np.stack([saved["prob_raw"][pos[int(e)]] for e in ids]),
                    prob=np.stack([saved["prob"][pos[int(e)]] for e in ids]))

# laptop scripts
(out / "setup_linux_mac.sh").write_text(r'''#!/usr/bin/env bash
# One-time setup on the laptop (needs internet once, ~1 GB download). Ubuntu: sudo apt install python3-venv
set -e
cd "$(dirname "$0")"
${PYTHON:-python3} -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [ "$(uname)" = "Darwin" ]; then pip install torch; else pip install torch --index-url https://download.pytorch.org/whl/cpu; fi
pip install -r ecg_mi_app/requirements.txt
python ecg_mi_app/selftest.py
bash ecg_mi_app/make_desktop_shortcut.sh || true
echo "Setup finished. Start the demo with:  bash start_demo.sh"
''')
(out / "start_demo.sh").write_text('#!/usr/bin/env bash\nexec bash "$(dirname "$0")/ecg_mi_app/start_demo.sh"\n')
(out / "setup_windows.bat").write_text(r'''@echo off
rem One-time setup on a Windows laptop (needs internet once). Requires Python 3.9-3.12 from python.org
rem with "Add python.exe to PATH" ticked during installation.
cd /d "%~dp0"
python -m venv .venv || (echo Python not found. Install it from python.org and tick "Add to PATH". & pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r ecg_mi_app\requirements.txt
python ecg_mi_app\selftest.py
echo.
echo Setup finished. Start the demo by double-clicking start_demo.bat
pause
''')
(out / "start_demo.bat").write_text(r'''@echo off
cd /d "%~dp0"
set ECG_MI_PROJECT=%~dp0.
call .venv\Scripts\activate.bat
echo Starting ECG MI demo ... keep this window open; close it to stop the app.
python -m streamlit run ecg_mi_app\app.py --server.port 8501 --browser.gatherUsageStats false --theme.base light --theme.primaryColor "#B3122E"
pause
''')
# ---- files for GitHub + Streamlit Community Cloud
(out / ".streamlit").mkdir()
(out / ".streamlit" / "config.toml").write_text(
    '[theme]\nbase = "light"\nprimaryColor = "#B3122E"\n\n[browser]\ngatherUsageStats = false\n\n[server]\nrunOnSave = false\n')
(out / ".gitignore").write_text(".venv/\n__pycache__/\n*.pyc\n*.zip\n.DS_Store\n")
(out / "push_to_github.sh").write_text(r"""#!/usr/bin/env bash
# First run: creates the GitHub repository and pushes. Later runs: commit + push (the web app updates itself).
#   bash push_to_github.sh "message"         REPO_NAME=ecg-mi-demo  REPO_VISIBILITY=public|private
set -e
cd "$(dirname "$0")"
command -v git >/dev/null || { echo "Install git:  sudo apt install git"; exit 1; }
command -v gh  >/dev/null || { echo "Install the GitHub CLI:  sudo apt install gh"; exit 1; }
gh auth status >/dev/null 2>&1 || gh auth login --web --git-protocol https
gh auth setup-git >/dev/null 2>&1 || true
[ -d .git ] || { git init -q; git checkout -q -b main; }
if [ -z "$(git config user.name)" ]; then
  login=$(gh api user -q .login)
  git config user.name "$login"
  git config user.email "$login@users.noreply.github.com"
fi
git add -A
git commit -qm "${1:-Update ECG MI demo app}" || echo "Nothing new to commit."
if git remote get-url origin >/dev/null 2>&1; then
  git push -u origin main
else
  gh repo create "${REPO_NAME:-ecg-mi-demo}" --"${REPO_VISIBILITY:-public}" --source=. --remote=origin --push
fi
echo "Pushed: $(gh repo view --json url -q .url)"
""")
(out / "README.md").write_text("""# ECG myocardial infarction screening (NORM / IMI / ASMI)

Web app that classifies a 12-lead ECG as **normal**, **inferior MI** or **anteroseptal MI** with an ensemble of
three dual-branch attention CNNs (SSPNet) trained on PTB-XL. It shows the median heartbeat of every lead, which
leads drove the decision, and the exact images the network saw.

**Live app:** <add your https://....streamlit.app link here>

## Contents
- `ecg_mi_app/app.py`: Streamlit app (entry point)
- `ecg_mi_app/ecg_preprocessing.py`, `ecg_mi_app/sspnet_model.py`: preprocessing and model, identical to the training notebook
- `runs/`: trained weights (3 seeds) and the ensemble configuration (class offsets)
- `ecg_mi_app/demo_examples/`: example ECGs from PTB-XL (not used for training)

## Run locally
`bash setup_linux_mac.sh` then `bash start_demo.sh` (Windows: `setup_windows.bat`, `start_demo.bat`).

## Deploy / update
1. `bash push_to_github.sh` (first time: creates the repository).
2. https://share.streamlit.io → *Create app* → this repository, branch `main`, main file `ecg_mi_app/app.py`,
   *Advanced settings* → Python 3.11 → *Deploy*.
3. Afterwards every `git push` (or `bash push_to_github.sh "message"`) redeploys automatically.

## Data
Example ECGs: PTB-XL v1.0.3 (Wagner et al., *Scientific Data* 7:154, 2020), PhysioNet, CC BY 4.0.

**Research prototype for a course project, not a medical device. Not for clinical use.**
""")

for f in ("setup_linux_mac.sh", "start_demo.sh", "push_to_github.sh"):
    (out / f).chmod(0o755)

import zipfile  # noqa: E402
with zipfile.ZipFile(str(out) + ".zip", "w", zipfile.ZIP_DEFLATED) as z:   # laptop copy, without .git
    for p in sorted(out.rglob("*")):
        rel = p.relative_to(out.parent)
        if p.is_file() and ".git" not in p.relative_to(out).parts[:1]:
            z.write(p, rel)
size = Path(str(out) + ".zip").stat().st_size / 1e6
print(f"\nDone: {out}  (folder = your GitHub repository)  +  {out}.zip ({size:.0f} MB) for a laptop.")
print(f"Web app: cd \"{out}\" && bash push_to_github.sh   then deploy on share.streamlit.io")
