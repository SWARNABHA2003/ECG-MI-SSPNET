"""
Real-patient ECG figures from PTB-XL for the presentation (no synthetic signals). Standalone: needs only
    pip install wfdb numpy scipy pandas matplotlib

Any computer with internet (downloads only the database table + ~150 small records from PhysioNet):
    python3 make_real_ecg_figures.py
Or, if PTB-XL is already on the computer:
    python3 make_real_ecg_figures.py --ptbxl /path/to/ptb-xl/1.0.3

Writes to real_ecg_figures/:
  real_normal_beat.png   labelled median beat (lead II) of a real normal ECG
  real_imi.png           NORM vs IMI (acute stage) vs IMI (old stage): leads II, III, aVF (facing) + I, aVL (reciprocal)
  real_asmi.png          NORM vs ASMI (acute) vs ASMI (old): leads V1-V4
  real_12lead_imi.png    full 10-s 12-lead ECG of the selected IMI record
  real_12lead_asmi.png   full 10-s 12-lead ECG of the selected ASMI record
  selected_records.csv   which PTB-XL records were used (for the citation) + top-5 alternatives per panel

Stages come from PTB-XL's own cardiologist annotation (columns infarction_stadium1/2):
  Stadium I = acute, Stadium II = evolving / subacute, Stadium III = old (chronic).
Records are taken from folds 9-10 (not used for training) and preferably validated by a human cardiologist.
Data: PTB-XL v1.0.3 (Wagner et al., Sci. Data 2020), PhysioNet, CC BY 4.0.
"""
import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, find_peaks, sosfiltfilt, resample_poly

ap = argparse.ArgumentParser()
ap.add_argument("--ptbxl", default=None, help="local PTB-XL folder; omit to download from PhysioNet")
ap.add_argument("--out", default="real_ecg_figures")
ap.add_argument("--folds", default="9,10", help="folds to pick examples from (default: not used for training)")
ap.add_argument("--max-candidates", type=int, default=None, help="records scored per panel (default 40 online, 400 local)")
args = ap.parse_args()
ONLINE = args.ptbxl is None
PN = "ptb-xl/1.0.3"
BASE = f"https://physionet.org/files/{PN}/"
root, out = (Path(args.ptbxl) if args.ptbxl else None), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
folds = [int(f) for f in args.folds.split(",")]
MAXC = args.max_candidates or (40 if ONLINE else 400)


class pp:  # the project's preprocessing steps (same constants as ecg_preprocessing.py)
    FS, PRE, POST = 100, 30, 60
    BEAT_LEN = PRE + POST
    LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    _SOS_ECG = butter(3, [0.5, 40.0], btype='bandpass', fs=100, output='sos')
    _SOS_QRS = butter(2, [5.0, 18.0], btype='bandpass', fs=100, output='sos')


def read_record(rel):
    import wfdb
    if ONLINE:
        d, name = rel.rsplit("/", 1)
        sig, info = wfdb.rdsamp(name, pn_dir=f"{PN}/{d}")
    else:
        sig, info = wfdb.rdsamp(str(root / rel))
    names = [n.upper() for n in info["sig_name"]]
    sig = np.asarray(sig, float)[:, [names.index(n.upper()) for n in pp.LEADS]]
    if int(round(info["fs"])) != pp.FS:
        sig = resample_poly(sig, pp.FS, int(round(info["fs"])), axis=0)
    return np.nan_to_num(sig[:10 * pp.FS])


def median_beat_mv(sig10):
    f = sosfiltfilt(pp._SOS_ECG, sig10, axis=0)
    env = np.sqrt((sosfiltfilt(pp._SOS_QRS, f, axis=0) ** 2).sum(1))
    env = np.convolve(env, np.ones(8) / 8, mode="same")
    peaks, _ = find_peaks(env, height=0.35 * np.percentile(env, 99), distance=30)
    ok = [p for p in peaks if p - pp.PRE >= 0 and p + pp.POST <= len(f)]
    if not ok:
        return None, 0, f
    beat = np.median(np.stack([f[p - pp.PRE:p + pp.POST] for p in ok]), axis=0)
    beat = beat - beat[pp.PRE - 12:pp.PRE - 4].mean(0, keepdims=True)
    return beat, len(ok), f

NAVY, RED, INK = "#1E2A44", "#B3122E", "#1B1F2A"
PMAJ, PMIN = "#F2C4CB", "#FBE6E9"
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 220, "savefig.bbox": "tight"})
L = {n: i for i, n in enumerate(pp.LEADS)}
R = pp.PRE                      # R peak index in the median beat (100 Hz)
t_beat = (np.arange(pp.BEAT_LEN) - R) / pp.FS

# ------------------------------------------------------------------ cohort
src = BASE if ONLINE else str(root) + "/"
print("Reading the PTB-XL tables from", src)
db = pd.read_csv(src + "ptbxl_database.csv", index_col="ecg_id")
scp = pd.read_csv(src + "scp_statements.csv", index_col=0)
diag = set(scp.index[scp.diagnostic == 1])
sup = scp.diagnostic_class.to_dict()


def label(codes):
    c = [k for k in ast.literal_eval(codes) if k in diag]
    return c[0] if len(c) == 1 and len({sup.get(x) for x in c}) == 1 and c[0] in ("NORM", "IMI", "ASMI") else None


db["label"] = db.scp_codes.map(label)
for col in ("infarction_stadium1", "infarction_stadium2"):
    if col not in db:
        db[col] = "unknown"
db["stage"] = db.infarction_stadium1.fillna("unknown").astype(str)
db = db[db.label.notna() & db.strat_fold.isin(folds)]
hv = db.validated_by_human if "validated_by_human" in db else pd.Series(True, index=db.index)
db = db.assign(human=hv.astype(bool))
print("Candidates per class/stage:\n", db.groupby(["label", "stage"]).size().to_string())

cache = {}


def load(eid):
    if eid not in cache:
        beat, nb, filt = median_beat_mv(read_record(db.loc[eid, "filename_lr"]))
        cache[eid] = (beat, nb, filt)
    return cache[eid]


def feats(beat, leads):
    idx = [L[x] for x in leads]
    st = beat[R + 11, idx].mean()                      # ~110 ms after R (ST segment)
    q = beat[R - 6:R + 1, idx].min(0).mean()           # deepest dip in the 60 ms before R
    r = beat[R - 2:R + 3, idx].max(0).mean()
    return st, q, r


def pick(cls, stage_ok, score, n=5):
    c = db[(db.label == cls) & stage_ok(db.stage)]
    if c.empty:
        return []
    c = c.sort_values("human", ascending=False)
    rows = []
    print(f"  scoring up to {MAXC} {cls} records ...", flush=True)
    for eid in c.index[:MAXC]:
        try:
            beat, nb, _ = load(eid)
        except Exception as e:
            print("   skip", eid, type(e).__name__)
            continue
        if beat is None or nb < 6 or np.abs(beat).max() > 4:
            continue
        rows.append((score(beat), eid))
    rows.sort(reverse=True)
    return [e for _, e in rows[:n]]


FACE = {"IMI": ["II", "III", "aVF"], "ASMI": ["V1", "V2", "V3", "V4"]}
RECIP = {"IMI": ["I", "aVL"], "ASMI": []}
is_acute = lambda s: s.str.strip().eq("Stadium I")
is_old = lambda s: s.str.strip().eq("Stadium III")
any_stage = lambda s: s == s

sel, records = {}, []
# normal: typical, flat ST, clear R in II
sel["NORM"] = pick("NORM", any_stage, lambda b: -abs(b[R + 11]).sum() - abs(b[R, L["II"]] - 1.0))
for cls in ("IMI", "ASMI"):
    f = FACE[cls]
    acute_score = lambda b, f=f, cls=cls: feats(b, f)[0] - (feats(b, RECIP[cls])[0] if RECIP[cls] else 0)
    old_score = lambda b, f=f: -feats(b, f)[1] - 2 * abs(feats(b, f)[0])
    sel[cls + "_acute"] = pick(cls, is_acute, acute_score)
    sel[cls + "_old"] = pick(cls, is_old, old_score)
    if not sel[cls + "_acute"]:                         # no stage annotation: fall back to the pattern itself
        print(f"[note] no 'Stadium I' {cls} in folds {folds}; using the strongest ST-elevation {cls} record")
        sel[cls + "_acute"] = pick(cls, any_stage, acute_score)
    if not sel[cls + "_old"]:
        print(f"[note] no 'Stadium III' {cls} in folds {folds}; using the clearest Q-wave {cls} record")
        sel[cls + "_old"] = pick(cls, any_stage, old_score)
    sel[cls + "_old"] = [e for e in sel[cls + "_old"] if e not in sel[cls + "_acute"][:1]]

for key, ids in sel.items():
    for rank, eid in enumerate(ids):
        records.append(dict(panel=key, rank=rank + 1, ecg_id=eid, label=db.loc[eid, "label"], stage=db.loc[eid, "stage"],
                            fold=int(db.loc[eid, "strat_fold"]), validated_by_human=bool(db.loc[eid, "human"]),
                            age=db.loc[eid].get("age"), sex=db.loc[eid].get("sex"), file=db.loc[eid, "filename_lr"]))
pd.DataFrame(records).to_csv(out / "selected_records.csv", index=False)
print("\nUsing:", {k: (v[0] if v else None) for k, v in sel.items()})


# ------------------------------------------------------------------ drawing helpers
def paper(ax, x0, x1, y0, y1):
    for x in np.arange(np.floor(x0 / 0.04) * 0.04, x1 + 1e-9, 0.04):
        ax.axvline(x, color=PMIN if round(x / 0.04) % 5 else PMAJ, lw=0.5, zorder=0)
    for y in np.arange(np.floor(y0 / 0.1) * 0.1, y1 + 1e-9, 0.1):
        ax.axhline(y, color=PMIN if round(y / 0.1) % 5 else PMAJ, lw=0.5, zorder=0)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect(0.4); ax.axis("off")


def stage_text(eid):
    s = db.loc[eid, "stage"]
    return {"Stadium I": "acute (Stadium I)", "Stadium III": "old (Stadium III)"}.get(s.strip(), s)


def beat_grid(name, cls, leads, facing):
    rows = [("Normal", sel["NORM"], INK), (f"{cls}, acute", sel[cls + "_acute"], RED), (f"{cls}, old", sel[cls + "_old"], NAVY)]
    rows = [r for r in rows if r[1]]
    beats = [load(r[1][0])[0] for r in rows]
    y0 = min(-1.0, min(b[:, [L[x] for x in leads]].min() for b in beats) - 0.1)
    y1 = max(1.2, max(b[:, [L[x] for x in leads]].max() for b in beats) + 0.1)
    fig, axs = plt.subplots(len(rows), len(leads), figsize=(2.5 * len(leads), 2.6 * len(rows) * (y1 - y0) / 2.4), squeeze=False)
    for i, ((title, ids, c), b) in enumerate(zip(rows, beats)):
        eid = ids[0]
        for j, ld in enumerate(leads):
            ax = axs[i, j]
            paper(ax, t_beat[0], t_beat[-1], y0, y1)
            ax.plot(t_beat, b[:, L[ld]], color=c, lw=2.0, zorder=3)
            if i == 0:
                ax.set_title(ld + ("" if ld in facing else "  (reciprocal)"), fontsize=13, fontweight="bold",
                             color=RED if ld in facing else NAVY)
        st = db.loc[eid, "stage"].strip()
        if title == "Normal":
            lab = "Normal"
        elif st in ("Stadium I", "Stadium III"):
            lab = f"{cls} · {stage_text(eid)}"
        else:   # no stage annotation: describe the pattern instead
            lab = f"{cls} · " + ("ST-elevation pattern" if "acute" in title else "Q-wave pattern") + " (stage not annotated)"
        axs[i, 0].text(t_beat[0] + 0.01, y1 - 0.05, lab, fontsize=11, fontweight="bold", color=c, va="top",
                       bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)
        axs[i, -1].text(t_beat[-1] - 0.01, y0 + 0.05, f"PTB-XL ecg_id {eid}", fontsize=8.5, color=INK, ha="right",
                        va="bottom", bbox=dict(fc="white", ec="none", pad=1), zorder=5)
    fig.subplots_adjust(wspace=0.03, hspace=0.08)
    fig.savefig(out / name, facecolor="white"); plt.close(fig)
    print("wrote", out / name)


def twelve_lead(name, eid, facing):
    _, _, x = load(eid)
    t = np.arange(len(x)) / pp.FS
    layout = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
    fig, ax = plt.subplots(figsize=(16, 7.6))
    gap, seg = 2.2, 2.5
    y0, y1 = -gap * 3 - 1.2, 1.3
    paper(ax, 0, 10, y0, y1)
    for r, row in enumerate(layout):
        for c, ld in enumerate(row):
            m = (t >= c * seg) & (t < (c + 1) * seg)
            ax.plot(t[m], x[m, L[ld]] - r * gap, color=INK, lw=1.0, zorder=3)
            ax.text(c * seg + 0.05, 0.75 - r * gap, ld, fontsize=12, fontweight="bold", zorder=5,
                    color=RED if ld in facing else INK)
    ax.plot(t, x[:, L["II"]] - 3 * gap, color=INK, lw=1.0, zorder=3)
    ax.text(0.05, 0.75 - 3 * gap, "II (rhythm)", fontsize=12, fontweight="bold", zorder=5, color=RED if "II" in facing else INK)
    ax.text(10, y0 + 0.1, f"PTB-XL ecg_id {eid} · {db.loc[eid, 'label']} · {stage_text(eid)} · 25 mm/s, 10 mm/mV, 0.5-40 Hz",
            ha="right", va="bottom", fontsize=9, color=INK, bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)
    fig.savefig(out / name, facecolor="white"); plt.close(fig)
    print("wrote", out / name)


def labelled_normal(name):
    eid = sel["NORM"][0]
    b = load(eid)[0][:, L["II"]]
    q = R - 6 + int(np.argmin(b[R - 6:R + 1])); s = R + int(np.argmin(b[R:R + 7]))
    p = R - 25 + int(np.argmax(b[R - 25:R - 8])); tt = R + 15 + int(np.argmax(np.abs(b[R + 15:R + 45])))
    y0, y1 = min(-0.6, b.min() - 0.15), max(1.3, b.max() + 0.25)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    paper(ax, t_beat[0], t_beat[-1], y0, y1)
    ax.plot(t_beat, b, color=INK, lw=2.4, zorder=3)
    for txt, i, dy in [("P", p, 0.1), ("Q", q, -0.17), ("R", R, 0.08), ("S", s, -0.17), ("T", tt, 0.1)]:
        ax.text(t_beat[i], b[i] + dy, txt, ha="center", fontsize=16, fontweight="bold", color=NAVY, zorder=5)
    j0, j1 = t_beat[min(s + 2, len(b) - 1)], t_beat[R + 15]
    ax.annotate("", xy=(j0, y0 + 0.15), xytext=(j1, y0 + 0.15), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
    ax.text((j0 + j1) / 2, y0 + 0.2, "ST segment", ha="center", color=RED, fontsize=11, fontweight="bold")
    ax.text(t_beat[-1] - 0.01, y1 - 0.05, f"Lead II median beat · PTB-XL ecg_id {eid}", ha="right", va="top", fontsize=9,
            color=INK, bbox=dict(fc="white", ec="none", pad=1), zorder=5)
    fig.savefig(out / name, facecolor="white"); plt.close(fig)
    print("wrote", out / name)


labelled_normal("real_normal_beat.png")
beat_grid("real_imi.png", "IMI", ["II", "III", "aVF", "I", "aVL"], FACE["IMI"])
beat_grid("real_asmi.png", "ASMI", ["V1", "V2", "V3", "V4"], FACE["ASMI"])
twelve_lead("real_12lead_imi.png", (sel["IMI_acute"] or sel["IMI_old"])[0], FACE["IMI"])
twelve_lead("real_12lead_asmi.png", (sel["ASMI_acute"] or sel["ASMI_old"])[0], FACE["ASMI"])
print(f"\nDone. Upload the 5 PNG files and selected_records.csv from {out.resolve()}/ to the chat.")
