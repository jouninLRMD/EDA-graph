#!/usr/bin/env python
"""Generate a synthetic dataset in the CASE folder format.

This is used solely to integration-test the extraction + classification
pipeline when the real dataset is not available on the machine running
the code. It emits

    <output>/annotations/sub_<i>.csv      (jstime, valence, arousal, video)
    <output>/physiological/sub_<i>.csv    (daqtime, gsr, video)

with per-class amplitude / phasic-burst characteristics so that the
classifier has *something* to learn. It is NOT a substitute for the
real data, just a smoke test.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_PROFILES = {
    # video_id: (class, baseline_SCL, scr_rate_per_min, scr_amp, drift)
    10: (0, 5.0, 2.0, 0.2, 0.0),   # Neutral start
    11: (0, 5.0, 2.0, 0.2, 0.0),   # Neutral end
    1:  (1, 6.0, 8.0, 0.6, 0.02),  # Amused - high arousal pos
    2:  (1, 6.1, 8.0, 0.6, 0.02),
    3:  (2, 4.8, 1.5, 0.15, -0.01),# Bored - low arousal neg
    4:  (2, 4.8, 1.5, 0.15, -0.01),
    5:  (3, 5.2, 2.5, 0.3, 0.0),   # Relaxed - low arousal pos
    6:  (3, 5.2, 2.5, 0.3, 0.0),
    7:  (4, 7.0, 12.0, 0.8, 0.05), # Scared - high arousal neg
    8:  (4, 7.0, 12.0, 0.8, 0.05),
}

VIDEO_ORDER = [10, 1, 2, 3, 4, 5, 6, 7, 8, 11]   # like the CASE protocol
VIDEO_DURATIONS_S = {10: 30, 11: 30, **{k: 120 for k in (1,2,3,4,5,6,7,8)}}


def _subject_signal(seed: int, fs: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    eda_rows = []
    ann_rows = []
    t0 = 0.0
    for vid in VIDEO_ORDER:
        cls, baseline, rate, amp, drift = CLASS_PROFILES[vid]
        dur = VIDEO_DURATIONS_S[vid]
        n = int(dur * fs)
        t = t0 + np.arange(n) / fs

        # Tonic baseline + slow drift specific to class.
        scl = baseline + drift * (t - t0) + 0.05 * np.sin(2 * np.pi * 0.01 * t)
        # Phasic bursts.
        n_scrs = max(int(dur / 60 * rate), 1)
        bumps = np.zeros(n)
        for tau in rng.uniform(t0 + 1, t0 + dur - 1, size=n_scrs):
            bumps += amp * np.exp(-((t - tau) ** 2) / (2 * 0.8 ** 2))
        noise = 0.03 * rng.normal(size=n)
        gsr = scl + bumps + noise
        video = np.full(n, vid, dtype=np.int32)

        eda_rows.append(pd.DataFrame({"daqtime": t, "gsr": gsr, "video": video}))
        # Annotations at 20 Hz (CASE convention).
        ann_fs = 20
        ann_t = np.arange(int(dur * ann_fs)) / ann_fs + t0
        val = 5 + rng.normal(scale=0.3, size=ann_t.size) + (1.0 if cls in (1, 3) else -1.0 if cls in (2, 4) else 0)
        aro = 5 + rng.normal(scale=0.3, size=ann_t.size) + (1.5 if cls in (1, 4) else -1.0 if cls in (2, 3) else 0)
        ann_rows.append(pd.DataFrame({"jstime": ann_t, "valence": val, "arousal": aro, "video": vid}))
        t0 += dur

    return pd.concat(ann_rows, ignore_index=True), pd.concat(eda_rows, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, help="Output folder (will contain annotations/ and physiological/).")
    ap.add_argument("--n-subjects", type=int, default=30)
    args = ap.parse_args()

    out = Path(args.output)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    (out / "physiological").mkdir(parents=True, exist_ok=True)

    for i in range(1, args.n_subjects + 1):
        ann, phys = _subject_signal(seed=i)
        ann.to_csv(out / "annotations" / f"sub_{i}.csv", index=False)
        phys.to_csv(out / "physiological" / f"sub_{i}.csv", index=False)
        print(f"  wrote sub_{i}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
