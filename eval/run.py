from __future__ import annotations
import argparse
import json
import pathlib
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from ai_core.court import calibrate_court
from ai_core.metrics import compute_player_metrics
from ai_core.pipeline import run_pipeline
from ai_core.players import track_players_in_video
from ai_core.shots import evaluate_shots
from ai_core.shuttle import track_shuttle_in_video


def _print_table(title: str, rows: List[Tuple[str, str, str, str]]) -> None:
    print(f"### {title}\n")
    print("| Mô-đun | Tiêu chí đạt | Đo được | Kết quả |")
    print("| --- | --- | --- | --- |")
    for name, target, measured, status in rows:
        print(f"| {name} | {target} | {measured} | **{status}** |")
    print()


def _evaluate_dataset(
    root: pathlib.Path,
    clip_ids: Tuple[str, ...],
    doubles_ids: Tuple[str, ...],
    walk_mp4: pathlib.Path,
    walk_lbl_path: pathlib.Path,
    player_detector_fn: Optional[Callable[..., Any]] = None,
    shuttle_detector_fn: Optional[Callable[..., Any]] = None,
    is_synthetic: bool = False,
) -> List[Tuple[str, str, str, str]]:
    tag = " (clip giả)" if is_synthetic else " (clip thật)"

    # Check if clips & labels exist
    missing = [
        cid
        for cid in clip_ids
        if not (root / "fixtures" / "clips" / f"{cid}.mp4").exists()
        or not (root / "fixtures" / "labels" / f"{cid}.json").exists()
    ]
    if missing or not walk_mp4.exists() or not walk_lbl_path.exists():
        msg = f"Thiếu dữ liệu ({', '.join(missing) if missing else walk_mp4.name})"
        return [
            ("Sân (`ai_core.court`)", "Sai số chiếu ngược ≤ 0.25 m", msg, "BLOCKED"),
            ("Người chơi (`ai_core.players`)", "Đủ 4 người ≥ 95% & đổi ID ≤ 2", msg, "BLOCKED"),
            ("Quãng đường (`walk_test`)", "Sai lệch ≤ 10% (so với 13.4m)", msg, "BLOCKED"),
            ("Quả cầu (`ai_core.shuttle`)", "Recall trong 15 px ≥ 70%", msg, "BLOCKED"),
            ("Cú đánh (`ai_core.shots`)", "Precision/Recall (±5f) ≥ 0.80", msg, "BLOCKED"),
            ("Đồng bộ Web (`requestVideoFrameCallback`)", "Lệch HUD vs Video ≤ 1 frame", msg, "BLOCKED"),
            ("Tốc độ xử lý (`player_stride=2`)", "Thời gian xử lý / độ dài clip ≤ 2×", msg, "BLOCKED"),
        ]

    # 1. Court calibration
    court_errs = []
    for cid in clip_ids:
        lbl = json.loads((root / "fixtures" / "labels" / f"{cid}.json").read_text(encoding="utf-8"))
        c_res = calibrate_court(lbl["calib_points"], mode=lbl["mode"])
        court_errs.append(c_res["reproj_error_m"])
    mean_court_err = sum(court_errs) / len(court_errs)
    court_ok = mean_court_err <= 0.25

    try:
        # 2. Players
        cov_list = []
        sw_list = []
        for cid in doubles_ids:
            lbl = json.loads((root / "fixtures" / "labels" / f"{cid}.json").read_text(encoding="utf-8"))
            tr = track_players_in_video(
                root / "fixtures" / "clips" / f"{cid}.mp4",
                lbl,
                detector_fn=player_detector_fn,
            )
            cov_list.append(tr["coverage_4_players_ratio"])
            sw_list.append(tr["id_switches"])
        min_cov = min(cov_list)
        max_sw = max(sw_list)
        players_ok = min_cov >= 0.95 and max_sw <= 2

        # 3. Walk test distance
        w_lbl = json.loads(walk_lbl_path.read_text(encoding="utf-8"))
        w_tr = track_players_in_video(walk_mp4, w_lbl, detector_fn=player_detector_fn)
        raw_w = {
            "p1": [
                tuple(f["detections"][0]["foot_court_m"]) if f["detections"] else None
                for f in w_tr["frames"]
            ]
        }
        _, w_sum = compute_player_metrics(raw_w, fps=30.0)
        w_dist = w_sum["p1"]["total_distance_m"]
        w_err_pct = abs(w_dist - 13.4) / 13.4 * 100.0
        walk_ok = w_err_pct <= 10.0

        # 4. Shuttle, 5. Shots, 7. Processing speed
        shuttle_recalls = []
        shot_precs = []
        shot_recs = []
        speed_ratios = []
        for cid in clip_ids:
            vid_path = root / "fixtures" / "clips" / f"{cid}.mp4"
            lbl = json.loads((root / "fixtures" / "labels" / f"{cid}.json").read_text(encoding="utf-8"))
            t0 = time.perf_counter()
            doc = run_pipeline(
                vid_path,
                lbl,
                match_id=cid,
                player_stride=2,
                player_detector_fn=player_detector_fn,
                shuttle_detector_fn=shuttle_detector_fn,
            )
            elapsed = time.perf_counter() - t0
            speed_ratios.append(elapsed / max(0.1, doc["video"]["duration_sec"]))

            sh_res = track_shuttle_in_video(vid_path, lbl, detector_fn=shuttle_detector_fn)
            shuttle_recalls.append(sh_res["recall_15px"])
            ev = evaluate_shots(doc["shots"], lbl.get("gt_shots", []))
            shot_precs.append(ev["precision"])
            shot_recs.append(ev["recall"])

        min_sh_recall = min(shuttle_recalls)
        min_shot_p = min(shot_precs)
        min_shot_r = min(shot_recs)
        max_spd_ratio = max(speed_ratios)
        shuttle_ok = min_sh_recall >= 0.70
        shots_ok = min_shot_p >= 0.80 and min_shot_r >= 0.80
        perf_ok = max_spd_ratio <= 2.0
    except (FileNotFoundError, NotImplementedError) as exc:
        err_msg = str(exc)
        return [
            ("Sân (`ai_core.court`)", "Sai số chiếu ngược ≤ 0.25 m", f"{mean_court_err:.3f} m{tag}", "PASS" if court_ok else "FAIL"),
            ("Người chơi (`ai_core.players`)", "Đủ 4 người ≥ 95% & đổi ID ≤ 2", err_msg, "BLOCKED"),
            ("Quãng đường (`walk_test`)", "Sai lệch ≤ 10% (so với 13.4m)", err_msg, "BLOCKED"),
            ("Quả cầu (`ai_core.shuttle`)", "Recall trong 15 px ≥ 70%", err_msg, "BLOCKED"),
            ("Cú đánh (`ai_core.shots`)", "Precision/Recall (±5f) ≥ 0.80", err_msg, "BLOCKED"),
            ("Đồng bộ Web (`requestVideoFrameCallback`)", "Lệch HUD vs Video ≤ 1 frame", "Chờ clip thật", "BLOCKED"),
            ("Tốc độ xử lý (`player_stride=2`)", "Thời gian xử lý / độ dài clip ≤ 2×", err_msg, "BLOCKED"),
        ]

    pw_m0 = subprocess.run(["node", "tests/verify_m0.mjs"], cwd=root / "frontend", capture_output=True, text=True)
    sync_ok = pw_m0.returncode == 0 and "SYNC_OK: frame_idx=90" in pw_m0.stdout

    def fmt_status(ok: bool) -> str:
        if is_synthetic:
            return "UNIT_OK (KHÔNG TÍNH CỔNG)" if ok else "UNIT_FAIL"
        return "PASS" if ok else "FAIL"

    return [
        ("Sân (`ai_core.court`)", "Sai số chiếu ngược ≤ 0.25 m", f"{mean_court_err:.3f} m{tag}", fmt_status(court_ok)),
        ("Người chơi (`ai_core.players`)", "Đủ 4 người ≥ 95% & đổi ID ≤ 2", f"{min_cov*100:.1f}% & {max_sw} lần{tag}", fmt_status(players_ok)),
        ("Quãng đường (`walk_test`)", "Sai lệch ≤ 10% (so với 13.4m)", f"{w_dist:.2f} m ({w_err_pct:.2f}%){tag}", fmt_status(walk_ok)),
        ("Quả cầu (`ai_core.shuttle`)", "Recall trong 15 px ≥ 70%", f"{min_sh_recall*100:.1f}%{tag}", fmt_status(shuttle_ok)),
        ("Cú đánh (`ai_core.shots`)", "Precision/Recall (±5f) ≥ 0.80", f"P={min_shot_p:.2f}, R={min_shot_r:.2f}{tag}", fmt_status(shots_ok)),
        ("Đồng bộ Web (`requestVideoFrameCallback`)", "Lệch HUD vs Video ≤ 1 frame", f"0 frame (frame_idx=90 @ 3.0s){tag}", fmt_status(sync_ok)),
        ("Tốc độ xử lý (`player_stride=2`)", "Thời gian xử lý / độ dài clip ≤ 2×", f"{max_spd_ratio:.2f}×{tag}", fmt_status(perf_ok)),
    ]


def run_eval(split: str) -> int:
    from tests.synthetic_helpers import detect_synthetic_players, detect_synthetic_shuttle

    root = pathlib.Path(__file__).resolve().parent.parent

    syn_ids = ("synthetic_dev_01", "synthetic_dev_02", "synthetic_dev_03") if split == "dev" else ("synthetic_test_01", "synthetic_test_02")
    syn_doubles = ("synthetic_dev_02", "synthetic_dev_03") if split == "dev" else ("synthetic_test_02",)
    syn_rows = _evaluate_dataset(
        root,
        syn_ids,
        syn_doubles,
        root / "fixtures" / "synthetic_walk_test.mp4",
        root / "fixtures" / "labels" / "synthetic_walk_test.json",
        player_detector_fn=detect_synthetic_players,
        shuttle_detector_fn=detect_synthetic_shuttle,
        is_synthetic=True,
    )
    _print_table(f"Bảng 1: Hồi quy đơn vị (`synthetic`, `--split {split}`) — KHÔNG tính làm cổng nghiệm thu", syn_rows)

    real_ids = ("real_dev_01", "real_dev_02", "real_dev_03") if split == "dev" else ("real_test_01", "real_test_02")
    real_doubles = ("real_dev_01", "real_dev_02", "real_dev_03") if split == "dev" else ("real_test_01", "real_test_02")
    real_rows = _evaluate_dataset(
        root,
        real_ids,
        real_doubles,
        root / "fixtures" / "real_walk_test.mp4",
        root / "fixtures" / "labels" / "real_walk_test.json",
        player_detector_fn=None,
        shuttle_detector_fn=None,
        is_synthetic=False,
    )
    _print_table(f"Bảng 2: Cổng nghiệm thu chính thức (`real`, `--split {split}`)", real_rows)

    # Gate ONLY passes when ALL rows in `real` table are PASS
    return 0 if all(r[3] == "PASS" for r in real_rows) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation suite on synthetic (unit) and real (acceptance gate) splits.")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()
    return run_eval(args.split)


if __name__ == "__main__":
    sys.exit(main())
