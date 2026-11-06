from __future__ import annotations
import argparse
import json
import pathlib
import sys
from schema.models import MatchDocument

TS_TEMPLATE = """// Auto-generated from schema/match.schema.json — DO NOT EDIT MANUALLY
export interface VideoMeta {
  fps: number;
  total_frames: number;
  duration_sec: number;
  width: number;
  height: number;
}

export interface CourtDims {
  width: number;
  length: number;
  net_y: number;
}

export interface CalibPoint {
  px: [number, number];
  court_m: [number, number];
}

export interface CourtMeta {
  mode: "singles" | "doubles";
  dims_m: CourtDims;
  calib_points: CalibPoint[];
  homography_img_to_court: number[];
  reproj_error_m: number;
}

export interface PlayerInfo {
  id: string;
  name: string;
  team: "A" | "B";
  color: string;
}

export interface Rally {
  id: number;
  start_frame: number;
  end_frame: number;
  shot_count: number;
  winner_team?: "A" | "B" | null;
  score_after?: [number, number] | null;
}

export interface Shot {
  idx: number;
  rally_id: number;
  frame: number;
  player_id: string;
  shot_type?: "Smash" | "Clear" | "Drop" | "Drive" | "Serve" | "Lift" | null;
  speed_kmh: number | null;
  speed_confidence: number;
  from_court_m: [number, number];
  to_court_m: [number, number];
}

export interface ShuttleFrame {
  x_px: number | null;
  y_px: number | null;
  visible: boolean;
  conf: number;
}

export interface PlayerFrame {
  bbox_px: [number, number, number, number] | null;
  foot_px: [number, number] | null;
  foot_court_m: [number, number] | null;
  speed_mps: number;
  dist_cum_m: number;
  energy: number;
  conf: number;
}

export interface FrameRecord {
  i: number;
  t: number;
  shuttle: ShuttleFrame;
  players: Record<string, PlayerFrame>;
}

export interface ZonesPct {
  front: number;
  mid: number;
  back: number;
}

export interface PlayerSummary {
  total_distance_m: number;
  avg_speed_mps: number;
  max_speed_mps: number;
  zones_pct: ZonesPct;
}

export interface MatchDocument {
  schema_version: "2.0";
  match_id: string;
  video: VideoMeta;
  court: CourtMeta;
  players: PlayerInfo[];
  rallies: Rally[];
  shots: Shot[];
  frames: FrameRecord[];
  summary: Record<string, PlayerSummary>;
}
"""


def export_artifacts() -> None:
    root = pathlib.Path(__file__).resolve().parent.parent
    schema_path = root / "schema" / "match.schema.json"
    schema_path.write_text(
        json.dumps(MatchDocument.model_json_schema(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ts_path = root / "frontend" / "src" / "types" / "match.ts"
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    ts_path.write_text(TS_TEMPLATE, encoding="utf-8")


def validate_file(path: pathlib.Path) -> MatchDocument:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return MatchDocument.model_validate(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate match JSON against v2.0 schema.")
    parser.add_argument("file", nargs="?", help="Path to match JSON file")
    parser.add_argument("--export", action="store_true", help="Export JSON Schema and TypeScript types")
    args = parser.parse_args()

    if args.export:
        export_artifacts()
        print("Exported schema/match.schema.json and frontend/src/types/match.ts")
        if not args.file:
            return 0

    if not args.file:
        parser.error("Provide a JSON file path or --export")

    try:
        doc = validate_file(pathlib.Path(args.file))
        print(f"OK: {doc.match_id} ({doc.video.total_frames} frames, {len(doc.shots)} shots)")
        return 0
    except Exception as exc:
        print(f"VALIDATION ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

