// Auto-generated from schema/match.schema.json — DO NOT EDIT MANUALLY
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
