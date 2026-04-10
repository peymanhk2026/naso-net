"""
VPI Sequence Extraction Script
================================
Extracts contiguous positive (VPI) and negative sequences from videos
based on annotation keyframes. Each sequence gets its own folder named
by frame range, e.g.:

  extracted_sequences/
    VPI-1/
      pos_219-411/
        219.jpg  220.jpg  ...  411.jpg
      neg_412-579/
        412.jpg  413.jpg  ...  579.jpg
      pos_580-689/
        ...

This is designed for sequence prediction tasks where each folder
represents one contiguous temporal segment.
"""

import json
import re
import sys
from pathlib import Path

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is not installed.")
    print("Run:  pip install opencv-python")
    sys.exit(1)

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────
BASE_DIR = Path(r"D:\PLOS ONE")
VIDEO_DIR = BASE_DIR / "VPI case videos"
ANNOT_DIR = BASE_DIR / "Annotations"
OUTPUT_DIR = VIDEO_DIR / "extracted_sequences"

ANNOTATION_FILES = [
    ANNOT_DIR / "1st_attempt" / "Case_Annotations" / "Annotations_Case_VPI_1_and_6.json",
    ANNOT_DIR / "4th-attempt" / "VPI_1_and_5-13-sequence-jmin.json",
    ANNOT_DIR / "4th-attempt" / "VPI2-4_sequence-jmin.json",
    ANNOT_DIR / "VPI_21-22.json",
    ANNOT_DIR / "VPI_35+37.json",
    # New annotations
    ANNOT_DIR / "VPI_14-15 17 19.json",
    ANNOT_DIR / "VPI_28-31 33.json",
]


# ──────────────────────────────────────────────────────────
# Helpers (shared with extract_frames.py)
# ──────────────────────────────────────────────────────────
def extract_video_name(video_path: str) -> str:
    filename = video_path.split("/")[-1]
    vpi_idx = filename.upper().find("VPI")
    if vpi_idx == -1:
        return filename.replace(".mp4", "")
    raw = filename[vpi_idx:].replace(".mp4", "")
    num_match = re.search(r"(\d+)", raw)
    if num_match:
        return f"VPI-{num_match.group(1)}"
    return raw


def determine_annotation_fps(frames_count: int, duration: float) -> int:
    if duration <= 0:
        return 30
    fps = frames_count / duration
    return 25 if abs(fps - 25) < abs(fps - 30) else 30


# ──────────────────────────────────────────────────────────
# Build contiguous sequences from keyframes
# ──────────────────────────────────────────────────────────
def build_sequences_from_keyframes(sequence: list, annot_fps: int,
                                   video_fps: float, total_frames: int):
    """
    Convert annotation keyframes into a list of contiguous segments.

    Each segment: { 'label': 'positive'|'negative',
                    'start_frame': int,   (1-based, video frame number)
                    'end_frame':   int }  (inclusive)

    The annotation keyframes use annotation-FPS frame numbers and times.
    We map them to actual video frames via time.
    """
    if not sequence:
        return []

    # Step 1: Build time-based intervals from annotation keyframes.
    # Merge consecutive keyframes with the same enabled state.
    intervals = []  # (start_time, end_time, enabled)
    current_enabled = sequence[0].get("enabled", True)
    current_start = sequence[0].get("time", (sequence[0]["frame"] - 1) / annot_fps)

    for i in range(1, len(sequence)):
        kf = sequence[i]
        kf_time = kf.get("time", (kf["frame"] - 1) / annot_fps)
        kf_enabled = kf.get("enabled", True)

        if kf_enabled != current_enabled:
            # State changed — close current interval
            intervals.append((current_start, kf_time, current_enabled))
            current_start = kf_time
            current_enabled = kf_enabled
        # If same state (e.g. true->true for bbox shift), just continue

    # Close the last interval to end of video
    video_duration = total_frames / video_fps if video_fps > 0 else 0
    intervals.append((current_start, video_duration + 1, current_enabled))

    # Step 2: Map time intervals to video frame numbers
    segments = []
    for start_t, end_t, enabled in intervals:
        start_frame = max(1, int(round(start_t * video_fps)) + 1)
        end_frame = min(total_frames, int(round(end_t * video_fps)))

        if end_frame < start_frame:
            continue

        label = "positive" if enabled else "negative"
        segments.append({
            "label": label,
            "start_frame": start_frame,
            "end_frame": end_frame,
        })

    # Step 3: Fix any gaps or overlaps between consecutive segments
    fixed = []
    for i, seg in enumerate(segments):
        if i == 0:
            seg["start_frame"] = 1  # ensure first segment starts at frame 1
        if i > 0:
            prev_end = fixed[-1]["end_frame"]
            seg["start_frame"] = prev_end + 1
        if seg["start_frame"] > seg["end_frame"]:
            continue
        fixed.append(seg)

    # Ensure last segment goes to the end
    if fixed and fixed[-1]["end_frame"] < total_frames:
        fixed[-1]["end_frame"] = total_frames

    return fixed


def get_bbox_at_time(time: float, sequence: list, annot_fps: int) -> dict:
    if not sequence:
        return {"x": 0, "y": 0, "width": 100, "height": 100}
    best = sequence[0]
    for kf in sequence:
        kf_time = kf.get("time", (kf["frame"] - 1) / annot_fps)
        if kf_time <= time:
            best = kf
        else:
            break
    return {
        "x": best.get("x", 0),
        "y": best.get("y", 0),
        "width": best.get("width", 100),
        "height": best.get("height", 100),
    }


# ──────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────
def load_annotations(annotation_files: list) -> dict:
    annotations = {}
    for annot_file in annotation_files:
        if not annot_file.exists():
            print(f"  [WARNING] Not found: {annot_file}")
            continue
        print(f"  Loading: {annot_file.relative_to(BASE_DIR)}")
        with open(annot_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            video_path = entry.get("video", "")
            video_name = extract_video_name(video_path)
            box = entry.get("box", [])
            if not box:
                continue
            box_data = box[0]
            frames_count = box_data.get("framesCount", 0)
            duration = box_data.get("duration", 0)
            sequence = box_data.get("sequence", [])
            annot_fps = determine_annotation_fps(frames_count, duration)
            annotations[video_name] = {
                "source_file": annot_file.name,
                "frames_count": frames_count,
                "duration": duration,
                "sequence": sequence,
                "annot_fps": annot_fps,
            }
            print(f"    {video_name}: {len(sequence)} keyframes, annot_fps={annot_fps}")
    return annotations


def find_video_files(video_dir: Path) -> dict:
    return {f.stem: f for f in sorted(video_dir.glob("*.mp4"))}


def extract_sequences(video_path: Path, video_name: str,
                      annot_data: dict, output_dir: Path):
    """Extract contiguous sequences of frames from a video."""
    print(f"\n  Processing {video_name}...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    [ERROR] Cannot open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    annot_fps = annot_data["annot_fps"]
    sequence = annot_data["sequence"]

    print(f"    Video: {total_frames} frames, {video_fps:.2f} fps")
    print(f"    Annot: {annot_data['frames_count']} frames, {annot_fps} fps "
          f"({annot_data['source_file']})")

    if not sequence:
        print(f"    [WARNING] Empty sequence, skipping")
        cap.release()
        return

    # Build contiguous segments
    segments = build_sequences_from_keyframes(
        sequence, annot_fps, video_fps, total_frames
    )

    pos_segs = [s for s in segments if s["label"] == "positive"]
    neg_segs = [s for s in segments if s["label"] == "negative"]
    print(f"    Segments: {len(segments)} total "
          f"({len(pos_segs)} positive, {len(neg_segs)} negative)")

    # Print segment table
    total_pos_frames = 0
    total_neg_frames = 0
    for seg in segments:
        n = seg["end_frame"] - seg["start_frame"] + 1
        tag = "POS" if seg["label"] == "positive" else "neg"
        if seg["label"] == "positive":
            total_pos_frames += n
        else:
            total_neg_frames += n
        print(f"      [{tag}] frames {seg['start_frame']:5d} - {seg['end_frame']:5d}  "
              f"({n} frames)")
    print(f"    Total: {total_pos_frames} positive, {total_neg_frames} negative frames")

    # Create output folders and build frame->folder mapping
    video_out = output_dir / video_name
    frame_to_folder = {}  # frame_num -> Path

    for seg in segments:
        prefix = "pos" if seg["label"] == "positive" else "neg"
        folder_name = f"{prefix}_{seg['start_frame']}-{seg['end_frame']}"
        folder_path = video_out / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)
        for fn in range(seg["start_frame"], seg["end_frame"] + 1):
            frame_to_folder[fn] = folder_path

    # Extract frames
    frame_num = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        folder = frame_to_folder.get(frame_num)
        if folder is None:
            continue

        # Crop to bounding box
        t = (frame_num - 1) / video_fps
        bbox = get_bbox_at_time(t, sequence, annot_fps)
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox["x"] / 100.0 * w))
        y1 = max(0, int(bbox["y"] / 100.0 * h))
        x2 = min(w, int((bbox["x"] + bbox["width"]) / 100.0 * w))
        y2 = min(h, int((bbox["y"] + bbox["height"]) / 100.0 * h))
        cropped = frame[y1:y2, x1:x2]

        out_path = folder / f"{frame_num}.jpg"
        cv2.imwrite(str(out_path), cropped)
        saved += 1

        if frame_num % 500 == 0:
            print(f"    Progress: {frame_num}/{total_frames}...")

    cap.release()
    print(f"    Saved {saved} frames -> {video_out}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  VPI Sequence Extraction (for sequence prediction)")
    print("=" * 65)

    print("\n[1] Loading annotations...")
    annotations = load_annotations(ANNOTATION_FILES)
    print(f"\n    Total annotated videos: {len(annotations)}")

    print("\n[2] Scanning for video files...")
    videos = find_video_files(VIDEO_DIR)
    print(f"    Found {len(videos)} .mp4 files")

    print("\n[3] Matching...")
    annotated = set(annotations.keys())
    available = set(videos.keys())
    matched = sorted(annotated & available)
    annot_only = sorted(annotated - available)
    video_only = sorted(available - annotated)

    print(f"    Matched: {len(matched)}")
    if annot_only:
        print(f"    Annotations WITHOUT video: {', '.join(annot_only)}")
    if video_only:
        print(f"    Videos WITHOUT annotations: {', '.join(video_only)}")

    print(f"\n[4] Extracting sequences to: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in matched:
        # Skip if already extracted
        existing = OUTPUT_DIR / name
        if existing.exists() and any(existing.rglob("*.jpg")):
            print(f"\n  [SKIP] {name} already extracted ({existing})")
            continue
        extract_sequences(videos[name], name, annotations[name], OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 65)
    print("  DONE")
    print("=" * 65)
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Videos processed: {len(matched)}")
    if annot_only:
        print(f"  Missing videos: {', '.join(annot_only)}")
    if video_only:
        print(f"  Missing annotations: {', '.join(video_only)}")
    print()


if __name__ == "__main__":
    main()
