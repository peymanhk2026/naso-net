"""
VPI Frame Extraction Script
============================
Reads annotation JSON files, matches them to video files (.mp4),
determines FPS from framesCount/duration, extracts all frames,
and classifies them as positive/negative based on the 'enabled' field
in annotation keyframes.

Frames are cropped to the annotated bounding box (removes black borders).

Output structure:
  D:\PLOS ONE\VPI case videos\extracted_frames\
    VPI-1\
      positive\
        <frame_number>.jpg
      negative\
        <frame_number>.jpg
    VPI-2\
      ...
"""

import json
import os
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
OUTPUT_DIR = VIDEO_DIR / "extracted_frames"

# Annotation files in priority order (later overrides earlier for same video)
ANNOTATION_FILES = [
    # 1st attempt – lowest priority
    ANNOT_DIR / "1st_attempt" / "Case_Annotations" / "Annotations_Case_VPI_1_and_6.json",
    # 4th attempt – higher priority, overrides 1st attempt for VPI-1 and VPI-6
    ANNOT_DIR / "4th-attempt" / "VPI_1_and_5-13-sequence-jmin.json",
    ANNOT_DIR / "4th-attempt" / "VPI2-4_sequence-jmin.json",
    # Standalone annotation files (unique videos, no conflicts)
    ANNOT_DIR / "VPI_21-22.json",
    ANNOT_DIR / "VPI_35+37.json",
    # New annotations
    ANNOT_DIR / "VPI_14-15 17 19.json",
    ANNOT_DIR / "VPI_28-31 33.json",
]


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
def extract_video_name(video_path: str) -> str:
    """
    Normalize the ugly annotation video path to a clean VPI-N name.

    Examples:
      '/data/upload/1/ec750c8d-VPI_21.mp4'     -> 'VPI-21'
      '/data/upload/2/e0e6e48f-VPI_-_1.mp4'    -> 'VPI-1'
      '/data/upload/19/089d7112-VPI-2.mp4'      -> 'VPI-2'
      '/data/upload/8/870c573a-VPI-10.mp4'      -> 'VPI-10'
    """
    filename = video_path.split("/")[-1]  # e.g. 'ec750c8d-VPI_21.mp4'

    # Remove everything before 'VPI'
    vpi_idx = filename.upper().find("VPI")
    if vpi_idx == -1:
        return filename.replace(".mp4", "")
    raw = filename[vpi_idx:]  # e.g. 'VPI_21.mp4', 'VPI_-_1.mp4', 'VPI-2.mp4'
    raw = raw.replace(".mp4", "")  # 'VPI_21', 'VPI_-_1', 'VPI-2'

    # Extract the number after VPI and any separators
    num_match = re.search(r"(\d+)", raw)
    if num_match:
        return f"VPI-{num_match.group(1)}"
    return raw


def determine_annotation_fps(frames_count: int, duration: float) -> int:
    """Determine if annotation was done at 25 or 30 fps."""
    if duration <= 0:
        return 30
    fps = frames_count / duration
    return 25 if abs(fps - 25) < abs(fps - 30) else 30


def build_time_segments(sequence: list, annot_fps: int):
    """
    Build a list of (start_time, end_time, enabled) segments from keyframes.

    The annotation sequence defines keyframes. Between consecutive keyframes,
    the 'enabled' state of the earlier keyframe applies.
    """
    if not sequence:
        return []

    segments = []
    for i, kf in enumerate(sequence):
        start_time = kf.get("time", (kf["frame"] - 1) / annot_fps)
        enabled = kf.get("enabled", True)

        if i + 1 < len(sequence):
            next_kf = sequence[i + 1]
            end_time = next_kf.get("time", (next_kf["frame"] - 1) / annot_fps)
        else:
            end_time = float("inf")

        segments.append((start_time, end_time, enabled))

    return segments


def classify_frame(time: float, segments: list) -> str:
    """Return 'positive' or 'negative' for a given time."""
    for start_t, end_t, enabled in segments:
        if start_t <= time < end_t:
            return "positive" if enabled else "negative"
    return "negative"


def get_bbox_at_time(time: float, sequence: list, annot_fps: int) -> dict:
    """
    Get the bounding box for a specific time by finding the nearest
    preceding keyframe. Falls back to the first keyframe.
    """
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
# Core Logic
# ──────────────────────────────────────────────────────────
def load_annotations(annotation_files: list) -> dict:
    """
    Load all annotation JSON files.
    Returns dict: { 'VPI-N': { source_file, frames_count, duration,
                                sequence, annot_fps } }
    Later files override earlier ones on name collision (priority order).
    """
    annotations = {}

    for annot_file in annotation_files:
        if not annot_file.exists():
            print(f"  [WARNING] Annotation file not found: {annot_file}")
            continue

        print(f"  Loading: {annot_file.relative_to(BASE_DIR)}")

        with open(annot_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data:
            video_path = entry.get("video", "")
            video_name = extract_video_name(video_path)
            box = entry.get("box", [])

            if not box:
                print(f"    [WARNING] No box data for {video_name}, skipping")
                continue

            box_data = box[0]
            frames_count = box_data.get("framesCount", 0)
            duration = box_data.get("duration", 0)
            sequence = box_data.get("sequence", [])
            annot_fps = determine_annotation_fps(frames_count, duration)

            annotations[video_name] = {
                "source_file": annot_file.name,
                "original_video_path": video_path,
                "frames_count": frames_count,
                "duration": duration,
                "sequence": sequence,
                "annot_fps": annot_fps,
            }
            print(
                f"    {video_name}: {frames_count} frames, {duration:.2f}s, "
                f"annot_fps={annot_fps}, {len(sequence)} keyframes"
            )

    return annotations


def find_video_files(video_dir: Path) -> dict:
    """Scan for .mp4 files. Returns dict: { 'VPI-N': Path }."""
    videos = {}
    for f in sorted(video_dir.glob("*.mp4")):
        videos[f.stem] = f
    return videos


def extract_frames(
    video_path: Path, video_name: str, annot_data: dict, output_dir: Path
):
    """Extract all frames from a video, classify and save them."""
    print(f"\n  Processing {video_name}...")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    [ERROR] Cannot open video: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / video_fps if video_fps > 0 else 0

    annot_fps = annot_data["annot_fps"]
    sequence = annot_data["sequence"]

    print(f"    Video  : {total_frames} frames, {video_fps:.2f} fps, {video_duration:.2f}s")
    print(
        f"    Annot  : {annot_data['frames_count']} frames, {annot_fps} fps, "
        f"{annot_data['duration']:.2f}s  (from {annot_data['source_file']})"
    )

    if not sequence:
        print(f"    [WARNING] Empty annotation sequence for {video_name}, skipping")
        cap.release()
        return

    # Build time-based segments for classification
    segments = build_time_segments(sequence, annot_fps)

    # Pre-classify all frames
    pos_count = 0
    neg_count = 0
    labels = {}
    for fn in range(1, total_frames + 1):
        t = (fn - 1) / video_fps
        label = classify_frame(t, segments)
        labels[fn] = label
        if label == "positive":
            pos_count += 1
        else:
            neg_count += 1

    print(f"    Labels : {pos_count} positive, {neg_count} negative")

    # Create output dirs
    video_out = output_dir / video_name
    pos_dir = video_out / "positive"
    neg_dir = video_out / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    frame_num = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        t = (frame_num - 1) / video_fps
        label = labels.get(frame_num, "negative")

        # Get bounding box for this time
        bbox = get_bbox_at_time(t, sequence, annot_fps)

        # Crop (bbox values are percentages)
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox["x"] / 100.0 * w))
        y1 = max(0, int(bbox["y"] / 100.0 * h))
        x2 = min(w, int((bbox["x"] + bbox["width"]) / 100.0 * w))
        y2 = min(h, int((bbox["y"] + bbox["height"]) / 100.0 * h))

        cropped = frame[y1:y2, x1:x2]

        if label == "positive":
            out_path = pos_dir / f"{frame_num}.jpg"
        else:
            out_path = neg_dir / f"{frame_num}.jpg"

        cv2.imwrite(str(out_path), cropped)
        saved += 1

        if frame_num % 500 == 0:
            print(f"    Progress: {frame_num}/{total_frames} frames...")

    cap.release()
    print(f"    Saved {saved} frames -> {video_out}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  VPI Frame Extraction")
    print("=" * 65)

    # ── Step 1: Load annotations ──
    print("\n[1] Loading annotations...")
    annotations = load_annotations(ANNOTATION_FILES)
    print(f"\n    Total annotated videos: {len(annotations)}")
    for name in sorted(annotations.keys()):
        print(f"      {name}  <-  {annotations[name]['source_file']}")

    # ── Step 2: Find video files ──
    print("\n[2] Scanning for video files...")
    videos = find_video_files(VIDEO_DIR)
    print(f"    Found {len(videos)} .mp4 files:")
    for name in sorted(videos.keys()):
        print(f"      {name}")

    # ── Step 3: Match & report ──
    print("\n[3] Matching annotations <-> videos...")
    annotated_names = set(annotations.keys())
    video_names = set(videos.keys())

    matched = sorted(annotated_names & video_names)
    annot_only = sorted(annotated_names - video_names)
    video_only = sorted(video_names - annotated_names)

    print(f"\n    MATCHED ({len(matched)} videos):")
    for name in matched:
        print(f"      [OK] {name}")

    if annot_only:
        print(f"\n    ANNOTATIONS WITHOUT VIDEO ({len(annot_only)}):")
        for name in annot_only:
            src = annotations[name]["source_file"]
            print(f"      [MISSING VIDEO] {name}  (annotated in {src})")

    if video_only:
        print(f"\n    VIDEOS WITHOUT ANNOTATIONS ({len(video_only)}):")
        for name in video_only:
            print(f"      [NO ANNOT] {name}  ({videos[name].name})")

    # ── Step 4: Extract frames ──
    print(f"\n[4] Extracting frames to: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in matched:
        # Skip if already extracted
        existing = OUTPUT_DIR / name
        if existing.exists() and any(existing.rglob("*.jpg")):
            print(f"\n  [SKIP] {name} already extracted ({existing})")
            continue
        extract_frames(videos[name], name, annotations[name], OUTPUT_DIR)

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"  Output directory : {OUTPUT_DIR}")
    print(f"  Videos processed : {len(matched)}")

    if annot_only:
        print(f"\n  Annotations WITHOUT video files:")
        for name in annot_only:
            print(f"    {name}  (from {annotations[name]['source_file']})")

    if video_only:
        print(f"\n  Video files WITHOUT annotations:")
        for name in video_only:
            print(f"    {name}")

    print()


if __name__ == "__main__":
    main()
