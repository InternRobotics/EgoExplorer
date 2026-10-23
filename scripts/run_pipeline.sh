#!/usr/bin/env bash
# Single-machine EgoExplorer pipeline (production configuration).
#
#   bash scripts/run_pipeline.sh <stage> [batch ...]
#
# Stages, in order:
#   check    tracker L/R, starting stance, glove pairing and handedness reports
#   fix      write L/R-corrected copies (originals untouched)
#   view     merged input view: corrected copy where one exists, otherwise raw
#   project  reference -> proposal -> L1/L2/L3 projection (one retry at half lr)
#   fuse     glove -> MANO -> fused into the body
#   export   unified 30 Hz npz per session (+ video/touch-only export for failures in unified_30hz_nobody/)
#   drift    tracker drift report (needs projection residuals)
#   review   three-pane review pages (needs ffmpeg for the 360p video copies)
#   gallery  overview page with every QC flag
#   all      every stage above
#   status   counts per stage
#
# Layout under $WORK (see the guidelines, chapter 7.3):
#   ego_data_sync/<batch>/<session>/   raw recordings
#   profiles/<batch>.json              subject profile for that batch (one subject, one donning)
# Everything else is written by this script.  Batches default to every folder in ego_data_sync/.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
: "${WORK:?set WORK to the work folder}"
PY="${PY:-python}"
CKPT="${CKPT:-$ROOT/ckpt/avatarposer_five/model.pt}"
BODY="${BODY:-$ROOT/models/smplx}"
DEVICE="${DEVICE:-$("$PY" -c 'import torch;print("cuda" if torch.cuda.is_available() else "cpu")' 2>/dev/null || echo cpu)}"
RAW="$WORK/ego_data_sync"; FIXED="$WORK/ego_data_lr_fixed"; INPUT="$WORK/input/ego_data_sync"
OUT="$WORK/out"; G="$OUT/gallery"; R="$OUT/review"
export PYTHONPATH="$ROOT/src:$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT" || exit 1

stage="${1:-help}"; shift || true
if [ $# -gt 0 ]; then BATCHES=("$@"); else
  BATCHES=(); for d in "$RAW"/*/; do [ -d "$d" ] && BATCHES+=("$(basename "$d")"); done
fi
say() { echo "[$(date '+%F %T')] $*"; }
profile_of() {
  local p="$WORK/profiles/$1.json"
  [ -f "$p" ] || { echo "missing subject profile $p (calibrate this batch first, guidelines chapter 5)" >&2; return 1; }
  echo "$p"
}
merge_lists() {  # <output> <inputs...>: concatenate JSON list reports
  "$PY" -c 'import json,sys; rows=[r for p in sys.argv[2:] for r in json.load(open(p,encoding="utf-8"))]; json.dump(rows,open(sys.argv[1],"w",encoding="utf-8"),ensure_ascii=False)' "$@"
}

stage_check() {
  mkdir -p "$G"; local lr=() gs=()
  for b in "${BATCHES[@]}"; do
    local p; p="$(profile_of "$b")" || return 1
    say "check $b"
    "$PY" scripts/check_tracker_lr.py     --sync-root "$RAW" --datasets "$b" --profile "$p" --output "$G/tracker_lr.$b.json" || return 1
    "$PY" scripts/check_startup_stance.py --sync-root "$RAW" --datasets "$b" --profile "$p" --output "$G/startup_stance.$b.json" || return 1
    "$PY" scripts/check_glove_sync.py     --sync-root "$RAW" --datasets "$b" --output "$G/glove_sync.$b.json" || return 1
    "$PY" scripts/check_hand_identity.py  --sync-root "$RAW" --datasets "$b" --output "$G/hand_identity.$b.json" || return 1
    lr+=("$G/tracker_lr.$b.json"); gs+=("$G/glove_sync.$b.json")
  done
  merge_lists "$G/tracker_lr.json" "${lr[@]}" && merge_lists "$G/glove_sync.json" "${gs[@]}"
}

stage_fix() {
  [ -f "$G/tracker_lr.json" ] && [ -f "$G/glove_sync.json" ] || { echo "run the check stage first" >&2; return 1; }
  "$PY" scripts/fix_tracker_lr.py --report "$G/tracker_lr.json" --sync-root "$RAW" --output-root "$FIXED" || return 1
  "$PY" scripts/fix_glove_lr.py   --report "$G/glove_sync.json" --tracker-report "$G/tracker_lr.json" \
    --sync-root "$RAW" --output-root "$FIXED"
}

stage_view() {
  # One link per file: Path.rglob does not descend into symlinked directories.
  local n=0
  for b in "${BATCHES[@]}"; do
    for d in "$RAW/$b"/*/; do
      [ -d "$d" ] || continue
      local s; s="$(basename "$d")"; local src="$FIXED/$b/$s"; [ -d "$src" ] || src="${d%/}"
      mkdir -p "$INPUT/$b/$s"
      for f in "$src"/*; do ln -sfn "$(cd "$(dirname "$f")" && pwd -P)/$(basename "$f")" "$INPUT/$b/$s/$(basename "$f")"; done
      n=$((n + 1))
    done
  done
  say "view: $n sessions under $INPUT"
}

stage_project() {
  for b in "${BATCHES[@]}"; do
    local p; p="$(profile_of "$b")" || return 1
    say "project $b (device=$DEVICE)"
    "$PY" scripts/run_five_tracker_pipeline.py --input-root "$INPUT/$b" --subject-profile "$p" --checkpoint "$CKPT" \
      --output-root "$OUT" --body-model-path "$BODY" --device "$DEVICE" --workers 1 --skip-existing
    # A diverged projection gets one retry at half the learning rate.
    local failed; failed=$("$PY" -c 'import json,sys; print(" ".join(x["session"] for x in json.load(open(sys.argv[1]))["sessions"] if x["status"]=="failed"))' "$OUT/pipeline_report.json" 2>/dev/null)
    for s in $failed; do
      say "retry $s at lr 0.002"
      "$PY" scripts/run_five_tracker_pipeline.py --input-root "$INPUT/$b" --subject-profile "$p" --checkpoint "$CKPT" \
        --output-root "$OUT" --body-model-path "$BODY" --device "$DEVICE" --workers 1 --skip-existing \
        --learning-rate 0.002 --session "$s" || true
    done
  done
}

stage_fuse() {
  for b in "${BATCHES[@]}"; do
    local p; p="$(profile_of "$b")" || return 1
    for d in "$INPUT/$b"/*/; do
      local s; s="$(basename "$d")"
      [ -f "$d/$s.npz" ] || continue                                   # no glove stream
      [ -f "$OUT/projected/$s/smplx_profile_driven.npz" ] || continue  # no body
      [ -f "$OUT/projected/$s/smplx_with_manus_hands.npz" ] && continue
      say "fuse $s"
      "$PY" -m five_tracker_smplx.cli fuse-manus-batch --input-root "${d%/}" --body-output-dir "$OUT/projected" \
        --subject-profile "$p" --body-model-path "$BODY" --device "$DEVICE" \
        --max-hand-frames 480 --iterations 60 --no-protocol --no-viewer || say "fuse $s FAILED"
    done
  done
}

stage_export() {
  "$PY" scripts/export_unified_30hz.py --projected-root "$OUT/projected" --output-root "$OUT/unified_30hz" --rate 30 --skip-existing || return 1
  # Failed projections get a video/touch-only export in its own folder: this script deletes any export
  # of a projected session from its output root, so it must never share unified_30hz/.
  "$PY" scripts/export_unified_nobody.py --sync-root "$INPUT" --batch-root "$OUT" --output-root "$OUT/unified_30hz_nobody" --skip-existing
}

stage_drift() {
  mkdir -p "$G"
  "$PY" scripts/check_tracker_drift.py --sync-root "$INPUT" --projected "$OUT/projected" --output "$G/tracker_drift.json"
}

stage_review() {
  mkdir -p "$R"; local ff; ff="$(command -v ffmpeg || true)"
  [ -n "$ff" ] || say "ffmpeg not found: review pages are built without video"
  for npz in "$OUT/unified_30hz"/*.npz "$OUT/unified_30hz_nobody"/*.npz; do
    [ -f "$npz" ] || continue
    local name slug mp4; name="$(basename "$npz" .npz)"
    # ASCII slug: Chrome refuses media whose file name carries the Chinese task name.
    slug="$(printf '%s' "$name" | sed -E 's/^(session_[0-9]+_[0-9]+).*/\1/')"
    [ -f "$R/$slug.review.html" ] && continue
    mp4="$(ls "$RAW"/*/"$name"/"$name".mp4 2>/dev/null | head -1)"
    if [ -n "$ff" ] && [ -n "$mp4" ] && [ ! -f "$R/$slug.360.mp4" ]; then
      "$ff" -y -v error -i "$mp4" -vf "scale=-2:360" -r 15 -c:v libx264 -preset veryfast -crf 32 -an \
        -movflags +faststart "$R/$slug.360.mp4" || rm -f "$R/$slug.360.mp4"
    fi
    "$PY" scripts/build_review_payload.py --unified "$npz" --output-dir "$R" --body-model-path "$BODY" --max-frames 300 || { say "review $name FAILED"; continue; }
    if [ "$name" != "$slug" ]; then
      mv -f "$R/$name.payload.json" "$R/$slug.payload.json"
      [ -f "$R/$name.mesh.i8c.gz" ] && mv -f "$R/$name.mesh.i8c.gz" "$R/$slug.mesh.i8c.gz"
    fi
    local v=(); [ -f "$R/$slug.360.mp4" ] && v=(--video)
    "$PY" scripts/make_review_html.py "$R/$slug.payload.json" --slug "$slug" "${v[@]}" || say "review $name FAILED"
  done
}

stage_gallery() {
  "$PY" scripts/build_ego_gallery.py --batch-root "$OUT" --output-dir "$G" --sync-root "$INPUT" \
    --body-model-path "$BODY" --device cpu --title "EgoExplorer review"
}

stage_status() {
  "$PY" - "$RAW" "$OUT" <<'PY'
import sys
from pathlib import Path
raw, out = Path(sys.argv[1]), Path(sys.argv[2])
count = lambda pattern, base=out: len(list(base.glob(pattern))) if base.is_dir() else 0
print(f"sessions        {count('*/*/*.tracker.npz', raw)}")
print(f"projected       {count('projected/*/smplx_profile_driven.npz')}")
print(f"fused hands     {count('projected/*/smplx_with_manus_hands.npz')}")
print(f"unified 30 Hz   {count('unified_30hz/*.npz')}")
print(f"review pages    {count('review/*.review.html')}")
print(f"gallery         {'yes' if (out / 'gallery' / 'index.html').exists() else 'no'}")
PY
}

case "$stage" in
  check) stage_check;; fix) stage_fix;; view) stage_view;; project) stage_project;; fuse) stage_fuse;;
  export) stage_export;; drift) stage_drift;; review) stage_review;; gallery) stage_gallery;; status) stage_status;;
  all) stage_check && stage_fix && stage_view && stage_project && stage_fuse && stage_export && stage_drift \
         && stage_review && stage_gallery && stage_status;;
  *) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//';;
esac
