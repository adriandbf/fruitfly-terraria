#!/usr/bin/env bash
# Run on the Mac. Bundles what the PC needs that isn't in git: the arena
# world, the character, the saved arena mark / Nurse house for that world,
# and the FlyWire annotation table. Writes fruitfly_pc_bundle.zip in the repo root.
#   bash scripts/pack_for_pc.sh                      # world "The Violent Heart"
#   bash scripts/pack_for_pc.sh "My World Name"
set -euo pipefail
cd "$(dirname "$0")/.."
WORLD="${1:-The Violent Heart}"
FILE_STEM="${WORLD// /_}"
TML=""
for d in "$HOME/Library/Application Support/Terraria/tModLoader" \
         "$HOME/Library/Application Support/Terraria/tModLoader-preview"; do
  [ -d "$d/Worlds" ] && TML="$d" && break
done
[ -n "$TML" ] || { echo "tModLoader save folder not found under ~/Library/Application Support/Terraria"; exit 1; }

OUT="$(pwd)/fruitfly_pc_bundle.zip"
STAGE="$(mktemp -d)"
mkdir -p "$STAGE/Worlds" "$STAGE/Players" "$STAGE/repo/logs/moonlord" "$STAGE/repo/brain/data"

found=0
for ext in wld twld; do
  if [ -f "$TML/Worlds/$FILE_STEM.$ext" ]; then cp "$TML/Worlds/$FILE_STEM.$ext" "$STAGE/Worlds/"; found=1; fi
done
[ "$found" = 1 ] || { echo "world '$WORLD' not found in $TML/Worlds (files are named like $FILE_STEM.wld)"; ls "$TML/Worlds"; exit 1; }
cp "$TML"/Players/*.plr "$STAGE/Players/" 2>/dev/null || true
cp "$TML"/Players/*.tplr "$STAGE/Players/" 2>/dev/null || true

MARK_STEM="$(echo "$WORLD" | sed -E 's/[^A-Za-z0-9_-]+/_/g; s/^_+|_+$//g')"
for stem in arena_mark nurse_house; do
  f="logs/moonlord/${stem}_${MARK_STEM}.json"
  if [ -f "$f" ]; then cp "$f" "$STAGE/repo/logs/moonlord/"; else echo "warning: no $f (the PC will need --mark-arena / --build-nurse-house)"; fi
done
cp brain/data/flywire_annotations.tsv.gz "$STAGE/repo/brain/data/"

rm -f "$OUT"
(cd "$STAGE" && zip -qr "$OUT" .)
rm -rf "$STAGE"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1)): world '$WORLD', $(ls "$TML"/Players/*.plr 2>/dev/null | wc -l | tr -d ' ') character(s), arena + Nurse house marks, annotations"
