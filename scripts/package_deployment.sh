#!/usr/bin/env bash
# Build and export on the developer machine; does not contact the server.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_DIR="$PROJECT_ROOT/release/stylist"
for item in face_landmarker.task face_shape face_parsing seasonal; do
    if [[ ! -e "$PROJECT_ROOT/models/$item" ]]; then
        echo "Missing model: models/$item" >&2
        exit 1
    fi
done
mkdir -p "$RELEASE_DIR"
cd "$PROJECT_ROOT"
docker buildx build --platform linux/amd64 --load -t ai-stylist:amd64 .
# Validate runtime imports and strict checkpoint loading in the Linux image.
docker run --rm --platform linux/amd64 \
    --mount "type=bind,source=$PROJECT_ROOT/models,target=/app/models,readonly" \
    ai-stylist:amd64 python -c \
    'import mediapipe, torchvision; from stylist.seasonal import load_seasonal_model; model, revision = load_seasonal_model(); print("Strict checkpoint load OK:", revision)'
docker save ai-stylist:amd64 | gzip > "$RELEASE_DIR/ai-stylist-amd64.tar.gz"
tar --exclude='*/.cache' --exclude='*/.DS_Store' \
    -czf "$RELEASE_DIR/models.tar.gz" -C "$PROJECT_ROOT" \
    models/face_landmarker.task models/face_shape models/face_parsing models/seasonal
cp deploy/compose.yaml deploy/nginx-stylist.conf deploy/README.md "$RELEASE_DIR/"
echo "Deployment bundle: $RELEASE_DIR"
du -sh "$RELEASE_DIR"
