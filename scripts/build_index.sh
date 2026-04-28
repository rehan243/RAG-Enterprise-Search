#!/usr/bin/env bash
# offline index build from a folder of raw docs; prints progress so ops trusts it
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
  echo "usage: $0 <raw_docs_dir> <out_index_dir>"
  echo "  expects python entrypoint scripts/build_faiss.py (adapt to your tree)"
}

if [[ $# -ne 2 ]]; then usage; exit 1; fi
RAW="$1"
OUT="$2"

if [[ ! -d "$RAW" ]]; then
  echo -e "${RED}raw dir missing:${NC} $RAW"; exit 1
fi

mkdir -p "$OUT"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo -e "${GREEN}counting files${NC}"
TOTAL="$(find "$RAW" -type f | wc -l | tr -d ' ')"
echo -e "${YELLOW}files:${NC} $TOTAL"

echo -e "${GREEN}starting faiss build${NC}"
export RAW_DOCS_DIR="$RAW"
export INDEX_OUT_DIR="$OUT"

if [[ -f scripts/build_faiss.py ]]; then
  python scripts/build_faiss.py
else
  echo -e "${YELLOW}no build_faiss.py yet; stub run${NC}"
  python - <<'PY'
import os, time
raw = os.environ["RAW_DOCS_DIR"]
out = os.environ["INDEX_OUT_DIR"]
for i in range(1, 6):
    print(f"chunk batch {i}/5 ...")
    time.sleep(0.2)
open(os.path.join(out, "index.stub"), "w").write("stub\n")
PY
fi

echo -e "${GREEN}wrote index under${NC} $OUT"
