#!/usr/bin/env bash
# Download the three SocioPatterns contact networks (CSV) from the Netzschleuder mirror
# and unzip them into data/raw/<name>/{edges.csv,nodes.csv,gprops.csv}.
#
# Original data and required citations: https://sociopatterns.org/datasets.html
# Mirror: https://networks.skewed.de/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"
mkdir -p "$RAW"
cd "$RAW"

declare -A FILES=(
  [sp_hospital]="sp_hospital/files/sp_hospital.csv.zip"
  [sp_primary_school]="sp_primary_school/files/sp_primary_school.csv.zip"
  [sp_high_school_proximity]="sp_high_school/files/proximity.csv.zip"
)

for name in "${!FILES[@]}"; do
  url="https://networks.skewed.de/net/${FILES[$name]}"
  echo "-> $name"
  curl -sL -o "$name.csv.zip" "$url"
  mkdir -p "$name"
  unzip -oq "$name.csv.zip" -d "$name"
  wc -l "$name/edges.csv" "$name/nodes.csv"
done
echo "done"
