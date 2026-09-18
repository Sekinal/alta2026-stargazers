#!/usr/bin/env bash
# Download raw AU/UK place-review sources to /workspace/raw/reviews (run on the remote).
# Sources, licences and filters are documented in build_reviews.py.
set -uo pipefail
RAW=${1:-/workspace/raw/reviews}
mkdir -p "$RAW"/{googlelocal,booking,nhs,airbnb}
dl() {  # dl URL DEST: resumable, skip if complete
  [[ -s "$2" && ! -e "$2.part" ]] && { echo "have $2"; return; }
  touch "$2.part"
  curl -fL --retry 5 --retry-delay 5 -C - -A "alta2026-research-corpus/0.1" -o "$2" "$1" && rm -f "$2.part" \
    && echo "ok $2 $(du -h "$2" | cut -f1)" || echo "FAILED $1"
}
G=https://mcauleylab.ucsd.edu/public_datasets/data/googlelocal
dl $G/places.clean.json.gz "$RAW/googlelocal/places.clean.json.gz" &
dl $G/reviews.clean.json.gz "$RAW/googlelocal/reviews.clean.json.gz" &
B=https://huggingface.co/datasets/Booking-com/accommodation-reviews/resolve/main
dl $B/train.csv "$RAW/booking/train.csv" &
dl $B/val.csv "$RAW/booking/val.csv" &
dl https://huggingface.co/datasets/Booking-com/accommodation-reviews/resolve/main/README.md "$RAW/booking/README.md"
dl https://huggingface.co/datasets/janduplessis886/england-nhs-gp-reviews/resolve/main/england-nhs-gp-reviews.csv \
  "$RAW/nhs/england-nhs-gp-reviews.csv" &
A=https://data.insideairbnb.com
for c in united-kingdom/england/london/2026-06-19 united-kingdom/scotland/edinburgh/2026-06-23 \
         united-kingdom/england/greater-manchester/2026-06-28 united-kingdom/england/bristol/2026-06-29 \
         australia/nsw/sydney/2026-06-16 australia/vic/melbourne/2026-06-16; do
  dl $A/$c/data/reviews.csv.gz "$RAW/airbnb/$(echo "$c" | cut -d/ -f1,3 | tr / _).csv.gz" &
done
wait
ls -la "$RAW"/*
