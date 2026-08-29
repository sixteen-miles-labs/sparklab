#!/usr/bin/env bash
#
# Publish built engine wheels to the SixteenMiles Labs rolling `beta` release.
#
# Usage:
#   scripts/publish-wheels.sh [dist-dir]       (default: ./dist)
#
# Deletes the release's previous wheels for each platform being published, THEN
# uploads the new ones -- in that order. Shipped Desktops resolve assets with a
# first-match scan over the release's asset list, so an old and a new wheel
# coexisting would keep serving the old one; a brief no-asset window (a clean
# install error and a retry) is the safer failure. Requires `gh` authenticated
# with write access to the target repo.
#
# Environment:
#   SPARKLAB_WEB_REPO   target repo  (default: sixteen-miles-labs/sparklab)
#   SPARKLAB_WEB_TAG    release tag  (default: beta)
set -euo pipefail

REPO="${SPARKLAB_WEB_REPO:-sixteen-miles-labs/sparklab}"
TAG="${SPARKLAB_WEB_TAG:-beta}"
DIST="${1:-dist}"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

command -v gh >/dev/null 2>&1 || die "gh not found"
[ -d "$DIST" ] || die "no such dist dir: $DIST"

shopt -s nullglob
wheels=("$DIST"/sparklab-*.whl "$DIST"/sparklab_kernel_cache-*.whl)
[ "${#wheels[@]}" -gt 0 ] || die "no sparklab wheels in $DIST"

# Platforms covered by this publish; pruning is per-platform, so a linux-only
# publish leaves the win_amd64 assets alone.
platforms="$(for w in "${wheels[@]}"; do
  case "${w##*/}" in
    *linux_x86_64*) echo linux_x86_64 ;;
    *win_amd64*) echo win_amd64 ;;
    *) echo UNKNOWN ;;
  esac
done | sort -u)"
grep -qx UNKNOWN <<<"$platforms" && die "cannot infer the platform of every wheel in $DIST"

# Publish gate: prune-then-upload replaces a platform's wheel pair WHOLESALE, so demand
# a complete, self-consistent pair up front -- exactly one runtime + one kernel-cache
# wheel per platform, with matching +g<sha> stamps when stamped. A half dist (a build
# that died between the two wheels) or a mixed-build pair must fail HERE, before any
# asset is deleted, not leave the release missing an asset it will never get back.
while IFS= read -r p; do
  rt_n=0; kc_n=0; rt_stamp=""; kc_stamp=""
  for w in "${wheels[@]}"; do
    b="${w##*/}"
    case "$b" in
      sparklab-*"$p"*.whl)
        rt_n=$((rt_n + 1))
        rt_stamp="$(grep -oE '\+g[0-9a-f]{7,}' <<<"$b" | head -1 || true)"
        ;;
      sparklab_kernel_cache-*"$p"*.whl)
        kc_n=$((kc_n + 1))
        kc_stamp="$(grep -oE '\.g[0-9a-f]{7,}' <<<"$b" | head -1 || true)"
        ;;
    esac
  done
  { [ "$rt_n" -eq 1 ] && [ "$kc_n" -eq 1 ]; } \
    || die "$DIST must hold exactly one runtime + one kernel-cache wheel for $p (found $rt_n + $kc_n)"
  if [ -n "$rt_stamp" ] && [ -n "$kc_stamp" ] && [ "${rt_stamp#+}" != "${kc_stamp#.}" ]; then
    die "stamp mismatch for $p: runtime has ${rt_stamp}, kernel-cache has ${kc_stamp} -- these wheels are from different builds"
  fi
done <<<"$platforms"

say "publishing to $REPO tag '$TAG':"
for w in "${wheels[@]}"; do say "  $(basename "$w")"; done

# Bootstrap the canonical rolling release when a repository has not published one yet.
# Re-runs and established compatibility repositories keep their existing release metadata.
if ! gh api "repos/$REPO/releases/tags/$TAG" >/dev/null 2>&1; then
  say "creating rolling prerelease $REPO tag '$TAG'"
  gh release create "$TAG" -R "$REPO" --prerelease \
    --title "SparkLab rolling beta" \
    --notes "Automated rolling wheels from SixteenMiles Labs. Assets may be replaced in place."
fi

# Prune the previous generation first (delete-then-upload; see header).
existing="$(gh api "repos/$REPO/releases/tags/$TAG" --jq '.assets[].name')"
while IFS= read -r p; do
  while IFS= read -r name; do
    case "$name" in
      sparklab-*"$p"*.whl | sparklab_kernel_cache-*"$p"*.whl)
        say "deleting old asset $name"
        gh release delete-asset "$TAG" "$name" -R "$REPO" --yes
        ;;
    esac
  done <<<"$existing"
done <<<"$platforms"

for w in "${wheels[@]}"; do
  say "uploading $(basename "$w")"
  gh release upload "$TAG" "$w" -R "$REPO"
done

say "release now carries:"
gh api "repos/$REPO/releases/tags/$TAG" \
  --jq '.assets[] | select(.name | endswith(".whl")) | "  \(.name)  \(.digest // "no-digest")  \(.updated_at)"'
