#!/usr/bin/env bash
# Commit, tag and push a release. Reads the version from version.py so the tag
# and the running app can never disagree.
#
#   ./release.sh "what changed in one line"
#
# Refuses to push if the tests fail, because a tagged commit that does not pass
# is worse than no tag at all.

set -euo pipefail
cd "$(dirname "$0")"

MESSAGE="${1:-}"
if [ -z "$MESSAGE" ]; then
  echo "usage: ./release.sh \"what changed in one line\"" >&2
  exit 1
fi

VERSION=$(python3 -c 'import version; print(version.__version__)')
echo "==> version $VERSION"

echo "==> tests"
python3 tests/test_designer.py

if [ -z "$(git status --porcelain)" ]; then
  echo "==> nothing to commit"
else
  git add -A
  git commit -m "v$VERSION: $MESSAGE"
fi

if git rev-parse "v$VERSION" >/dev/null 2>&1; then
  echo "==> tag v$VERSION already exists, leaving it alone"
else
  git tag -a "v$VERSION" -m "$MESSAGE"
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push -u origin "$BRANCH"
git push origin "v$VERSION"
echo "==> pushed v$VERSION to $BRANCH"
