#!/bin/bash
# Clone eval9 repos (shallow to save disk)
# Estimated total: ~2-3GB

REPO_DIR="/home/chenlibin/grepo_agent/data/repos"
mkdir -p "$REPO_DIR"

declare -A REPO_URLS
REPO_URLS[astropy]="https://github.com/astropy/astropy.git"
REPO_URLS[dvc]="https://github.com/iterative/dvc.git"
REPO_URLS[ipython]="https://github.com/ipython/ipython.git"
REPO_URLS[pylint]="https://github.com/pylint-dev/pylint.git"
REPO_URLS[scipy]="https://github.com/scipy/scipy.git"
REPO_URLS[sphinx]="https://github.com/sphinx-doc/sphinx.git"
REPO_URLS[streamlink]="https://github.com/streamlink/streamlink.git"
REPO_URLS[xarray]="https://github.com/pydata/xarray.git"
REPO_URLS[geopandas]="https://github.com/geopandas/geopandas.git"

for repo in astropy dvc ipython pylint scipy sphinx streamlink xarray geopandas; do
    url="${REPO_URLS[$repo]}"
    dest="$REPO_DIR/$repo"
    if [ -d "$dest" ]; then
        echo "SKIP: $repo (already exists)"
        continue
    fi
    echo "Cloning $repo from $url..."
    git clone --depth 1 "$url" "$dest" 2>&1 | tail -1
    echo "  Done: $(du -sh $dest | cut -f1)"
done

echo ""
echo "Total repos disk usage:"
du -sh "$REPO_DIR"
