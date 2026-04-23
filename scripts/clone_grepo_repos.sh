#!/usr/bin/env bash
# clone_grepo_repos.sh — Clone all 71 GREPO repos (shallow, parallel).
# Usage: bash scripts/clone_grepo_repos.sh
#
# Clones into data/repos/ with --depth 1.
# Skips repos that already exist.
# Uses xargs -P 4 for parallel cloning.

set -euo pipefail

REPO_DIR="/home/chenlibin/grepo_agent/data/repos"
mkdir -p "${REPO_DIR}"

# ── repo_name -> github org/repo mapping (71 repos) ──────────────────────────
declare -A REPO_MAP=(
    [Cirq]="quantumlib/Cirq"
    [Flexget]="Flexget/Flexget"
    [PyBaMM]="pybamm-team/PyBaMM"
    [PyPSA]="PyPSA/PyPSA"
    [Radicale]="Kozea/Radicale"
    [Solaar]="pwr-Solaar/Solaar"
    [WeasyPrint]="Kozea/WeasyPrint"
    [aiogram]="aiogram/aiogram"
    [arviz]="arviz-devs/arviz"
    [astroid]="pylint-dev/astroid"
    [astropy]="astropy/astropy"
    [attrs]="python-attrs/attrs"
    [babel]="python-babel/babel"
    [beets]="beetbox/beets"
    [briefcase]="beeware/briefcase"
    [cfn-lint]="aws-cloudformation/cfn-lint"
    [crawlee-python]="apify/crawlee-python"
    [csvkit]="wireservice/csvkit"
    [datasets]="huggingface/datasets"
    [dspy]="stanfordnlp/dspy"
    [dvc]="iterative/dvc"
    [dynaconf]="dynaconf/dynaconf"
    [falcon]="falconry/falcon"
    [faststream]="airtai/faststream"
    [feature_engine]="feature-engine/feature_engine"
    [filesystem_spec]="fsspec/filesystem_spec"
    [flask]="pallets/flask"
    [fonttools]="fonttools/fonttools"
    [geopandas]="geopandas/geopandas"
    [haystack]="deepset-ai/haystack"
    [icloud_photos_downloader]="icloud-photos-downloader/icloud_photos_downloader"
    [instructlab]="instructlab/instructlab"
    [ipython]="ipython/ipython"
    [jax]="jax-ml/jax"
    [jupyter-ai]="jupyterlab/jupyter-ai"
    [kedro]="kedro-org/kedro"
    [litellm]="BerriAI/litellm"
    [llama-stack]="meta-llama/llama-stack"
    [llama_deploy]="run-llama/llama_deploy"
    [marshmallow]="marshmallow-code/marshmallow"
    [mesa]="projectmesa/mesa"
    [networkx]="networkx/networkx"
    [openai-agents-python]="openai/openai-agents-python"
    [patroni]="patroni/patroni"
    [pipenv]="pypa/pipenv"
    [poetry]="python-poetry/poetry"
    [privacyidea]="privacyidea/privacyidea"
    [pvlib-python]="pvlib/pvlib-python"
    [pydicom]="pydicom/pydicom"
    [pylint]="pylint-dev/pylint"
    [pyomo]="Pyomo/pyomo"
    [python]="python/cpython"
    [python-control]="python-control/python-control"
    [python-telegram-bot]="python-telegram-bot/python-telegram-bot"
    [pyvista]="pyvista/pyvista"
    [qtile]="qtile/qtile"
    [scipy]="scipy/scipy"
    [scrapy-splash]="scrapy-plugins/scrapy-splash"
    [segmentation_models.pytorch]="qubvel/segmentation_models.pytorch"
    [shapely]="shapely/shapely"
    [smolagents]="huggingface/smolagents"
    [sphinx]="sphinx-doc/sphinx"
    [sqlfluff]="sqlfluff/sqlfluff"
    [streamlink]="streamlink/streamlink"
    [tablib]="jazzband/tablib"
    [torchtune]="pytorch/torchtune"
    [transitions]="pytransitions/transitions"
    [twine]="pypa/twine"
    [urllib3]="urllib3/urllib3"
    [wemake-python-styleguide]="wemake-services/wemake-python-styleguide"
    [xarray]="pydata/xarray"
)

# ── Build list of repos to clone (skip existing) ─────────────────────────────
TO_CLONE=()
SKIPPED=0
for repo_name in "${!REPO_MAP[@]}"; do
    if [[ -d "${REPO_DIR}/${repo_name}" ]]; then
        echo "[skip] ${repo_name} already exists"
        SKIPPED=$((SKIPPED + 1))
    else
        TO_CLONE+=("${repo_name}")
    fi
done

TOTAL=${#REPO_MAP[@]}
CLONE_COUNT=${#TO_CLONE[@]}
echo ""
echo "GREPO repos: ${TOTAL} total, ${SKIPPED} already cloned, ${CLONE_COUNT} to clone."
echo ""

if [[ ${CLONE_COUNT} -eq 0 ]]; then
    echo "Nothing to clone. All repos already present."
    exit 0
fi

# ── Clone in parallel (4 workers) ────────────────────────────────────────────
clone_repo() {
    local repo_name="$1"
    local org_repo="${REPO_MAP[${repo_name}]}"
    local url="https://github.com/${org_repo}.git"
    local dest="${REPO_DIR}/${repo_name}"

    echo "[clone] ${repo_name} <- ${url}"
    if git clone --depth 1 "${url}" "${dest}" 2>&1; then
        echo "[done]  ${repo_name}"
    else
        echo "[FAIL]  ${repo_name} — clone failed for ${url}" >&2
    fi
}
export -f clone_repo
export REPO_DIR
# Export the associative array by serializing it
for repo_name in "${!REPO_MAP[@]}"; do
    export "REPO_MAP_${repo_name}=${REPO_MAP[${repo_name}]}"
done

# We need a wrapper since exported associative arrays are tricky in bash.
# Instead, pass repo_name and org/repo as a single delimited argument.
CLONE_ARGS=()
for repo_name in "${TO_CLONE[@]}"; do
    CLONE_ARGS+=("${repo_name}|${REPO_MAP[${repo_name}]}")
done

clone_one() {
    local entry="$1"
    local repo_name="${entry%%|*}"
    local org_repo="${entry##*|}"
    local url="https://github.com/${org_repo}.git"
    local dest="${REPO_DIR}/${repo_name}"

    echo "[clone] ${repo_name} <- ${url}"
    if git clone --depth 1 "${url}" "${dest}" 2>&1; then
        echo "[done]  ${repo_name}"
    else
        echo "[FAIL]  ${repo_name} — clone failed for ${url}" >&2
    fi
}
export -f clone_one
export REPO_DIR

printf '%s\n' "${CLONE_ARGS[@]}" | xargs -I {} -P 4 bash -c 'clone_one "$@"' _ {}

echo ""
echo "Cloning complete."
