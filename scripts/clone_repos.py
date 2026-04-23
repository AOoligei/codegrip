"""Clone all GREPO repos needed for file summary extraction."""
import os
import subprocess
import json

REPO_DIR = "/home/chenlibin/grepo_agent/data/repos"

# repo_name -> github org/repo
REPOS = {
    "Cirq": "quantumlib/Cirq",
    "Flexget": "Flexget/Flexget",
    "PyBaMM": "pybamm-team/PyBaMM",
    "PyPSA": "PyPSA/PyPSA",
    "Radicale": "Kozea/Radicale",
    "Solaar": "pwr-Solaar/Solaar",
    "WeasyPrint": "Kozea/WeasyPrint",
    "aiogram": "aiogram/aiogram",
    "arviz": "arviz-devs/arviz",
    "astroid": "pylint-dev/astroid",
    "astropy": "astropy/astropy",
    "attrs": "python-attrs/attrs",
    "babel": "python-babel/babel",
    "beets": "beetbox/beets",
    "briefcase": "beeware/briefcase",
    "cfn-lint": "aws-cloudformation/cfn-lint",
    "crawlee-python": "apify/crawlee-python",
    "csvkit": "wireservice/csvkit",
    "datasets": "huggingface/datasets",
    "dspy": "stanfordnlp/dspy",
    "dvc": "iterative/dvc",
    "dynaconf": "dynaconf/dynaconf",
    "falcon": "falconry/falcon",
    "faststream": "airtai/faststream",
    "feature_engine": "feature-engine/feature_engine",
    "filesystem_spec": "fsspec/filesystem_spec",
    "flask": "pallets/flask",
    "fonttools": "fonttools/fonttools",
    "geopandas": "geopandas/geopandas",
    "haystack": "deepset-ai/haystack",
    "icloud_photos_downloader": "icloud-photos-downloader/icloud_photos_downloader",
    "instructlab": "instructlab/instructlab",
    "ipython": "ipython/ipython",
    "jax": "jax-ml/jax",
    "jupyter-ai": "jupyterlab/jupyter-ai",
    "kedro": "kedro-org/kedro",
    "litellm": "BerriAI/litellm",
    "llama-stack": "meta-llama/llama-stack",
    "llama_deploy": "run-llama/llama_deploy",
    "marshmallow": "marshmallow-code/marshmallow",
    "mesa": "projectmesa/mesa",
    "networkx": "networkx/networkx",
    "openai-agents-python": "openai/openai-agents-python",
    "patroni": "patroni/patroni",
    "pipenv": "pypa/pipenv",
    "poetry": "python-poetry/poetry",
    "privacyidea": "privacyidea/privacyidea",
    "pvlib-python": "pvlib/pvlib-python",
    "pydicom": "pydicom/pydicom",
    "pylint": "pylint-dev/pylint",
    "pyomo": "Pyomo/pyomo",
    "python": "python/cpython",
    "python-control": "python-control/python-control",
    "python-telegram-bot": "python-telegram-bot/python-telegram-bot",
    "pyvista": "pyvista/pyvista",
    "qtile": "qtile/qtile",
    "scipy": "scipy/scipy",
    "scrapy-splash": "scrapy-plugins/scrapy-splash",
    "segmentation_models.pytorch": "qubvel/segmentation_models.pytorch",
    "shapely": "shapely/shapely",
    "smolagents": "huggingface/smolagents",
    "sphinx": "sphinx-doc/sphinx",
    "sqlfluff": "sqlfluff/sqlfluff",
    "streamlink": "streamlink/streamlink",
    "tablib": "jazzband/tablib",
    "torchtune": "pytorch/torchtune",
    "transitions": "pytransitions/transitions",
    "twine": "pypa/twine",
    "urllib3": "urllib3/urllib3",
    "wemake-python-styleguide": "wemake-services/wemake-python-styleguide",
    "xarray": "pydata/xarray",
}

os.makedirs(REPO_DIR, exist_ok=True)
cloned = 0
skipped = 0
failed = 0

for name in sorted(REPOS.keys()):
    dest = os.path.join(REPO_DIR, name)
    if os.path.isdir(dest):
        skipped += 1
        continue
    org_repo = REPOS[name]
    url = f"https://github.com/{org_repo}.git"
    print(f"[clone] {name} <- {url}", flush=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, dest],
            capture_output=True, timeout=300,
        )
        if os.path.isdir(dest):
            print(f"  OK", flush=True)
            cloned += 1
        else:
            print(f"  FAIL (no dir)", flush=True)
            failed += 1
    except Exception as e:
        print(f"  FAIL: {e}", flush=True)
        failed += 1

print(f"\nDone: {len(REPOS)} total, {skipped} existed, {cloned} cloned, {failed} failed")
