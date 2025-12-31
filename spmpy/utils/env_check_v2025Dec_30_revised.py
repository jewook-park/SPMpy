# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python (DataAnalysis)
#     language: python
#     name: dataanalysis
# ---

# # SPMpy Environment Check (Stage-1 internal)
#
# **Notebook:** `env_check_v.2025Dec_30_revised.ipynb`
#
# This notebook is the **standard environment check & auto-setup** entry for SPMpy users.
#
# ## Policy
# - Priority: **mamba → conda → pip** (conda-forge first)
# - If a package is missing, this notebook may attempt installation.
# - **No hard stop** is enforced; instead, a **short summary** is printed at the end.
# - If installs/updates happened, **restart the kernel** and re-run this notebook (or Quickstart).
#
# ## Outputs
# At the end, this notebook defines and prints:
# - `ENV_OK` (bool)
# - `MISSING_REQUIRED` (list)
# - `INSTALLED_NOW` (list)
# - `WARNINGS` (list)
#
# Quickstart will read these variables and show a concise message.

# +
# ==============================================================================
# SPMpy Environment Inspection & Auto-Setup (Single Jupyter Cell)
# Priority: mamba -> conda -> pip (conda-forge first)
# ==============================================================================
import importlib
import subprocess
import sys
import shutil
from warnings import warn
from typing import Optional, Tuple

# Summary variables (consumed by Quickstart)
ENV_OK = True
MISSING_REQUIRED = []
INSTALLED_NOW = []
WARNINGS = []

def _note_warn(msg: str) -> None:
    WARNINGS.append(msg)
    warn(msg)

def run_raw(cmd: str) -> bool:
    try:
        result = subprocess.run(
            cmd, shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Command failed] {cmd}")
        if e.stderr:
            print(f"[Error] {e.stderr.strip()}")
        return False

def run_py_module(mod_and_args: str) -> bool:
    return run_raw(f"{sys.executable} -m {mod_and_args}")

def pick_pkg_manager() -> Tuple[Optional[str], Optional[str]]:
    has_mamba = shutil.which('mamba') is not None
    has_conda = shutil.which('conda') is not None
    if has_mamba and has_conda:
        return ('mamba', 'conda')
    if has_mamba:
        return ('mamba', None)
    if has_conda:
        return ('conda', None)
    return (None, None)

PKG_MGR, PKG_MGR_FALLBACK = pick_pkg_manager()
print(f"[Info] Package manager priority: {PKG_MGR or 'None'}" + (f" -> {PKG_MGR_FALLBACK}" if PKG_MGR_FALLBACK else ''))

def get_jupyterlab_major() -> Optional[int]:
    try:
        out = subprocess.check_output(
            f"{sys.executable} -m jupyter lab --version", shell=True, text=True
        ).strip()
        return int(out.split('.')[0])
    except Exception:
        return None

JL_MAJOR = get_jupyterlab_major()
print(f"[Info] Detected JupyterLab major version: {JL_MAJOR}")

def is_importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def conda_install(spec: str) -> bool:
    if PKG_MGR:
        if run_raw(f"{PKG_MGR} install -y -c conda-forge {spec}"):
            return True
    if PKG_MGR_FALLBACK:
        if run_raw(f"{PKG_MGR_FALLBACK} install -y -c conda-forge {spec}"):
            return True
    return False

def pip_install(spec: str) -> bool:
    return run_py_module(f"pip install {spec}")

def smart_install(name: str,
                  import_name: Optional[str] = None,
                  prefer_conda: bool = True,
                  conda_spec: Optional[str] = None,
                  pip_spec: Optional[str] = None,
                  forbid_pip: bool = False,
                  required: bool = False) -> bool:
    global ENV_OK
    import_name = import_name or name
    conda_spec = conda_spec or name
    pip_spec = pip_spec or name

    if is_importable(import_name):
        print(f"[OK] {import_name} already importable.")
        return True

    print(f"[Installing] {name} ...")
    if prefer_conda:
        if conda_install(conda_spec) and is_importable(import_name):
            print(f"[Imported] {import_name} after conda install.")
            INSTALLED_NOW.append(name)
            return True
        if not forbid_pip and pip_install(pip_spec) and is_importable(import_name):
            print(f"[Imported] {import_name} after pip install.")
            INSTALLED_NOW.append(name)
            return True
    else:
        if not forbid_pip and pip_install(pip_spec) and is_importable(import_name):
            print(f"[Imported] {import_name} after pip install.")
            INSTALLED_NOW.append(name)
            return True
        if conda_install(conda_spec) and is_importable(import_name):
            print(f"[Imported] {import_name} after conda install.")
            INSTALLED_NOW.append(name)
            return True

    print(f"[Failed] Could not install {name}")
    if required:
        ENV_OK = False
        MISSING_REQUIRED.append(name)
    return False

print('[Step 1] Widgets ...')
smart_install('ipywidgets', import_name='ipywidgets', prefer_conda=True, pip_spec='ipywidgets>=8', required=True)
smart_install('jupyterlab_widgets', import_name='jupyterlab_widgets', prefer_conda=True, required=False)

if JL_MAJOR and JL_MAJOR < 4:
    print('[Build] JupyterLab <4 detected. Rebuilding Lab ...')
    run_raw('jupyter lab build --dev-build=False --minimize=False')
else:
    print('[Info] JupyterLab >=4 detected: no lab build required.')

print('[Step 2] Plotly ...')
smart_install('plotly', import_name='plotly', prefer_conda=True, required=False)

print('[Step 3] HoloViz stack ...')
for spec, imp in [('panel','panel'), ('holoviews','holoviews'), ('jupyter-bokeh','jupyter_bokeh')]:
    smart_install(spec, import_name=imp, prefer_conda=True, required=False)

print('[Step 4] Stage-1 minimum scientific stack ...')
smart_install('numpy', import_name='numpy', prefer_conda=True, required=True)
smart_install('xarray', import_name='xarray', prefer_conda=True, required=True)
smart_install('matplotlib', import_name='matplotlib', prefer_conda=True, required=True)

smart_install('scipy', import_name='scipy', prefer_conda=True, required=False)
smart_install('pandas', import_name='pandas', prefer_conda=True, required=False)
smart_install('scikit-image', import_name='skimage', conda_spec='scikit-image', pip_spec='scikit-image', prefer_conda=True, required=False)
smart_install('xrft', import_name='xrft', prefer_conda=True, required=False)
smart_install('hvplot', import_name='hvplot', prefer_conda=True, required=False)
smart_install('gwyfile', import_name='gwyfile', prefer_conda=True, required=False)
smart_install('netcdf4', import_name='netCDF4', conda_spec='netcdf4', pip_spec='netCDF4', prefer_conda=True, required=False)
smart_install('h5netcdf', import_name='h5netcdf', prefer_conda=True, required=False)
smart_install('python-pptx', import_name='pptx', conda_spec='python-pptx', pip_spec='python-pptx', prefer_conda=True, required=False)
smart_install('pyqt', import_name='PyQt5', conda_spec='pyqt', forbid_pip=True, prefer_conda=True, required=False)

print('[Step 5] Best-effort visualization backend config ...')
try:
    from bokeh.io import output_notebook
    import holoviews as hv
    output_notebook()
    hv.extension('bokeh')
    print('[Config] Bokeh and HoloViews activated.')
except Exception as e:
    _note_warn(f'Bokeh/HoloViews configuration skipped: {e}')

print('\n[Done] Base environment setup cell finished.')

# -

# ## Notes: Revised `nanonispy` (Jewook fork)
#
# The original `nanonispy` has not been actively updated for years. For SPMpy usage,
# use the **Jewook fork** (updated `np.float` → `float`, etc.).
#
# **Example install (local clone):**
# ```bash
# pip install /path/to/your/local/nanonispy
# ```
#

# ## Notes: `seaborn-image` (install separately)
#
# - Install `seaborn-image` **separately** (by design)
# - Jewook fork / higher version is recommended if you experienced matplotlib-related errors
# - After changing visualization packages, **restart the kernel**
#

# +
# Optional checks: PyTorch & scikit-learn
from importlib import import_module

def _check(pkg, import_name=None):
    import_name = import_name or pkg
    try:
        m = import_module(import_name)
        v = getattr(m, '__version__', 'installed')
        print(f"[OK] {pkg}: {v}")
        return True
    except Exception as e:
        print(f"[Missing] {pkg}: {e}")
        return False

_check('torch')
_check('scikit-learn', import_name='sklearn')


# +
# ===== Short Summary (for Quickstart) =====
print('\n================ SPMpy ENV SUMMARY ================')
if len(MISSING_REQUIRED) == 0:
    ENV_OK = True
    print('[SPMpy] ENV OK ✅  (Required deps satisfied)')
else:
    ENV_OK = False
    print('[SPMpy] ENV NOT OK ❌  Missing required packages:')
    for m in MISSING_REQUIRED:
        print('  -', m)

if INSTALLED_NOW:
    print('\n[SPMpy] Installed/updated in this run:')
    for p in INSTALLED_NOW:
        print('  -', p)

if WARNINGS:
    print('\n[SPMpy] Warnings (non-fatal):')
    for w in WARNINGS:
        print('  -', w)

if not ENV_OK:
    print('\n[Action] Please address missing required packages, restart the kernel, then re-run env_check or Quickstart.')
elif INSTALLED_NOW:
    print('\n[Action] Packages were installed/updated. Please restart the kernel, then re-run Quickstart.')
else:
    print('\n[Action] You may proceed to Quickstart.')
print('===================================================\n')
