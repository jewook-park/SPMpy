# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SPMpy Quickstart v0.1
#
# SPMpy is an open-source collection of Python tools for analyzing multi-dimensional scanning probe microscopy (SPM) data,
# including STM/S and AFM. It uses **`xarray`** as the primary data container to preserve both data and metadata.
#
# **Authors:** Dr. Jewook Park (CNMS, ORNL)  
# **Contact:** parkj1@ornl.gov
#
# ### Stages in this notebook
# - **Stage 0:** Environment check + repository bootstrap
# - **Stage 1:** Data loading (Nanonis `.sxm`, `.3ds`) into `xarray.Dataset`
# - **Stage 2 (planned):** Visualization and analysis utilities
#
# ### License note
# This repository is provided for internal and collaborative review. Licensing terms will be finalized according to ORNL/DOE policies.
#

# %% [markdown]
# ## Notebook Navigation
#
# - **Stage 0** — Environment check & bootstrap
#   - Step 1: Set `REPO_ROOT` and import `spmpy`
#   - Step 2: Run structured environment diagnostics
#   - Step 3: Decision & next action
# - **Stage 1** — Data loading (STM/SPM files)
#   - Stage 1.1: 2D image data (`.sxm`) → `xarray.Dataset`
#   - Stage 1.2 (planned): GridSpectroscopy (`.3ds`) → `xarray.Dataset`
# - **Stage 2 (planned)** — Visualization & analysis
#
# **Tip:** Run cells from top to bottom. Markdown cells describe what to do and what to expect.
#

# %% [markdown]
# ## Stage-0 Step 1 — Local repository bootstrap

# %% [markdown]
# ## Stage 0 — Step 1: Bootstrap the local repository
#
# Set `REPO_ROOT` to your local SPMpy clone folder, add it to `sys.path`, then import `spmpy`.
#

# %%
import sys
from pathlib import Path

# IMPORTANT: set this to your local SPMpy repository root
REPO_ROOT = Path(r"C:\\Users\\gkp\\Documents\\GitHub\\SPMpy")

if not REPO_ROOT.exists():
    raise RuntimeError(
        f"[SPMpy] Repo root does not exist: {REPO_ROOT}\n"
        "[Action] Edit REPO_ROOT to match your local clone location."
    )

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import spmpy
print('[SPMpy] Imported from:', spmpy.__file__)


# %% [markdown]
# ## Stage-0 Step 2 — Environment diagnostic (read-only)

# %%
# Safe import of env_check module (explicit module path)
try:
    import spmpy.utils.env_check_v2025Dec_30_revised as env_check
except ImportError as e:
    raise RuntimeError(
        '[SPMpy] Failed to import env_check module.\n'
        'Reason: module file not found or misnamed.\n'
        'Action: verify file name and restart kernel.'
    ) from e


# %%
from dataclasses import dataclass

@dataclass
class EnvStatus:
    ok: bool = False
    needs_restart: bool = False
    inconclusive: bool = False
    missing: list | None = None

def interpret_env_check(env):
    status = EnvStatus()

    if hasattr(env, 'ENV_OK'):
        status.ok = bool(env.ENV_OK)
        status.needs_restart = bool(getattr(env, 'INSTALLED_NOW', False))
        status.missing = getattr(env, 'MISSING_REQUIRED', None)
        return status

    status.inconclusive = True
    return status

status = interpret_env_check(env_check)


# %% [markdown]
# ## Stage 0 — Step 3: Decision & next action
#
# Based on the diagnostic result, follow the instruction printed by the next cell.
#

# %% [markdown]
# ## Stage 0 — Step 2: Run environment diagnostics
#
# This checks whether required packages are installed and whether a kernel restart is needed.
# The next cell will create a `status` object used by the decision step.
#

# %%
if status.ok and not status.needs_restart:
    print('[SPMpy] ✅ Environment ready.')
    print('[Next] Continue to Stage-1 below (Data Loading).')

elif status.ok and status.needs_restart:
    print('[SPMpy] ✅ Environment updated.')
    print('[Action] Restart the kernel, then re-run Stage-0 in this notebook.')

elif status.inconclusive:
    print('[SPMpy] ⚠ Environment status inconclusive.')
    print('[Action] Run the diagnostic notebook:')
    print('        notebooks/env_check_v_2025Dec_30_revised.ipynb')
    print('[Then] Return here, restart kernel if needed, and re-run Stage-0.')

else:
    print('[SPMpy] ❌ Environment not ready.')
    if status.missing:
        print('Missing packages:')
        for m in status.missing:
            print('  -', m)
    print('[Action] Fix the environment, restart kernel, then re-run Stage-0.')


# %% [markdown]
# # Stage 1 — File Loading (SXM, 3DS)
#
# Stage 1 loads **Nanonis files** and standardizes them into **`xarray.Dataset`** objects.
#
# - **Stage 1.1:** 2D image data (`.sxm`) → `xarray.Dataset`
# - **Stage 1.2 (planned):** grid data (`.3ds`) → `xarray.Dataset`
#
# **Important:** Stage 1 performs *loading only* (no plane fit, no flattening, no filtering).
# Processing functions will be organized separately under a data-processing module.
#

# %% [markdown]
# ## Stage 1.1 — 2D Image Data Loading (`.sxm`)
#
# ### What you will do in this section
# 1. Select a working folder (GUI folder picker).
# 2. List files in the folder as a DataFrame (for reproducible selection).
# 3. Choose an `.sxm` file name from the table.
# 4. Load the file into an **`xarray.Dataset`** using `img2xr`.
#
# ### Why this workflow
# This is intentionally designed to support future workflows where you load **multiple files** and build a dataset collection in a consistent way.
#

# %% [markdown]
# ### Imports for Stage 1
#
# In this Quickstart, the I/O logic is **not** defined inline.
# Instead, we import the legacy-compatible I/O functions from the package:
#
# - `select_folder()` — GUI folder picker
# - `files_in_folder()` — folder inventory → DataFrame (**no `os.chdir()`**)
# - `img2xr()` — `.sxm` → `xarray.Dataset`
# - `grid2xr()` — `.3ds` → `xarray.Dataset` (used in Stage 1.2)
#
# This keeps the Quickstart focused on workflow, while the implementation lives in `spmpy/io/`.
#

# %%
# I/O function set (paired .py lives in: spmpy/io/spmpy_io_library_v0_1.py)
from spmpy.io import spmpy_io_library_v0_1 as io

select_folder = io.select_folder
files_in_folder = io.files_in_folder
img2xr = io.img2xr
grid2xr = io.grid2xr


# %% [markdown]
# ### Step 1 — Select a working folder
#
# Run the next cell to pick a folder that contains your `.sxm` / `.3ds` files.
#

# %%
selected_folder = select_folder()
if selected_folder:
    print(f"Selected folder: {selected_folder}")
else:
    print("No folder selected.")


# %% [markdown]
# ### Step 2 — Inventory the folder as a DataFrame
#
# This creates a DataFrame inventory so you can reproducibly select files by name.
#
# **Note:** Because we do not use `os.chdir()`, the DataFrame includes a full `file_path` column.
# Use `file_path` when loading files, and define an explicit `output_dir` when saving results later.
#

# %%
folder_path = selected_folder
print(f"Selected folder: {folder_path}")

files_df = files_in_folder(folder_path)
files_df

# %% [markdown]
# ### Step 3 — Select an `.sxm` file from the inventory
#
# Pick a file name from the DataFrame. You can keep a list for future multi-file loading.
#

# %%
# List all SXM files
file_list = files_df[files_df.type=='sxm'].file_name
file_list

# %%
# Choose one file (edit as needed)
sxm_name = file_list.iloc[0] if len(file_list) else None
sxm_name

# %% [markdown]
# ### Step 4 — Load the SXM file into an `xarray.Dataset`
#
# No plotting is performed here. The returned `xarray.Dataset` is sufficient for validation.
#

# %%
from pathlib import Path

if sxm_name is None:
    raise RuntimeError('No .sxm files found in the selected folder.')

# Prefer explicit file_path if provided by files_in_folder()
if 'file_path' in files_df.columns:
    sxm_path = Path(files_df.loc[files_df.file_name == sxm_name, 'file_path'].iloc[0])
else:
    sxm_path = Path(folder_path) / sxm_name

print('[SPMpy] Loading:', sxm_path)

ds_sxm = img2xr(str(sxm_path), center_offset=False)
ds_sxm

# %% [markdown]
# ### Step 5 — Add experiment metadata (attrs)
#
# SPMpy keeps experiment context in `Dataset.attrs`. Edit the values below to match your experiment.
# These fields are user-defined and will be used later in analysis/plotting pipelines.
#

# %%
# Edit these values for your dataset
ds_sxm.attrs['tip'] = 'PtIr'
ds_sxm.attrs['sample'] = 'Cu(111)'
ds_sxm.attrs['ref_a0_nm'] = 0.255
ds_sxm.attrs['temperature'] = '4.35K'

# Example alternative (commented):
# ds_sxm.attrs['tip'] = 'Ni'
# ds_sxm.attrs['sample'] = 'FeTeSe'
# ds_sxm.attrs['ref_a0_nm'] = 0.384
# ds_sxm.attrs['temperature'] = '40mK'

ds_sxm

# %% [markdown]
# ## End of Stage 1.1
#
# At this point you have a **2D SXM image** loaded as an **`xarray.Dataset`**.
#
# Next (planned):
# - **Stage 1.2:** `.3ds` grid loading (`grid2xr`) into `xarray.Dataset`
# - **Stage 2:** visualization and data-processing steps (plane fit / flattening) from a dedicated module
#

# %% [markdown]
# ## Stage 1.2 — GridSpectroscopy (.3ds) loading
#
# This section loads a Nanonis GridSpectroscopy file (`.3ds`) and converts it into an `xarray.Dataset`.
#
# ### What you will do
# 1. Select a `.3ds` file name from the folder inventory (`files_df`).
# 2. Load it with `grid2xr()` using an explicit `file_path`.
# 3. Add experiment metadata to `ds_grid.attrs`.
#
# **Note:** This stage performs loading only. Processing (plane fit / flattening / filtering) belongs to a
# dedicated data-processing module (Stage 2).
#

# %%
# Select one or more .3ds files from the inventory
file_list_3ds = files_df[files_df.type == '3ds'].file_name
file_list_3ds

# %%
# Choose a single file for loading (edit as needed)
if len(file_list_3ds) == 0:
    raise RuntimeError('No .3ds files found in the selected folder.')

grid_name = file_list_3ds.iloc[0]
print('Selected .3ds file:', grid_name)

# %%
from pathlib import Path

# Prefer explicit file_path if provided by files_in_folder()
if 'file_path' in files_df.columns:
    grid_path = Path(files_df.loc[files_df.file_name == grid_name, 'file_path'].iloc[0])
else:
    grid_path = Path(folder_path) / grid_name

print('[SPMpy] Loading:', grid_path)

ds_grid = grid2xr(str(grid_path))
ds_grid

# %% [markdown]
# ### Step — Add experiment metadata (attrs)
#
# Edit the values below to match your experiment.
# These fields are intentionally user-defined and will be used later in analysis/plotting pipelines.
#

# %%
# Edit these values for your grid dataset
ds_grid.attrs['tip'] = 'PtIr'
ds_grid.attrs['sample'] = 'Cu(111)'
ds_grid.attrs['ref_a0_nm'] = 0.255
ds_grid.attrs['temperature'] = '4.35K'

# Example alternative (commented):
# ds_grid.attrs['tip'] = 'Ni'
# ds_grid.attrs['sample'] = 'FeTeSe'
# ds_grid.attrs['ref_a0_nm'] = 0.384
# ds_grid.attrs['temperature'] = '40mK'

ds_grid

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# ## End of Stage 1
#
# At this point you have:
#
# - `ds_sxm`: a 2D SXM image loaded as an `xarray.Dataset`
# - `ds_grid`: a GridSpectroscopy dataset loaded as an `xarray.Dataset`
#
# Next (planned):
#
# - **Stage 2:** Visualization and data-processing steps (plane fit / flattening) from a dedicated module.
#
