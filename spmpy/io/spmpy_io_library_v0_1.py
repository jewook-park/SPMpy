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
# # SPMpy I/O Library v0.1 (Notebook-paired)
#
# This notebook contains the **I/O function set** intended to live under `spmpy/io/`.
# It is designed to be **paired with a `.py` file via jupytext**.
#
# ## Goals
# - Preserve the legacy workflow interface where it makes sense.
# - Avoid hidden global side effects (especially `os.chdir()`).
# - Keep I/O responsibilities limited to: **read + standardize to `xarray.Dataset`**.
#
# ## Included functions
# - `select_folder()` — GUI folder picker (PyQt5)
# - `files_in_folder()` — inventory a folder into a `pandas.DataFrame` (**no `chdir`**)
# - `img2xr()` — load `.sxm` into `xarray.Dataset` (NetCDF-safe attrs)
#

# %% [markdown]
# ## Why we avoid `os.chdir()`
#
# `os.chdir()` changes the **process-wide current working directory**. In a notebook workflow, this can silently
# affect unrelated cells and libraries that use relative paths.
#
# ### The design used here
# - We keep your **working folder** as an explicit variable, e.g. `folder_path`.
# - We store **full paths** for each file in the inventory DataFrame (`file_path`).
#
# ### Implication for saving results
# Yes, this means that **saving should also use explicit paths**. For example:
#
# - If you want outputs to go next to the raw data: use `output_dir = folder_path`.
# - If you want a clean separation: use `output_dir = Path(folder_path) / 'processed'`.
#
# In other words, you choose the target folder once (explicitly), then every save uses that folder.
# This is more reproducible than relying on whatever the current working directory happens to be.
#

# %% [markdown]
# ## Imports
#
# These are standard dependencies for the I/O layer.
# (`nanonispy` is required only when `img2xr()` is called.)
#

# %%
from __future__ import annotations

from pathlib import Path
import os
import glob
import json
import math
import re

import numpy as np
import pandas as pd
import xarray as xr

# Optional GUI dependency (only needed when select_folder() is used)
try:
    from PyQt5.QtWidgets import QApplication, QFileDialog
except Exception:
    QApplication = None
    QFileDialog = None


# %% [markdown]
# ## `select_folder()`
#
# Folder picker used in Quickstart Stage-1.
#

# %%
def select_folder() -> str:
    """Open a folder selection dialog and return the selected folder path.

    Returns
    -------
    str
        Selected folder path. Empty string if no folder was selected.
    """
    if QApplication is None or QFileDialog is None:
        raise ModuleNotFoundError(
            "PyQt5 is required for select_folder(). Install PyQt5 or use a non-GUI path workflow."
        )

    app = QApplication.instance()
    if app is None:
        import sys
        app = QApplication(sys.argv)

    file_dialog = QFileDialog()
    folder_path = file_dialog.getExistingDirectory(None, "Select Folder")
    return str(folder_path) if folder_path else ""



# %% [markdown]
# ## `files_in_folder()` (no `chdir`)
#
# This function inventories a folder and returns a DataFrame with the **same columns** as your legacy workflow:
#
# - `group`, `num`, `file_name`, `type`
#
# Additionally, it includes two columns that make multi-folder workflows safer:
#
# - `folder_path` — the folder that was scanned
# - `file_path` — full path to each file
#
# Because `file_path` is explicit, later stages can load and save deterministically without relying on `os.chdir()`.
#

# %%
def files_in_folder(path_input: str, print_all: bool = False) -> pd.DataFrame:
    """Generate a DataFrame listing files in the specified folder.

    Parameters
    ----------
    path_input : str
        Folder path.
    print_all : bool, optional
        If True, prints the full DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: ['group', 'num', 'file_name', 'type', 'folder_path', 'file_path'].
    """
    folder = Path(path_input)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    # Keep this informational print (legacy-friendly) WITHOUT changing CWD.
    print("Current Path =", os.getcwd())
    print("Target Folder =", str(folder))

    # Inventory by extension (no chdir)
    sxm_files  = sorted([p.name for p in folder.glob('*.sxm')])
    grid_files = sorted([p.name for p in folder.glob('*.3ds')])
    csv_files  = sorted([p.name for p in folder.glob('*.csv')])
    gwy_files  = sorted([p.name for p in folder.glob('*.gwy')])
    xlsx_files = sorted([p.name for p in folder.glob('*.xlsx')])
    nc_files   = sorted([p.name for p in folder.glob('*.nc')])

    def _df_for(files, ext_len, num_slice=True):
        rows = []
        for fn in files:
            if num_slice:
                group = fn[:-7]
                num = fn[-7:-4]
            else:
                group = fn[:-ext_len]
                num = np.nan
            rows.append([group, num, fn])
        return pd.DataFrame(rows, columns=['group', 'num', 'file_name'])

    file_list_sxm_df  = _df_for(sxm_files,  4, num_slice=True)
    file_list_3ds_df  = _df_for(grid_files, 4, num_slice=True)
    file_list_csv_df  = _df_for(csv_files,  4, num_slice=True)
    file_list_gwy_df  = _df_for(gwy_files,  4, num_slice=False)
    file_list_xlsx_df = _df_for(xlsx_files, 5, num_slice=False)
    file_list_nc_df   = _df_for(nc_files,   3, num_slice=False)

    file_list_df = pd.concat(
        [file_list_sxm_df, file_list_3ds_df, file_list_csv_df,
         file_list_gwy_df, file_list_xlsx_df, file_list_nc_df],
        ignore_index=True
    )

    file_list_df['type'] = [fn[-3:] for fn in file_list_df.file_name]
    file_list_df.loc[file_list_df.type == 'lsx', 'type'] = 'xlsx'
    file_list_df.loc[file_list_df.type == '.nc', 'type'] = 'nc'

    # Add explicit paths
    file_list_df['folder_path'] = str(folder)
    file_list_df['file_path'] = [str(folder / fn) for fn in file_list_df.file_name]

    if print_all:
        print(file_list_df)

    # Legacy-style summary prints
    sxm_file_groups = list(set(file_list_sxm_df['group']))
    for group in sxm_file_groups:
        print('sxm file groups:', group, ': # of files =',
              len(file_list_sxm_df[file_list_sxm_df['group'] == group]))

    if len(file_list_df[file_list_df['type'] == '3ds']) == 0:
        print('No GridSpectroscopy data')
    else:
        print('# of GridSpectroscopy',
              list(set(file_list_df[file_list_df['type'] == '3ds'].group))[0],
              '=', file_list_df[file_list_df['type'] == '3ds'].group.count())

    return file_list_df



# %% [markdown]
# ## `grid2xr()` (3DS → `xarray.Dataset`)
#
# This function loads Nanonis GridSpectroscopy `.3ds` files and standardizes them into an `xarray.Dataset`.
#
# Design rules:
# - I/O only: reading + metadata/coords standardization.
# - No plane fit / flattening / filtering here.
#
# If `nanonispy` (or other required dependencies) are missing, the function should raise a clear error.
#

# %%
#griddata_file = file_list_df[file_list_df.type=='3ds'].iloc[0].file_name

def grid2xr(griddata_file, center_offset = True): 
    """
    An xarray DataSet representing grid data from a Nanonis 3ds file.

    This DataSet contains multiple variables corresponding to different data channels, such as "I_fwd" (Forward Current), "I_bwd" (Backward Current), "LIX_fwd" (Lock-In X Forward), "LIX_bwd" (Lock-In X Backward), and "topography" (Topography). The data is organized along three dimensions: "Y" (Y-coordinate), "X" (X-coordinate), and "bias_mV" (Bias Voltage in mV).

    Attributes:
        - title (str): A title or description of the grid data.
        - image_size (list): A list containing the size of the image in X and Y dimensions.
        - X_spacing (float): The spacing between X-coordinates in nanometers.
        - Y_spacing (float): The spacing between Y-coordinates in nanometers.

    Additional Information:
    - The "bias_mV" dimension represents the bias voltage values in mV, and it includes values that are adjusted to have a "zero" bias point.
    - Depending on the `center_offset` parameter used during conversion, the X and Y coordinates may be adjusted to represent positions in the real scanner field of view or with (0,0) as the origin of the image.

    Example Usage:

    Convert a Nanonis 3ds file to a grid_xr DataSet
    grid_xr = grid2xr("example.3ds")

    Access data variables
    topography_data = grid_xr["topography"]
    forward_current_data = grid_xr["I_fwd"]

    Access attributes
    title = grid_xr.attrs["title"]
    image_size = grid_xr.attrs["image_size"]
    x_spacing = grid_xr.attrs["X_spacing"]
    y_spacing = grid_xr.attrs["Y_spacing"]


    Note: This DataSet is suitable for further analysis, visualization, and manipulation using the xarray library in Python.


    ---
    Summary 
    
    Here's a breakdown of the main steps in the grid2xr function:
    Read the Nanonis 3ds file using NanonisFile and extract relevant information such as grid dimensions, position, size, step sizes, channels (e.g., topography, current), and bias values.
    Check the topography data and reshape it if necessary. This step is for handling cases where the topography data is not in the expected shape.
    Process and interpolate bias values to ensure they include "zero" bias and have an odd number of points. This step is necessary to account for different bias settings in the data.
    Interpolate the current and lock-in data (both forward and backward) to match the new bias values.
    Create an xarray DataSet named grid_xr with the following variables: "I_fwd," "I_bwd," "LIX_fwd," "LIX_bwd," and "topography." These variables are associated with dimensions "Y," "X," and "bias_mV."
    Assign various attributes to the grid_xr DataSet, including the title, image size, spacing, and frequency information.
    Optionally, adjust the scan center position in real scanner field-of-view based on the center_offset parameter.
    Check and handle cases where the XY dimensions are not equal and may require interpolation.
    Return the grid_xr DataSet as the result of the function.
    This function seems to be designed for specific data formats and processing tasks related to Nanonis data. You can call this function with a Nanonis 3ds file as input to convert it into an xarray DataSet with the described attributes and dimensions.
    
    ---
    
    """

    # Import necessary libraries
    import re
    import numpy as np
    import xarray as xr
    import scipy.interpolate as sp
    import nanonispy as nap

    file = griddata_file
    #####################
    # conver the given 3ds file
    # to  xarray DataSet (check the attributes)
    NF = nap.read.NanonisFile(file)
    Gr = nap.read.Grid(NF.fname)#
    channel_name = Gr.signals.keys()  
    #print (channel_name)
    N = len(file);
    f_name = file[0:N-4]
    print (f_name) # Gr.basename
    # Extract data from the Nanonis file
    #####################################
    #  Header part
    #  Gr.header
    #####################################
    [dim_px,dim_py] = Gr.header['dim_px'] 
    [cntr_x, cntr_y] = Gr.header['pos_xy']
    [size_x,size_y] = Gr.header['size_xy']
    [step_dx,step_dy] = [ size_x/dim_px, size_y/dim_py] 
    #pixel_size =  size / pixel 
    '''
    ####   nX, nY --> x,y real scale  np array 
    #nX = np.array([step_dx*(i+1/2) for i in range (0,dim_px)])
    #nY = np.array([step_dy*(i+1/2) for i in range (0,dim_py)])

    # x = cntr_x - size_x + nX
    # y = cntr_y - size_y + nY
    # real XY position in nm scale, Center position & scan_szie + XY position
    '''

    ### Correct X and Y values extraction
    x = np.linspace(cntr_x - size_x / 2, cntr_x + size_x / 2, dim_px)
    y = np.linspace(cntr_y - size_y / 2, cntr_y + size_y / 2, dim_py)

    
    #####################################
    # signal part
    # Gr.signals
    #####################################
    topography = Gr.signals['topo']
    params_v = Gr.signals['params'] 
    # params_v.shape = (dim_px,dim_py,15) 
    # 15: 3ds infos. 
    bias = Gr.signals['sweep_signal']
    # check the shape (# of 'original' bias points)

    ##########################################
    # * if there is no bwd --> (bwd <= fwd)
    # * fwd bwd data average 
    ##########################################

    
    I_fwd = Gr.signals['Current (A)'] # 3d set (dim_px,dim_py,bias)
    #I_bwd = Gr.signals['Current [bwd] (A)'] # I bwd
    try:
        I_bwd = Gr.signals['Current [bwd] (A)']
    except KeyError: # if bwd channel was not saved.
        I_bwd = I_fwd
        print ("there is no [bwd] channel")
    # sometimes, LI channel names are inconsistent depends on program ver. 
    # find 'LI Demod 1 X (A)'  or  'LI X 1 omega (A)'

    #print( [s  for s in Gr.signals.keys()  if "LI"  in s  if "X" in s ])
    # 'LI' & 'X' in  channel name (signal.keys) 
    LIX_keys = [s  for s in Gr.signals.keys()  if "LI"  in s  if "X" in s ]
    # 0 is fwd, 1 is bwd 
    '''LIX_fwd, LIX_bwd = Gr.signals[LIX_keys[0]] ,Gr.signals[LIX_keys[1] ]'''
    if len(LIX_keys) == 2:
        LIX_fwd, LIX_bwd = Gr.signals[LIX_keys[0]], Gr.signals[LIX_keys[1]]
    elif len(LIX_keys) == 1:
        # If LIX_keys list length is 1
        LIX_fwd = Gr.signals[LIX_keys[0]]
        LIX_bwd = Gr.signals[LIX_keys[0]]
    else: 
        print ("Define LIX again")
    # same for LIY
    #print( [s  for s in Gr.signals.keys()  if "LI"  in s  if "Y" in s ])
    # 'LI' & 'Y' in  channel name (signal.keys) 
    LIY_keys = [s  for s in Gr.signals.keys()  if "LI"  in s  if "Y" in s ]
    # 0 is fwd, 1 is bwd 
    '''LIY_fwd, LIY_bwd = Gr.signals[LIY_keys[0]] ,Gr.signals[LIY_keys[1] ]'''
    if len(LIY_keys) == 2:
        LIY_fwd, LIY_bwd = Gr.signals[LIY_keys[0]], Gr.signals[LIY_keys[1]]
    elif len(LIY_keys) == 1:
        # If LIX_keys list length is 1
        LIY_fwd = Gr.signals[LIY_keys[0]]
        LIY_bwd = Gr.signals[LIY_keys[0]]
    else: 
        print ("Define LIY again")
    ###########################################################
    #plt.imshow(topography) # toppography check
    #plt.imshow(I_fwd[:,:,0]) # LIX  check
    ###########################################################

    ##########################################################
    #		 Title for Grid data 
    #       grid size, pixel, bias condition, and so on.
    #############################################################
    # Gr.header.get('Bias>Bias (V)') # bias condition 
    # Gr.header.get('Z-Controller>Setpoint') # current set  condition
    # Gr.header.get('dim_px')  # jpixel dimension 
    title = Gr.basename +' \n ('  + str(
        float(Gr.header.get('Bias Spectroscopy>Sweep Start (V)'))
    ) +' V ~ ' +str( 
        float(Gr.header.get('Bias Spectroscopy>Sweep End (V)'))
    )+ ' V) \n at Bias = '+ Gr.header.get(
        'Bias>Bias (V)'
    )[0:-3]+' mV, I_t =  ' + Gr.header.get(
        'Z-Controller>Setpoint'
    )[0:-4]+ ' pA, '+str(
        Gr.header.get('dim_px')[0]
    )+' x '+str(
        Gr.header.get('dim_px')[1]
    )+' points'
    #############################################################       
    ###########################
    # Bias segment check      #
    ###########################
    if 'Segment Start (V), Segment End (V), Settling (s), Integration (s), Steps (xn), Lockin, Init. Settling (s)' in Gr.header.keys():
            segment_info = Gr.header['Segment Start (V), Segment End (V), Settling (s), Integration (s), Steps (xn), Lockin, Init. Settling (s)']
            segments = [list(map(float, seg.split(',')[:5])) for seg in segment_info]
            print ('bias sweep is using Segment')
            
            bias_segment = []
            for start, end, _, _, steps in segments:
                bias_segment.extend(np.linspace(start, end, int(steps)))
            
            #print ('bias_segment: ', bias_segment)
            # 중복 제거 (세그먼트 연결 부분)
            def remove_duplicates(arr):
                seen = set()
                result = []
                for item in arr:
                    if item not in seen:
                        result.append(item)
                        seen.add(item)
                return result
        
            bias_segment = remove_duplicates(bias_segment)
            # use the 'remove_duplicates' function instead of np.unique 
            # in order not to mix the bias segment ascending /descending order
            # print ('after remove duplicate bias_segment: ', bias_segment)
            if len(bias_segment) != I_fwd.shape[2]:
                print("Error: check Segment")
                return None
            
            # 가장 작은 간격을 절대값 기준으로 계산
            min_step = np.min(np.abs(np.diff(bias_segment)))
            print('min_step:', min_step)
            
            # bias_segment의 시작과 끝을 비교하여 bias_new 생성
            if bias_segment[0] > bias_segment[-1]:
                # 시작점이 양수인 경우 (양수에서 음수로 가는 경우)
                bias_new = np.arange(bias_segment[0], bias_segment[-1] - min_step, -min_step)
            else:
                # 시작점이 음수인 경우 (음수에서 양수로 가는 경우)
                bias_new = np.arange(bias_segment[0], bias_segment[-1] + min_step, min_step)
            
            #print('bias_new:', bias_new)
            
            # bias_new의 방향을 설정
            if bias_new[0] > bias_new[-1]:
                bias_segment_direction_change = False
                print('bias_segment_direction_change = False')
            else:
                bias_segment_direction_change = True
                print('bias_segment_direction_change = True')
                print('Flipped bias_new to start from positive and end at negative')
            '''
            # 가장 작은 간격을 기준으로 bias_new 생성
            min_step = np.min( (np.diff(bias_segment)))
            print ('min_step',min_step)
            bias_new = np.arange(bias_segment[0], bias_segment[-1] + min_step, min_step)
            print ('bias_new: ', bias_new)
            # bias_new의 방향을 양수에서 음수로 설정
            if bias_new[0] > bias_new[-1]:
                # bias_new가 양수에서 음수 방향이면 그대로 유지
                bias_segment_direction_change = False
                print ('bias_segment_direction_change = False')
                pass
            else:
                # bias_new가 음수에서 양수 방향이면 반전시킴
                bias_new = np.flip(bias_new)
                bias_segment_direction_change = True
                print ('bias_segment_direction_change = True')
                print('Flipped bias_new to start from positive and end at negative')
            '''
            # bias_new가 0을 포함하도록 조정
            if len(bias_new) % 2 == 0:
                bias_new = np.linspace(bias_new[0], bias_new[-1], len(bias_new) + 1)
            
            nearest_zero_bias = np.argmin(np.abs(bias_new))
            bias_new -= bias_new[nearest_zero_bias]
            print('bias_new<-- segment')
            # 아래쪽 기능을 활용학위해서 nanonisV5 이상인 경우 
            # 위쪽 Segment 를 적용했으면 아래쪽  segment 는작동안함 
            Segment_V5 = True 


    else:
        bias = Gr.signals['sweep_signal']
        Segment_V5 =  False 
    
    Segment = Gr.header['Bias>Bias (V)']
    # bias unit : '(V)' 

    if (type(Segment) == str ) & (Segment_V5 == False): # single segment case
        print ('No Segments\n'+ 'Grid data acquired at bias = '+  str(float(Segment)) + 'V')    
    ## No Segments # +  bias setting 

    ########################
    # bias interpolation to have a "zero" bias 
    # interpolate bias_mV that include "zero" bias 
    # in 3D data : center x,y bias interpolation 
    # e.g  256--> including end points + zero  = 256+1 ( the center is "0")
        if len(bias)%2==0:
            bias_new = np.linspace(bias[0],bias[-1],num=(len(bias)+1)) 
            # if bias length is even_number 
            # including "0", total size is "len+1" 
        else:# if bias length is odd_number 
            bias_new = np.linspace(bias[0],bias[-1],num=(len(bias))) 
            # bias_new make a odd number of length
            # make only one value is closest to the zero. 
            
        nearest_zero_bias = np.where(abs(bias_new) == np.amin(abs(bias_new))) 
        # find the index of closest to "0" bias 
        bias_new = bias_new - bias_new[nearest_zero_bias] 
        # assign closest zero vavlue as a zero. 
        #bias_new[np.where(bias_new == np.amin(abs(bias_new)))]=0

    ##############################################
    #'Segment Start (V), Segment End (V), Settling (s), Integration (s), Steps (xn)'
    elif (len(Segment) == 3) & (Segment_V5 == False):
        print('Number of Segments =' + str(len(Segment))) 
        Segments = np.array([[ float(Segments) 
                              for Segments in Seg.split(',') ] 
                             for Seg in Segment], dtype = np.float64)
        # in the Segment, split strings sith "," 
        #  make a array after change it as float. 
        # check Nanonispy version
        # bias value could be not correct. 
        
        Seg1 = np.linspace(Segments[0,0],Segments[0,1],int(Segments[0,-1]))
        Seg2 = np.linspace(Segments[1,0],Segments[1,1],int(Segments[1,-1]))
        Seg3 = np.linspace(Segments[2,0],Segments[2,1],int(Segments[2,-1]))
        # except boundary end points,  combine segments ([1:]), Seg1, Seg2[1:], Seg3[1:] 
        bias_Seg = np.append(np.append(Seg1,Seg2[1:]),Seg3[1:]) 
        # Seg1 +  Seg2[1:] +  Se3[1:] 
        # make a clever & shoter way 'later...'
        print ('bias_Seg size = ' + str(len(bias_Seg)))
        bias_Nsteps=int(int(Segments[1,-1])/
                        (Seg2[-1]-Seg2[0])*(bias_Seg[-1]-bias_Seg[0]))
        # New bias Steps uses smallest step as a new stpe size. 
        bias_Nsteps_size = (Seg2[-1]-Seg2[0])/(Segments[1,-1])
        # (Segments[1,0]-Segments[1,1])/int(Segments[1,-1]) # bias step size    
        Neg_bias=-1*np.arange(
            0,bias_Nsteps_size*bias_Nsteps/2, bias_Nsteps_size)
        Pos_bias=np.flip(
            np.arange(0,bias_Nsteps_size*bias_Nsteps/2,bias_Nsteps_size))
        bias_new = np.flip( np.append(Pos_bias,Neg_bias[1:])) 
        # after segments, 
        # bias is called as  bias_new
        ##################################
        # now make the bias_new as an odd number. 
        ###################################
        if len(bias_new)%2==0:
            bias_new = np.linspace(bias_new[0],bias_new[-1],num=(len(bias_new)+1)) 
        else:
            bias_new = np.linspace(bias_new[0],bias_new[-1],num=(len(bias_new))) 
        # check  bias_new contians "zero" 
        nearest_zero_bias = np.where(abs(bias_new) == np.amin(abs(bias_new))) 
        # check index of the nearest value to zero "0"
        bias_new = bias_new - bias_new[nearest_zero_bias] 
        # adjust bias range for bias_new has "zero" 
        print ('bias_new size = ' + str(len(bias_new)))
        # bias 
    # make a new list for Bias
    else:
        if Segment_V5 == False : 
            print ("Segment error /n grid2xr is only support 3 segment case at this moment /n code a 5 Sements case")
        else: print ('Segment_V5 == True') 
    #
    ######################################################################
    # make a new bias length (including Segments) as a odd number, including zero
    ######################################################################


    ######################################################################
    # interpolation using bias_new 
    # I_fwd, I_bwd, LIX_fwd, LIX_bwd
    # => I_fwd_interpolate
    #######################################################################
    # assign a function using interpolation 
    # the same as original bias values 
    # make empty np array  & interpolate using scipy
    # xy dim is not changed here, 
    # only 3rd axis changed as new bias 
    ###########################
    # Interpolate current and lock-in data to match the new bias values
    def sweep_interpolation(np3Ddata, bias, bias_new):
        np3Ddata_interpolate = np.empty(
                    (np3Ddata.shape[0],np3Ddata.shape[1],bias_new.shape[0])) 

        for x_i,np3Ddata_xi in enumerate(np3Ddata):
            for y_j,np3Ddata_xi_yj in enumerate(np3Ddata_xi):
                #print (np3Ddata_xi_yj.shape)
                Interpolation1D_i_f = sp.interpolate.interp1d(
                    bias,
                    np3Ddata_xi_yj,
                    fill_value = "extrapolate",
                    kind = 'cubic')
                np3Ddata_interpolate[x_i,y_j,:] = Interpolation1D_i_f(bias_new)
        return np3Ddata_interpolate
        # bias_mV 생성
    bias_mV = bias_new * 1000

    # 데이터 보간
    I_fwd_interpolate = sweep_interpolation(I_fwd, bias_segment if 
                                            'Segment Start (V), Segment End (V), Settling (s), Integration (s), Steps (xn), Lockin, Init. Settling (s)' 
                                            in Gr.header.keys() else bias, bias_new)
    I_bwd_interpolate = sweep_interpolation(I_bwd, bias_segment if 
                                            'Segment Start (V), Segment End (V), Settling (s), Integration (s), Steps (xn), Lockin, Init. Settling (s)' 
                                            in Gr.header.keys() else bias, bias_new)
    LIX_fwd_interpolate = sweep_interpolation(LIX_fwd, bias_segment if 
                                              'Segment Start (V), Segment End (V), Settling (s), Integration (s), Steps (xn), Lockin, Init. Settling (s)'
                                              in Gr.header.keys() else bias, bias_new)
    LIX_bwd_interpolate = sweep_interpolation(LIX_bwd, bias_segment if  
                                              'Segment Start (V), Segment End (V), Settling (s), Integration (s), Steps (xn), Lockin, Init. Settling (s)'
                                              in Gr.header.keys() else bias, bias_new)
    '''
    I_fwd_interpolate = sweep_interpolation (I_fwd, bias, bias_new)
    I_bwd_interpolate = sweep_interpolation (I_bwd, bias, bias_new)
    LIX_fwd_interpolate = sweep_interpolation (LIX_fwd, bias, bias_new)
    LIX_bwd_interpolate = sweep_interpolation (LIX_bwd, bias, bias_new)
    '''
    ####################################################
    # to prevent error for bias direction 
    # 
    ##
    #  assign the bias direction 
    ## up or down ==> up anyway. 
    ###################################################
    if Segment_V5 == True: 
        print ('check_Segment_V5')
        if bias_segment_direction_change == False: 
            
            # if starting point is larger than end point. 
            # start from pos & end to neg
            # no changes. 
            print ('start from POS bias')
            I_fwd = I_fwd_interpolate
            I_bwd = I_bwd_interpolate
            LIX_fwd = LIX_fwd_interpolate
            LIX_bwd = LIX_bwd_interpolate
            bias_mV = bias_new*1000
        else:  # if end point is larger than start point. 
            # start from neg & end to pos
            # change to negative 
            
            print ('bias_new[0]>bias_new[-1]: False') 
            print ('start from NEG bias')
            
            I_fwd = I_fwd_interpolate
            I_bwd = I_bwd_interpolate
            LIX_fwd = LIX_fwd_interpolate
            LIX_bwd = LIX_bwd_interpolate
            ## Neg 에서 시작하는 grid 의 경우를 맞춰기위해서 flip없앰. 
            #I_fwd = np.flip(I_fwd_interpolate,2)
            #I_bwd = np.flip(I_bwd_interpolate,2)
            #LIX_fwd = np.flip(LIX_fwd_interpolate,2)
            #LIX_bwd = np.flip(LIX_bwd_interpolate,2)
            #bias_new_flip = np.flip(bias_new)
            bias_mV = bias_new*1000
            print ('After Flip => now all start from POS bias')
        ####################################################

    else:      
        if bias[0]>bias[-1]: 
            # if starting point is larger than end point. 
            # start from pos & end to neg
            # no changes. 
            print ('start from POS bias')
            I_fwd = I_fwd_interpolate
            I_bwd = I_bwd_interpolate
            LIX_fwd = LIX_fwd_interpolate
            LIX_bwd = LIX_bwd_interpolate
            bias_mV = bias_new*1000
        else:  # if end point is larger than start point. 
            # start from neg & end to pos
            # change to negative 
            print ('start from NEG bias')
            I_fwd = np.flip(I_fwd_interpolate,2)
            I_bwd = np.flip(I_bwd_interpolate,2)
            LIX_fwd = np.flip(LIX_fwd_interpolate,2)
            LIX_bwd = np.flip(LIX_bwd_interpolate,2)
            bias_new_flip = np.flip(bias_new)
            bias_mV = bias_new_flip*1000
            print ('Flip => start from POS bias')
            ####################################################
    
    ###################################################
    # convert data XR DataSet
    ####################################################
    

    # col = x 
    # row = y
    # I_fwd grid data ==> [Y, X, bias]
    print(I_fwd.shape)
    grid_xr = xr.Dataset(
        {
            "I_fwd" : (["Y","X","bias_mV"], I_fwd),
            "I_bwd" : (["Y","X","bias_mV"], I_bwd),
            "LIX_fwd" : (["Y","X","bias_mV"], LIX_fwd),
            "LIX_bwd" : (["Y","X","bias_mV"], LIX_bwd),
            "topography" : (["Y","X"], topography)
        },
        coords = {
            "X": (["X"], x),
            "Y": (["Y"], y),
            "bias_mV": (["bias_mV"], bias_mV)
        }
    )
    grid_xr.attrs["title"] = title
    #grid_xr.attrs['image_size'] = 
    #grid_xr.attrs['samlpe'] = 
    
    grid_xr.attrs['image_size']= [size_x,size_y]
    grid_xr.attrs['X_spacing']= step_dx
    grid_xr.attrs['Y_spacing']= step_dy    
    #grid_xr.attrs['freq_X_spacing']= 1/step_dx
    #grid_xr.attrs['freq_Y_spacing']= 1/step_dy
    # use the complex128 = True for xrft, 
    # then xrdata_fft.freq_X.spacing 
    # use the attrs in axis info 
    # in case of real X Y ( center & size of XY)
    if center_offset == True:
        # move the scan center postion in real scanner field of view
        grid_xr.assign_coords( X = (grid_xr.X + cntr_x -  size_x/2))
        grid_xr.assign_coords( Y = (grid_xr.Y + cntr_y -  size_y/2))
    else :
        pass
        # (0,0) is the origin of image 
    

    ############################
    # check the XY ratio 
    ############################
    #    if  size_x == size_y : 
    if  dim_px == dim_py : 

        pass
    else : 
        print ('dim_px != dim_py')
    # if xy size is not same, report it! 

    if step_dx != step_dy :
        xystep_ratio = step_dy/step_dx # check the XY pixel_ratio
        X_interp = np.linspace(grid_xr.X[0], grid_xr.X[-1], grid_xr.X.shape[0]*1)
        step_dx = step_dx # step_dx check 

        Y_interp = np.linspace(grid_xr.Y[0], grid_xr.Y[-1], int(grid_xr.Y.shape[0]*xystep_ratio)) 
        step_dy = step_dy/ xystep_ratio # step_dy check 

        # interpolation ratio should be int
        grid_xr= grid_xr.interp(X = X_interp, Y = Y_interp, method="linear")
        print('step_dx/step_dy = ', xystep_ratio)
        print ('grid_xr ==> reshaped')
    else: 
        grid_xr =grid_xr
        print('step_dx == step_dy')
    #print('z_LIX_fNb_xr', 'step_dx, step_dy = ',  z_LIX_fNb_xr.dims)
    print('grid_xr', 'step_dx, step_dy = ', 
          re.findall(r'\{([^}]+)', str(grid_xr.dims)))
    # regex practice
    
    #################################
    # assign attributes 
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if 'Wtip' in title:
        grid_xr.attrs['tip'] = 'W'
    elif 'PtIr' in title:
        grid_xr.attrs['tip'] = 'PtIr'
    elif '_Ni' in title:
        grid_xr.attrs['tip'] = 'Ni'
    elif 'Co_coated' in title:
        grid_xr.attrs['tip'] = 'Co_coated'
    elif 'AFM' in title:
        grid_xr.attrs['tip'] = 'AFM'
    else: 
        grid_xr.attrs['tip'] = 'To Be Announced'
        print('tip material will be announced later')
    
    if 'NbSe2' in title:
        grid_xr.attrs['sample'] = 'NbSe2'
    elif 'Cu(111)' in title:
        grid_xr.attrs['sample'] = 'Cu(111)'
    elif 'Au(111)' in title:
        grid_xr.attrs['sample'] = 'Au(111)'
    elif 'MoS2' in title:
        grid_xr.attrs['sample'] = 'MoS2'
    elif 'FeTe0.55Se0.45' in title:
        grid_xr.attrs['sample'] = 'FeTe0.55Se0.45'
    else: 
        grid_xr.attrs['sample'] = 'To Be Announced'
        print('sample type will be announced later')
        
    if '40mK' in title:
        grid_xr.attrs['temperature'] = '40mK'
    elif 'LHe' in title:
        grid_xr.attrs['temperature'] = 'LHe'
    elif 'LN2T' in title:
        grid_xr.attrs['temperature'] = 'LN2T'
    elif 'RT' in title:
        grid_xr.attrs['temperature'] = 'RT'
    else: 
        grid_xr.attrs['temperature'] = 'To Be Announced'
        print('temperature will be announced later')


    
    
    return grid_xr


# %% [markdown]
# ## `img2xr()` (SXM → `xarray.Dataset`)
#
# This is the same `img2xr_updated` logic you provided (robust multipass detection + NetCDF-safe attrs),
# kept as an I/O-only function.
#
# Stage-0 should ensure dependencies are available. If `nanonispy` is missing, this function raises a clear error.
#

# %%
def img2xr(loading_sxm_file: str, center_offset: bool = False) -> xr.Dataset:
    """Load a Nanonis .sxm file and convert it to an xarray.Dataset.

    Multipass detection order:
      (1) exact header key 'multipass-config'
      (2) any header key containing 'multipass' (case-insensitive)
      (3) any signal key containing 'P<number>'

    Produces NetCDF-safe attributes (dicts/lists-with-dicts serialized as JSON).

    IMPORTANT:
    - In single-pass mode, LIX is detected ONLY by names containing LI and X (case-insensitive).
      CURRENT is not treated as a substitute for LIX.
    - CURRENT, if present, is saved separately as CURR_fwd / CURR_bwd.
    """
    try:
        import nanonispy as nap
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "nanonispy is required for reading .sxm files. "
            "Please install dependencies (run env_check / Stage-0) and restart the kernel."
        ) from e

    # ----------------------------- NetCDF-safe attribute sanitizer -----------------------------
    def _sanitize_attr_value(v):
        if isinstance(v, (bool, np.bool_)):
            return int(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, dict):
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                return str(v)
        if isinstance(v, (list, tuple)):
            if any(isinstance(x, dict) for x in v):
                try:
                    return json.dumps(v, ensure_ascii=False)
                except Exception:
                    return str(v)
            out = []
            for x in v:
                if isinstance(x, (bool, np.bool_)):
                    out.append(int(x))
                elif isinstance(x, (np.integer,)):
                    out.append(int(x))
                elif isinstance(x, (np.floating,)):
                    out.append(float(x))
                elif isinstance(x, np.ndarray):
                    if x.dtype.kind in ("i", "u", "f"):
                        out.extend([float(xx) for xx in x.ravel().tolist()])
                    else:
                        out.append(str(x.tolist()))
                elif isinstance(x, bytes):
                    out.append(x.decode("utf-8", errors="ignore"))
                elif isinstance(x, (str, int, float)):
                    out.append(x)
                else:
                    out.append(str(x))
            return out
        if isinstance(v, np.ndarray):
            if v.dtype.kind in ("i", "u", "f"):
                return v.astype(float).tolist()
            return str(v.tolist())
        if v is None:
            return ""
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="ignore")
        if isinstance(v, str):
            return v
        return str(v)

    def _sanitize_dataset_attrs(ds: xr.Dataset) -> xr.Dataset:
        ds = ds.copy()
        ds.attrs = {k: _sanitize_attr_value(v) for k, v in ds.attrs.items()}
        for name, var in ds.variables.items():
            if var.attrs:
                var.attrs = {k: _sanitize_attr_value(v) for k, v in var.attrs.items()}
        return ds

    # ----------------------------- helpers -----------------------------
    _NUM_PAT = re.compile(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?")

    def parse_signed_float(x, *, assume_unit="V", allow_nan=True):
        if x is None:
            return np.nan if allow_nan else 0.0
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.upper() in {"N/A", "NA", "NONE"}:
            return np.nan if allow_nan else 0.0
        s = (s.replace("\u2212", "-")
               .replace("\u2013", "-")
               .replace("\u2014", "-")
               .replace("\u2213", "+/-"))
        has_mV = "mv" in s.lower()
        has_V = " v" in s.lower() or s.lower().endswith("v")
        s_clean = (s.replace(",", " ")
                   .replace("mV", " ").replace("MV", " ")
                   .replace("v", " ").replace("V", " ")
                   .strip())
        m = _NUM_PAT.search(s_clean)
        if not m:
            try:
                return float(s)
            except Exception:
                return np.nan if allow_nan else 0.0
        val = float(m.group(0))
        if has_mV and not has_V:
            return val / 1000.0
        if (not has_mV) and (not has_V):
            return val if assume_unit.upper() == "V" else (val / 1000.0 if assume_unit.upper() == "MV" else val)
        return val

    def as_float_scalar(a):
        try:
            return float(np.asarray(a).item() if np.ndim(a) == 0 else np.asarray(a)[()])
        except Exception:
            return float(a)

    def flip_xy(arr, flip_x=False, flip_y=False):
        if flip_x:
            arr = arr[:, ::-1]
        if flip_y:
            arr = arr[::-1, :]
        return arr

    def fillna_mean(arr):
        a = np.array(arr, dtype=float)
        if np.isnan(a).any():
            m = np.nanmean(a)
            if np.isnan(m):
                m = 0.0
            a = np.nan_to_num(a, nan=m)
        return a

    def classify_channel_name(sig_key):
        sk = str(sig_key).upper()
        m = re.search(r"P\s*(\d+)", sk)
        pidx = int(m.group(1)) if m else None
        if "Z" in sk and "LI" not in sk:
            kind = "Z"
        elif "LI" in sk and "X" in sk:
            kind = "LI_X"
        elif "LI" in sk and "Y" in sk:
            kind = "LI_Y"
        elif "CURRENT" in sk:
            kind = "Current"
        else:
            kind = "Other"
        return kind, pidx

    # ----------------------------- open file + tolerant header access -----------------------------
    NF = nap.read.NanonisFile(loading_sxm_file)
    Scan = nap.read.Scan(NF.fname)

    def get_case_insensitive_key(d, candidates):
        if not isinstance(d, dict):
            return None
        low = {str(k).lower(): k for k in d.keys()}
        for c in candidates:
            if c.lower() in low:
                return low[c.lower()]
        return None

    k_bias = get_case_insensitive_key(Scan.header, ["bias>bias (v)", "bias>bias (V)", "bias"])
    k_setpt = get_case_insensitive_key(Scan.header, ["z-controller>setpoint", "z-controller>setpoint (A)", "setpoint"])

    V_b = parse_signed_float(Scan.header[k_bias], assume_unit="V") if k_bias else np.nan
    I_t = parse_signed_float(Scan.header[k_setpt], assume_unit="A") if k_setpt else np.nan
    if np.isnan(V_b):
        V_b = 0.0
    if np.isnan(I_t):
        I_t = 0.0

    size_x, size_y = Scan.header["scan_range"]
    cntr_x, cntr_y = Scan.header["scan_offset"]
    dim_px, dim_py = Scan.header["scan_pixels"]
    step_dx, step_dy = size_x / dim_px, size_y / dim_py
    Rot_Rad = math.radians(float(Scan.header["scan_angle"]))
    scan_dir = Scan.header.get("scan_dir", "up")
    basename = getattr(Scan, "basename", NF.fname)

    # ----------------------------- multipass detection -----------------------------
    if "multipass-config" in Scan.header.keys():
        mp_cfg = Scan.header.get("multipass-config", {})
        is_multipass = True
    else:
        header_keys_lower = {str(k).lower(): k for k in Scan.header.keys()}
        mp_header_key = None
        for lk, orig in header_keys_lower.items():
            if "multipass" in lk:
                mp_header_key = orig
                break
        has_mp_cfg = mp_header_key is not None
        mp_cfg = Scan.header.get(mp_header_key, {}) if has_mp_cfg else {}
        if not isinstance(mp_cfg, dict):
            mp_cfg = {}
        has_pnum = any(re.search(r"P\s*\d+", str(k), flags=re.IGNORECASE) for k in Scan.signals.keys())
        is_multipass = bool(has_mp_cfg or has_pnum)

    # ----------------------------- coords -----------------------------
    X_idx = np.arange(dim_px)
    Y_idx = np.arange(dim_py)
    X_coords = (X_idx + 0.5) * step_dx
    Y_coords = (Y_idx + 0.5) * step_dy
    ds = xr.Dataset(coords=dict(X=("X", X_coords), Y=("Y", Y_coords)))

    # ----------------------------- multipass path -----------------------------
    if is_multipass:
        bias_map = {}
        values = mp_cfg.get("Bias_override_value", [])
        if isinstance(values, (str, int, float)):
            values = [values]
        try:
            vals = [parse_signed_float(v, assume_unit="V") for v in values]
            vals = [float(v) for v in vals if not np.isnan(v)]
        except Exception:
            vals = []
        n_passes = len(vals) // 2 if len(vals) >= 2 else 0
        if n_passes > 0:
            for k in range(n_passes):
                bias_map[(k + 1, "forward")] = float(vals[2 * k + 0])
                bias_map[(k + 1, "backward")] = float(vals[2 * k + 1])

        channels_by_pass = {}
        for key in Scan.signals.keys():
            kind, pidx = classify_channel_name(key)
            if pidx is None:
                continue
            channels_by_pass.setdefault(pidx, {})
            if kind not in channels_by_pass[pidx]:
                channels_by_pass[pidx][kind] = key

        flip_y = str(scan_dir).lower() == "down"

        def _name_for(k):
            return "Z" if k == "Z" else ("LIX" if k == "LI_X" else ("LIY" if k == "LI_Y" else "CURR"))

        for pidx in sorted(channels_by_pass.keys()):
            pass_ch = channels_by_pass[pidx]
            kinds = []
            if "Z" in pass_ch:
                kinds.append("Z")
            if "LI_X" in pass_ch:
                kinds.append("LI_X")
            elif "Current" in pass_ch:
                kinds.append("Current")
            if "LI_Y" in pass_ch:
                kinds.append("LI_Y")

            for kind in kinds:
                src_key = pass_ch[kind]
                sig = Scan.signals[src_key]
                fwd = np.array(sig.get("forward", None)) if "forward" in sig else None
                bwd = np.array(sig.get("backward", None)) if "backward" in sig else None

                if fwd is not None:
                    fwd = fillna_mean(fwd)
                    fwd = flip_xy(fwd, flip_x=False, flip_y=flip_y)
                if bwd is not None:
                    bwd = fillna_mean(bwd)
                    bwd = flip_xy(bwd, flip_x=True, flip_y=flip_y)

                bias_fwd_V = float(bias_map.get((pidx, "forward"), V_b))
                bias_bwd_V = float(bias_map.get((pidx, "backward"), V_b))

                if fwd is not None:
                    var_name = f"{_name_for(kind)}_P{pidx}_fwd"
                    ds[var_name] = xr.DataArray(fwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
                    ds[var_name].attrs.update(
                        dict(
                            kind=("Z" if kind == "Z" else ("LI_X" if kind == "LI_X" else ("LI_Y" if kind == "LI_Y" else "Current"))),
                            pass_index=int(pidx),
                            direction="forward",
                            bias_V=bias_fwd_V,
                            bias_mV=1000.0 * bias_fwd_V,
                            setpoint_A=I_t,
                            bias_set_V=V_b,
                            scan_angle_deg=float(math.degrees(Rot_Rad)),
                            source_channel_name=str(src_key),
                            flip_applied_Y=bool(flip_y),
                            nan_filled="mean_of_array",
                        )
                    )

                if bwd is not None:
                    var_name = f"{_name_for(kind)}_P{pidx}_bwd"
                    ds[var_name] = xr.DataArray(bwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
                    ds[var_name].attrs.update(
                        dict(
                            kind=("Z" if kind == "Z" else ("LI_X" if kind == "LI_X" else ("LI_Y" if kind == "LI_Y" else "Current"))),
                            pass_index=int(pidx),
                            direction="backward",
                            bias_V=bias_bwd_V,
                            bias_mV=1000.0 * bias_bwd_V,
                            setpoint_A=I_t,
                            bias_set_V=V_b,
                            scan_angle_deg=float(math.degrees(Rot_Rad)),
                            source_channel_name=str(src_key),
                            flip_applied_Y=bool(flip_y),
                            nan_filled="mean_of_array",
                        )
                    )

        if not np.isclose(step_dx, step_dy):
            ny, nx = ds.sizes["Y"], ds.sizes["X"]
            x0, x1 = as_float_scalar(ds["X"].values[0]), as_float_scalar(ds["X"].values[-1])
            y0, y1 = as_float_scalar(ds["Y"].values[0]), as_float_scalar(ds["Y"].values[-1])
            ratio = step_dy / step_dx
            ny_new = max(int(round(ny * ratio)), 1)
            X_new = np.linspace(float(x0), float(x1), int(nx))
            Y_new = np.linspace(float(y0), float(y1), int(ny_new))
            ds = ds.interp(X=X_new, Y=Y_new, method="linear")
            eff_dx = (float(X_new[-1]) - float(X_new[0])) / max(len(X_new) - 1, 1)
            eff_dy = (float(Y_new[-1]) - float(Y_new[0])) / max(len(Y_new) - 1, 1)
        else:
            eff_dx, eff_dy = step_dx, step_dy

        if not center_offset:
            ds = ds.assign_coords(
                X=(ds["X"] + (cntr_x - size_x / 2.0)),
                Y=(ds["Y"] + (cntr_y - size_y / 2.0)),
            )

        ds.attrs.update(
            dict(
                multipass=True,
                n_passes=int(len(channels_by_pass.keys())) if channels_by_pass else 1,
                image_size=[float(size_x), float(size_y)],
                X_spacing=float(eff_dx),
                Y_spacing=float(eff_dy),
                scan_angle_deg=float(math.degrees(Rot_Rad)),
                scan_dir=str(scan_dir),
                data_vars_list=list(ds.data_vars.keys()),
            )
        )

        ds = _sanitize_dataset_attrs(ds)
        return ds

    # ----------------------------- single-pass path -----------------------------
    def _prep(a, flipx=False, flipy=False):
        if a is None:
            return None
        a = fillna_mean(a)
        return flip_xy(a, flip_x=flipx, flip_y=flipy)

    z_fwd = np.array(Scan.signals["Z"]["forward"]) if "Z" in Scan.signals else None
    z_bwd = np.array(Scan.signals["Z"]["backward"])[:, ::-1] if "Z" in Scan.signals else None

    # STRICT LIX (no CURRENT fallback)
    lix_key = None
    for k in Scan.signals.keys():
        s = str(k).upper()
        if ("LI" in s) and ("X" in s):
            lix_key = k
            break
    lix_fwd = np.array(Scan.signals[lix_key]["forward"]) if lix_key else None
    lix_bwd = np.array(Scan.signals[lix_key]["backward"])[:, ::-1] if lix_key else None

    liy_key = None
    for k in Scan.signals.keys():
        s = str(k).upper()
        if ("LI" in s) and ("Y" in s):
            liy_key = k
            break
    liy_fwd = np.array(Scan.signals[liy_key]["forward"]) if liy_key else None
    liy_bwd = np.array(Scan.signals[liy_key]["backward"])[:, ::-1] if liy_key else None

    curr_key = None
    for k in Scan.signals.keys():
        if "CURRENT" in str(k).upper():
            curr_key = k
            break
    curr_fwd = np.array(Scan.signals[curr_key]["forward"]) if curr_key else None
    curr_bwd = np.array(Scan.signals[curr_key]["backward"])[:, ::-1] if curr_key else None

    flip_y = str(Scan.header.get("scan_dir", "up")).lower() == "down"
    z_fwd = _prep(z_fwd, flipx=False, flipy=flip_y)
    z_bwd = _prep(z_bwd, flipx=False, flipy=flip_y)
    lix_fwd = _prep(lix_fwd, flipx=False, flipy=flip_y)
    lix_bwd = _prep(lix_bwd, flipx=False, flipy=flip_y)
    liy_fwd = _prep(liy_fwd, flipx=False, flipy=flip_y)
    liy_bwd = _prep(liy_bwd, flipx=False, flipy=flip_y)
    curr_fwd = _prep(curr_fwd, flipx=False, flipy=flip_y)
    curr_bwd = _prep(curr_bwd, flipx=False, flipy=flip_y)

    if z_fwd is not None:
        ds["Z_fwd"] = xr.DataArray(z_fwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
    if z_bwd is not None:
        ds["Z_bwd"] = xr.DataArray(z_bwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
    if lix_fwd is not None:
        ds["LIX_fwd"] = xr.DataArray(lix_fwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
    if lix_bwd is not None:
        ds["LIX_bwd"] = xr.DataArray(lix_bwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
    if liy_fwd is not None:
        ds["LIY_fwd"] = xr.DataArray(liy_fwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
    if liy_bwd is not None:
        ds["LIY_bwd"] = xr.DataArray(liy_bwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
    if curr_fwd is not None:
        ds["CURR_fwd"] = xr.DataArray(curr_fwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})
    if curr_bwd is not None:
        ds["CURR_bwd"] = xr.DataArray(curr_bwd, dims=("Y", "X"), coords={"Y": ds["Y"], "X": ds["X"]})

    if not np.isclose(step_dx, step_dy):
        ny, nx = ds.sizes["Y"], ds.sizes["X"]
        x0, x1 = as_float_scalar(ds["X"].values[0]), as_float_scalar(ds["X"].values[-1])
        y0, y1 = as_float_scalar(ds["Y"].values[0]), as_float_scalar(ds["Y"].values[-1])
        ratio = step_dy / step_dx
        ny_new = max(int(round(ny * ratio)), 1)
        X_new = np.linspace(float(x0), float(x1), int(nx))
        Y_new = np.linspace(float(y0), float(y1), int(ny_new))
        ds = ds.interp(X=X_new, Y=Y_new, method="linear")
        eff_dx = (float(X_new[-1]) - float(X_new[0])) / max(len(X_new) - 1, 1)
        eff_dy = (float(Y_new[-1]) - float(Y_new[0])) / max(len(Y_new) - 1, 1)
    else:
        eff_dx, eff_dy = step_dx, step_dy

    if not center_offset:
        ds = ds.assign_coords(
            X=(ds["X"] + (cntr_x - size_x / 2.0)),
            Y=(ds["Y"] + (cntr_y - size_y / 2.0)),
        )

    ds.attrs.update(
        dict(
            multipass=False,
            n_passes=1,
            image_size=[float(size_x), float(size_y)],
            X_spacing=float(eff_dx),
            Y_spacing=float(eff_dy),
            scan_angle_deg=float(math.degrees(Rot_Rad)),
            scan_dir=str(scan_dir),
            data_vars_list=list(ds.data_vars.keys()),
        )
    )

    ds = _sanitize_dataset_attrs(ds)
    return ds


# %%
