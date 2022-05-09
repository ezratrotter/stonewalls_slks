#%%
yellow_follow = "//wsl$/Ubuntu-20.04/home/afer/yellow/"
from json import load
import os
import sys
from turtle import setundobuffer

# from debugpy import listen 
sys.path.append(yellow_follow)
sys.path.append(os.path.join(yellow_follow, "buteo/"))


import math
import time
import numpy as np
from glob import glob
# from osgeo import gdal, ogr
import pandas as pd
import geopandas as gpd
import subprocess

# import ml_utils 
from machine_learning.patch_extraction import extract_patches
from raster.io import raster_to_array, raster_to_metadata, internal_raster_to_metadata
from raster.clip import clip_raster
from vector.io import open_vector
from vector.reproject import internal_reproject_vector
#import terrain.dtm


#%%
import matplotlib.pyplot as plt

out_path = "R:/PROJ/10/415/217/10_Databehandling/102_Training_Data/patches_arrays/"
in_path = "V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/"

#%%

vrt = in_path + "dtm/dtm.vrt"

vectors = glob("R:/PROJ/10/415/217/10_Databehandling/102_Training_Data/patches_geoms/" + "*.gpkg")
# walls_vector = "R:/PROJ/10/415/217/10_Databehandling/102_Training_Data/patches_geoms/walls_10km_611_57_patches_geom_64.gpkg"
# abs_vector = "R:/PROJ/10/415/217/10_Databehandling/102_Training_Data/patches_geoms/absence_10km_611_57_patches_geom_64.gpkg"

#%%
# num = 0

# for tif in tiled_tifs:
#     num = num + 1
#     # print("raster_" + str(num))
#     print("extracting tile {0}".format(num))

for vector in vectors:
    print("extracting " + vector + "...")
    extract_patches(
        vrt,
        out_dir=out_path,
        # prefix=str(num)+"_",
        prefix="",
        postfix="",
        size=64,
        offsets=None,
        generate_grid_geom=False,
        generate_border_patches=False,
        clip_geom=vector,
        verify_output=True,
        overwrite=True,
        verification_samples=100,
        verbose=1,
    )
print("finished extracting patches.")

#%%

##TODO
# define model

# for vector in liste
#     patches = extract_patches(etc)

#     do setundobuffer
    
    
#     train model.


# save model

#%%

# # aligned = folder + "hot_aeroe.tif"
# # numpy_arrays = out_path + "aspect_cos_aeroe.npy"
# # grid = out + "patches_64_patches_hot.gpkg"

# # test_extraction(aligned, numpy_arrays, grid)

# #import pdb; pdb.set_trace()

# files = sorted(glob(out_path + '/*.npy'))
# arrays = []
# for f in files:
#     arrays.append(np.load(f))
# data = np.concatenate(arrays)

# np.save(out_path + "aspect_sin_aeroe_64.npy", data)