#%%
yellow_follow = "//wsl$/Ubuntu-20.04/home/afer/yellow/"
import os
import sys 
sys.path.append(yellow_follow)
sys.path.append(os.path.join(yellow_follow, "buteo/"))


import math
import time
import numpy as np
from glob import glob
# from osgeo import gdal, ogr
import pandas as pd
import geopandas as gpd


# import ml_utils 
from machine_learning.patch_extraction import extract_patches
from raster.io import raster_to_array, raster_to_metadata, internal_raster_to_metadata
from raster.clip import clip_raster
from vector.io import open_vector
from vector.reproject import internal_reproject_vector
#import terrain.dtm


#%%
import matplotlib.pyplot as plt

out_path = "R:/PROJ/10/415/217/10_Databehandling/102_Training_Data/patches/"
in_path = "V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dtm/"

#%%

km10training = gpd.read_file(r'V:\2022-03-31_Stendiger_EZRA\stonewalls_slks\data\pilot_training_data\training_10km_grid.gpkg')

km10_tile = km10training[km10training['KN10kmDK'] == "10km_611_57"]

km1 = gpd.read_file(r'V:\2022-03-31_Stendiger_EZRA\stonewalls_slks\data\grids\dki_1km.gpkg')

km1_list = gpd.overlay(km1, km10_tile, how='intersection')['dki_1km'].tolist()

#%%

dtm_tiles = glob(in_path + "*.tif")

for tif in dtm_tiles:
    dtm_tile_base = os.path.basename(tif).split('/')[-1]
    dtm_tile_name = str((dtm_tile_base).split(".")[0].split("DTM_")[1])
##works until here##

    our_tifs = [x for x in dtm_tile_name if x in km1_list]
    print(our_tifs)
    # if tif in km1_list:
    #     print(tif)

#%%
file_names = [x.split('.')[0] for x in os.path.basename(in_path)]
# our_tifs = [x for x in file_names if x in km1_list]




file_names
# our_tifs
# km1_list
# file_names
# for tif in our_tifs:
#     print(tif)



#%%
tiled_tifs = glob(in_path + "aspect_sin*.tif")

# tif = glob(in_path + "absence_10km_611_57_bin.tif")

vector = "R:/PROJ/10/415/217/10_Databehandling/102_Training_Data/absence/absence_10km_611_57_patches_64.gpkg"


#%%
# raster1 = None
# band = None

# #check vector
# geom = ogr.Open(vector, 1)
# layer = geom.GetLayer()
# featureCount = layer.GetFeatureCount()

# metadata = internal_raster_to_metadata(raster)
# clip_layer_index = 0

# clip_ref = open_vector(
#     internal_reproject_vector(vector, metadata["projection_osr"])
# )

# clip_layer = clip_ref.GetLayerByIndex(clip_layer_index)

#import pdb; pdb.set_trace()
#%%
# geom = None
# layer = None
# featureCount = None
#%%
num = 0

# for tif in tiled_tifs:
#     num = num + 1
#     # print("raster_" + str(num))
#     print("extracting tile {0}".format(num))

extract_patches(
    tif,
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

#%%

# aligned = folder + "hot_aeroe.tif"
# numpy_arrays = out_path + "aspect_cos_aeroe.npy"
# grid = out + "patches_64_patches_hot.gpkg"

# test_extraction(aligned, numpy_arrays, grid)

#import pdb; pdb.set_trace()

files = sorted(glob(out_path + '/*.npy'))
arrays = []
for f in files:
    arrays.append(np.load(f))
data = np.concatenate(arrays)

np.save(out_path + "aspect_sin_aeroe_64.npy", data)