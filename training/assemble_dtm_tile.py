#%%
import geopandas as gpd
import os
import glob
import numpy as np

km10 = gpd.read_file(r'D:\stonewalls_slks\data\pilot_training_data\training_10km_grid.gpkg')
km1 = gpd.read_file(r'D:\stonewalls_slks\data\grids\dki_1km.gpkg')

bob = np.array(km1.intersects(km10).to_list()) 

import pdb; pdb.set_trace()


# pilot_grids_gdf = gpd.read_file(r'D:\stonewalls_slks\data\pilot_training_data\training_10km_grid.gpkg')
# # grids_list = pilot_grids_gdf['KN10kmDK'].tolist()


# all_1km_grids_gdf = gpd.read_file(r'D:\stonewalls_slks\data\grids\dki_1km.gpkg')


# our_1km_grids_gdf = all_1km_grids_gdf[all_1km_grids_gdf['dki_10km'].isin(grids_list)]

# # %%
# pilot_grids_list = our_1km_grids_gdf['dki_1km'].tolist()
# pilot_files_list = ['DTM_' + x for x in pilot_grids_list]
# dir_to_tifs = r'M:\Ekstern_datasamling\Danmark\Frie_DATA\DTM_2019\DTM_Grid_1km_TIFF' 

# list_of_tifs = glob.glob(os.path.join(dir_to_tifs, '*.tif'))
# # %%
# smaller_list = [x for x in list_of_tifs if os.path.basename(os.path.splitext(x)[0]) in pilot_files_list]
# # os.path.basename(os.path.splitext(path)[0])
# #%%
# # if move to local

# local_folder = 'D:/tifs'
# os.path.isdir(local_folder)

# [shutil.copy(x, local_folder + '/' + pilot_files_list[i]) for x in enumerate(smaller_list)]

#%%


# from osgeo import gdal 

# vrt_options = gdal.BuildVRTOptions(resampleAlg='cubic', addAlpha=True)
# my_vrt = gdal.BuildVRT('my.vrt', smaller_list)
# my_vrt = None
