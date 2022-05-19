#%%
import geopandas as gpd
import shutil
import time
from osgeo import ogr, gdal
#%%
start = time.perf_counter()
#### list of tiles that we need
### CHANGE THIS LINE user supplies km10 shape as a .shp file with the tiles we want
# km10 =  ""
############################################################################################
# THIS IS THE ONLY LINE YOU NEED TO EDIT

km10 = gpd.read_file("../data/pilot_training_data/Pilot_10km_grid.gpkg")

############################################################################################

km1 = gpd.read_file('../data/grids/dki_1km.gpkg')

tiles_gdf = gpd.overlay(km1, km10, how='intersection')
tiles_list = tiles_gdf['dki_1km'].tolist()


tiles_list = tiles_list[:10]

cp1 = time.perf_counter()

print('time to read files: ', cp1 - start)

# transfer tiles to destination directory
#
# PRESUMED TILE LOCATIONS/////////////////////////////////////////////////////////
# DTM_source_dir = 'M:/Ekstern_datasamling/Danmark/Frie_DATA/DTM_2019/DTM_Grid_1km_TIFF/'
# DSM_source_dir = 'M:/Ekstern_datasamling/Danmark/Frie_DATA/DTM_2019/DSM_Grid_1km_TIFF/'


dest_dir = 'V:/2022-03-31_Stendiger_EZRA/data/'
if not os.path.isdir(dest_dir):
    raise Exception('destination directory does not exist')

dtm_tifs = []
dsm_tifs = []

for i,t in enumerate(tiles_list):

    # prepare file names
    source_DTM = 'M:/Ekstern_datasamling/Danmark/Frie_DATA/DTM_2019/DTM_Grid_1km_TIFF/DTM_{}.tif'.format(t)
    source_DSM = 'M:/Ekstern_datasamling/Danmark/Frie_DATA/DTM_2019/DSM_Grid_1km_TIFF/DSM_{}.tif'.format(t)

    # dest_dir = 'V:/2022-03-31_Stendiger_EZRA/data/'
    
    
    dest_DTM = '{}dtm/DTM_{}.tif'.format(dest_dir, t)
    dest_DSM = '{}dsm/DSM_{}.tif'.format(dest_dir, t)

    dtm_tifs.append(dest_DTM)
    dsm_tifs.append(dest_DSM)

    # execute transfer
    if not os.path.isfile(dest_DTM):
        shutil.copy(source_DTM, dest_DTM)
    if not os.path.isfile(dest_DSM):
        shutil.copy(source_DSM, dest_DSM)

    if (i)%10 == 0:
    # print progress
        print('{}/{}'.format(i - len(tiles_list), i))




#### validate transfer

count = 0
untransfered_tiles = []
for t in tiles_list:

    dest_DTM = '{}dtm/DTM_{}.tif'.format(dest_dir, t)
    dest_DSM = '{}dsm/DSM_{}.tif'.format(dest_dir, t)

    if not os.path.exists(dest_DTM):
        print(dest_DTM, '... is in the tile list, but not on V drive')
        count += 1
        untransfered_tiles.append(t)
        
    if not os.path.exists(dest_DSM):
        print(dest_DSM, '... is in the tile list, but not on V drive')
        count += 1
        if (t not in untransfered_tiles):
            untransfered_tiles.append(t)
print('{} tiles were not transfered'.format(count))
print("untransfered_tiles holds: {} files".format(untransfered_tiles))
print("dtm_tifs holds: {} files".format(len(dtm_tifs)))
print("dsm_tifs holds: {} files".format(len(dsm_tifs)))

cp2 = time.perf_counter()

print('time to transfer tiles: ', cp2 - cp1)

#%%
############################################################################################
#######################################################################
####################GET ALL CURRENT TILES IN THE FOLDERS#######################
#######################################################################
#######################################################################

### run this if the tiles have already been transferred
import glob
dest_dir = 'V:/2022-03-31_Stendiger_EZRA/data/'
dtm_tifs = glob.glob('{}dtm/*.tif'.format(dest_dir))
dsm_tifs = glob.glob('{}dsm/*.tif'.format(dest_dir))

dtm_tifs.sort()
dsm_tifs.sort()

for dtm, dsm in zip(dtm_tifs, dsm_tifs):
    if not (os.path.basename(dtm) == os.path.basename(dsm).replace('DSM', 'DTM')):
        ### this exception means that the contents of the folders do not match
        raise Exception
    
print('all tiles are present')


#%%
############################################################################################
##############CREATE HAT AND SOBEL FILTERS####################
#######################################################################
from osgeo import ogr, gdal
import numpy as np
from zobel_filter import zobel_filter

yellow_path = "V:/2022-03-31_Stendiger_EZRA/buteo"
import sys; sys.path.append(yellow_path); sys.path.append(yellow_path + 'buteo/'); sys.path.append(yellow_path + 'buteo/machine_learning/'); sys.path.append(yellow_path + 'buteo/filters/'); sys.path.append(yellow_path + 'buteo/raster/'); sys.path.append(yellow_path + 'buteo/convolutions/')
from buteo.raster.io import *
from scipy import ndimage

for dsm, dtm in zip(dsm_tifs, dtm_tifs):

    dsm_raster = gdal.Open(dsm)
    dsm_bandarr = dsm_raster.GetRasterBand(1).ReadAsArray()
    dsm_npy = np.array(dsm_bandarr)

    dtm_raster = gdal.Open(dtm)
    dtm_bandarr = dtm_raster.GetRasterBand(1).ReadAsArray()
    dtm_npy = np.array(dtm_bandarr)


    # sobel_npy = zobel_filter(
    #         dtm_npy, size=[5, 5], normalised_sobel=False, gaussian_preprocess=False
    #     )
    hat_npy = dsm_npy - dtm_npy
    sobel_npy = ndimage.sobel(dtm_npy, axis=-1, mode='constant', cval=0)

    sobel_path = dtm.replace('DTM', 'SOBEL1').replace('dtm', 'sobel')
    hat_path = dtm.replace('DTM', 'HAT').replace('dtm', 'hat')

    array_to_raster(sobel_npy, reference=dtm, out_path=sobel_path, creation_options=["COMPRESS=LZW"])
    array_to_raster(hat_npy, reference=dtm, out_path=hat_path, creation_options=["COMPRESS=LZW"])
    print('done with', dtm)

# # create vrt's
# 
#%%
### run this if the tiles have already been transferred and hat and sobel have been made
import glob

dest_dir = 'V:/2022-03-31_Stendiger_EZRA/data/'
dtm_tifs = glob.glob('{}dtm/*.tif'.format(dest_dir))
dsm_tifs = glob.glob('{}dsm/*.tif'.format(dest_dir))
hat_tifs = glob.glob('{}hat/*.tif'.format(dest_dir))
sobel_tifs = glob.glob('{}sobel/*.tif'.format(dest_dir))

dtm_tifs.sort()
dsm_tifs.sort()
hat_tifs.sort()
sobel_tifs.sort()


for dtm, dsm, hat, sobel in zip(dtm_tifs, dsm_tifs, hat_tifs, sobel_tifs):
    if not (os.path.basename(dtm) == os.path.basename(dsm).replace('DSM', 'DTM') == os.path.basename(hat).replace('HAT', 'DTM') == os.path.basename(sobel).replace('SOBEL', 'DTM')):
        ### this exception means that the contents of the folders do not match
        raise Exception("catastrphic error")

    t = os.path.basename(dtm).replace('DTM_', '').replace('.tif', '.vrt')
    vrt = 'V:/2022-03-31_Stendiger_EZRA/data/vrts/' + t
    print(vrt)  

    gdal.BuildVRT(vrt, [dtm, hat, sobel], options=gdal.BuildVRTOptions(separate=True))


cp3 = time.perf_counter()
print('time to create vrts: ', cp3 - cp2)
print('total time: ', cp3 - start)

# %%


