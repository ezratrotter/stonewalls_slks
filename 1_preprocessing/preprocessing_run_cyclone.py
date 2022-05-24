# #%%
# import glob
# import geopandas as gpd
# import shutil
# import time
# from osgeo import ogr, gdal
# import numpy as np
# from zobel_filter import zobel_filter

# yellow_path = "V:/2022-03-31_Stendiger_EZRA/buteo"
# import sys; sys.path.append(yellow_path); sys.path.append(yellow_path + 'buteo/'); sys.path.append(yellow_path + 'buteo/machine_learning/'); sys.path.append(yellow_path + 'buteo/filters/'); sys.path.append(yellow_path + 'buteo/raster/'); sys.path.append(yellow_path + 'buteo/convolutions/')
# from buteo.raster.io import *
# from scipy import ndimage

#%%

# km10 = gpd.read_file(r"\\niras.int\root\PROJ\10\415\217\20_Aflevering/raekkefoelge.gpkg")

# ############################################################################################

# tiles_list = km10[km10['lev_blok'] == 1]['tilename'].tolist()
# base_path = '//pc116900/S Drone div/STENDIGER'

#################################
### MAKE HAT AND SOBEL LAYERS ###
#################################

# dtm_vrt_list = []
# hat_vrt_list = []
# sobel_vrt_list = []

# #
# for tile in tiles_list:

#     dsm_folder = '{}/DSM/{}_TIF_UTM32-ETRS89/'.format(base_path, tile.replace('10km', 'DSM'))
#     dtm_folder = '{}/DTM/{}_TIF_UTM32-ETRS89/'.format(base_path, tile.replace('10km', 'DTM'))
#     hat_folder = dtm_folder.replace('DTM', 'HAT')
#     sobel_folder = dtm_folder.replace('DTM', 'SOBEL')
    
#     # print(hat_folder)
#     # print(sobel_folder)
#     if not os.path.exists(hat_folder):
#         os.makedirs(hat_folder, exist_ok = False)
#     if not os.path.exists(sobel_folder):
#         os.makedirs(sobel_folder, exist_ok = False)

#     dsm_tifs = glob.glob(dsm_folder + '*.tif')    
#     dtm_tifs = glob.glob(dtm_folder + '*.tif')

#     dsm_tifs.sort()
#     dtm_tifs.sort()
#     # for dtm, dsm in zip(dtm_tifs, dsm_tifs):
#     #     if not (os.path.basename(dtm) == os.path.basename(dsm).replace('DSM', 'DTM')):
#     #         ### this exception means that the contents of the folders do not match
#     #         raise Exception
#     # print('all tiles are present')
#     for dsm, dtm in zip(dsm_tifs, dtm_tifs):

#         dsm_raster = gdal.Open(dsm)
#         dsm_bandarr = dsm_raster.GetRasterBand(1).ReadAsArray()
#         dsm_npy = np.array(dsm_bandarr)

#         dtm_raster = gdal.Open(dtm)
#         dtm_bandarr = dtm_raster.GetRasterBand(1).ReadAsArray()
#         dtm_npy = np.array(dtm_bandarr)


#         sobel_npy = zobel_filter(
#                 dtm_npy, size=[5, 5], normalised_sobel=False, gaussian_preprocess=False
#             )

#         # sobel_npy = ndimage.sobel(dtm_npy, axis=-1, mode='constant', cval=0)

#         hat_npy = dsm_npy - dtm_npy

#         sobel_path = dtm.replace('DTM', 'SOBEL')
#         hat_path = dtm.replace('DTM', 'HAT')

#         array_to_raster(sobel_npy, reference=dtm, out_path=sobel_path, creation_options=["COMPRESS=LZW"])
#         array_to_raster(hat_npy, reference=dtm, out_path=hat_path, creation_options=["COMPRESS=LZW"])
        
#         dtm_vrt_list.append(dtm)
#         hat_vrt_list.append(hat_path)
#         sobel_vrt_list.append(sobel_path)
        
#         print('done with', dtm)

# #%%

<<<<<<< HEAD
# for dtm, sobel, hat in zip(dtm_vrt_list, sobel_vrt_list, hat_vrt_list,):
#     if not (os.path.basename(dtm) == os.path.basename(hat).replace('HAT', 'DTM') == os.path.basename(sobel).replace('SOBEL', 'DTM')):
#         ### this exception means that the contents of the folders do not match
#         raise Exception("catastrphic error")
=======
#################################
##### MAKE STACKED 1KM VRTS #####
#################################

for dtm, sobel, hat in zip(dtm_vrt_list, sobel_vrt_list, hat_vrt_list,):
    if not (os.path.basename(dtm) == os.path.basename(hat).replace('HAT', 'DTM') == os.path.basename(sobel).replace('SOBEL', 'DTM')):
        ### this exception means that the contents of the folders do not match
        raise Exception("catastrphic error")
>>>>>>> 067f782612c8ce2aa33fe6f6ee3360e56f66347a

#     t = os.path.basename(dtm).replace('DTM_', '').replace('.tif', '.vrt')
#     vrt = '//pc116900/S Drone div/STENDIGER/vrts/' + t
#     print(vrt)  

#     gdal.BuildVRT(vrt, [dtm, hat, sobel], options=gdal.BuildVRTOptions(separate=True))

<<<<<<< HEAD
# #%%
# vrt_dir = '//pc116900/S Drone div/STENDIGER/vrts/'
=======
#%%
#################################
######## MAKE BIG VRT ###########
#################################
vrt_dir = '//pc116900/S Drone div/STENDIGER/vrts/'
>>>>>>> 067f782612c8ce2aa33fe6f6ee3360e56f66347a

# vrts = glob.glob(vrt_dir + '*.vrt')

<<<<<<< HEAD
# gdal.BuildVRT('//pc116900/S Drone div/STENDIGER/vrts/merged.vrt', vrts)
=======
## extent of whole Danmark 10km tiles (land)
[xmax, xmin, ymax, ymin] = [900000, 440000, 6410000, 6040000]
>>>>>>> 067f782612c8ce2aa33fe6f6ee3360e56f66347a

ds = gdal.BuildVRT('//pc116900/S Drone div/STENDIGER/vrts/merged.vrt', vrts, options=gdal.BuildVRTOptions(outputBounds=(xmin, ymin, xmax, ymax)))
ds.FlushCache()



# #%%
# source_root = '//pc116900/S Drone div/STENDIGER/'


# #%%
# ### run this if the tiles have already been transferred and hat and sobel have been made
# # # create vrt's
# # 
# import glob

# dest_dir = 'V:/2022-03-31_Stendiger_EZRA/data/'
# dtm_tifs = glob.glob('{}dtm/*.tif'.format(dest_dir))
# dsm_tifs = glob.glob('{}dsm/*.tif'.format(dest_dir))
# hat_tifs = glob.glob('{}hat/*.tif'.format(dest_dir))
# sobel_tifs = glob.glob('{}sobel/*.tif'.format(dest_dir))

# dtm_tifs.sort()
# dsm_tifs.sort()
# hat_tifs.sort()
# sobel_tifs.sort()


# for dtm, dsm, hat, sobel in zip(dtm_tifs, dsm_tifs, hat_tifs, sobel_tifs):
#     if not (os.path.basename(dtm) == os.path.basename(dsm).replace('DSM', 'DTM') == os.path.basename(hat).replace('HAT', 'DTM') == os.path.basename(sobel).replace('SOBEL', 'DTM')):
#         ### this exception means that the contents of the folders do not match
#         raise Exception("catastrphic error")

#     t = os.path.basename(dtm).replace('DTM_', '').replace('.tif', '.vrt')
#     vrt = 'V:/2022-03-31_Stendiger_EZRA/data/vrts/' + t
#     print(vrt)  

#     gdal.BuildVRT(vrt, [dtm, hat, sobel], options=gdal.BuildVRTOptions(separate=True))


# cp3 = time.perf_counter()
# print('time to create vrts: ', cp3 - cp2)
# print('total time: ', cp3 - start)

# %%



from pathlib import Path

import glob
import geopandas as gpd
import shutil
import time
from osgeo import ogr, gdal
import numpy as np
from zobel_filter import zobel_filter

# yellow_path = "V:/2022-03-31_Stendiger_EZRA/buteo"
# import sys; sys.path.append(yellow_path); sys.path.append(yellow_path + 'buteo/'); sys.path.append(yellow_path + 'buteo/machine_learning/'); sys.path.append(yellow_path + 'buteo/filters/'); sys.path.append(yellow_path + 'buteo/raster/'); sys.path.append(yellow_path + 'buteo/convolutions/')
# from buteo.raster.io import *
from scipy import ndimage

#%%
km10 = gpd.read_file(r"\\niras.int\root\PROJ\10\415\217\20_Aflevering/raekkefoelge.gpkg")

############################################################################################

tiles = km10[km10['lev_blok'] == 1]
tiles_list = km10[km10['lev_blok'] == 1]['geometry'].tolist()
buffered_tiles_list = [x.buffer(100.0) for x in tiles_list]

onekm = gpd.read_file(r'\\Niras.int\root\PROJ\10\415\217\20_Aflevering\dki_1km_lev1_buffered.gpkg')['dki_1km'].tolist()
formatted_onekm = ['DTM_{}.tif'.format(x) for x in onekm]
#%%


dtm_list = [str(x) for x in Path('//pc116900/S Drone div/STENDIGER/DTM').rglob('*.tif') if x.name in formatted_onekm]
dsm_list = [str(x) for x in Path('//pc116900/S Drone div/STENDIGER/DSM').rglob('*.tif') if x.name in formatted_onekm]

#%%
dest_sobel_list = [str(x) for x in Path('//pc116900/S Drone div/STENDIGER/SOBEL').rglob('*.tif')]
dest_hat_list = [str(x) for x in Path('//pc116900/S Drone div/STENDIGER/HAT').rglob('*.tif')]


#%%
dtm_list_filtered = [x for x in dtm_list if x.replace('DTM', 'SOBEL') not in dest_sobel_list]
dsm_list_filtered = [x for x in dtm_list if x.replace('DSM', 'SOBEL') not in dest_sobel_list]



# base_path = '//pc116900/S Drone div/STENDIGER'

#%%
# ### run this if the tiles have already been transferred
# import glob
# dest_dir = 'V:/2022-03-31_Stendiger_EZRA/data/'
# dtm_tifs = glob.glob('{}dtm/*.tif'.format(dest_dir))
# dsm_tifs = glob.glob('{}dsm/*.tif'.format(dest_dir))

# dtm_tifs.sort()
# dsm_tifs.sort()

for dtm, dsm in zip(dtm_list, dsm_list):
    if not (os.path.basename(dtm) == os.path.basename(dsm).replace('DSM', 'DTM')):
        ### this exception means that the contents of the folders do not match
        raise Exception
    
print('all tiles are present')

#%%
#%%
############################################################################################
##############CREATE HAT AND SOBEL FILTERS####################
#######################################################################
from osgeo import ogr, gdal
import numpy as np
from zobel_filter import zobel_filter
from pathlib import Path

yellow_path = "V:/2022-03-31_Stendiger_EZRA/buteo"
import sys; sys.path.append(yellow_path); sys.path.append(yellow_path + 'buteo/'); sys.path.append(yellow_path + 'buteo/machine_learning/'); sys.path.append(yellow_path + 'buteo/filters/'); sys.path.append(yellow_path + 'buteo/raster/'); sys.path.append(yellow_path + 'buteo/convolutions/')
from buteo.raster.io import *
from scipy import ndimage

for dsm, dtm in zip(dsm_list_filtered, dtm_list_filtered):

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
    sobel_path = dtm.replace('DTM', 'SOBEL').replace('dtm', 'sobel')
    hat_path = dtm.replace('DTM', 'HAT').replace('dtm', 'hat')
    
    sobel_path = r'//pc116900/S Drone div/STENDIGER/SOBEL/' + os.path.basename(sobel_path)
    hat_path = r'//pc116900/S Drone div/STENDIGER/HAT/' + os.path.basename(hat_path)
    
    print(sobel_path)
    print(hat_path)

    array_to_raster(sobel_npy, reference=dtm, out_path=sobel_path, creation_options=["COMPRESS=LZW"])
    array_to_raster(hat_npy, reference=dtm, out_path=hat_path, creation_options=["COMPRESS=LZW"])
    print('done with', dtm)
#%%



sobel_list = [str(x).replace('\\', '/') for x in Path('//pc116900/S Drone div/STENDIGER/SOBEL').rglob('*.tif')]
hat_list = [str(x).replace('\\', '/') for x in Path('//pc116900/S Drone div/STENDIGER/HAT').rglob('*.tif')]
dtm_big_list = [str(x).replace('\\', '/') for x in Path('//pc116900/S Drone div/STENDIGER/DTM').rglob('*.tif')]
#%%

buff10gdf = gpd.read_file(r'\\Niras.int\root\PROJ\10\415\217\20_Aflevering\lev1_10kmbuffered.gpkg') 
#%%

myDict = {}

for i, row in buff10gdf.iterrows():
    km1 = row['1kmtile_listname'].split(',')
    km10 = row['10km_plus_1km_name']

    # km10dir ='DTM_{}_TIF_UTM32-ETRS89'.format(km10.split('10km_')[1])
    myDict[km10] = {}
    
    
    
    for t in km1:
        dtm = [x for x in dtm_big_list if t in x]
        if len(dtm) != 1:
            raise Exception
        dtm = dtm[0]
        sobel = [x for x in sobel_list if t in x]
        if len(sobel) != 1:
            raise Exception
        sobel = sobel[0]
        hat = [x for x in hat_list if t in x]
        if len(hat) != 1:
            raise Exception
        hat = hat[0]
        myDict[km10][t] = [dtm, sobel, hat]
     


#%%

vrtdir = '//pc116900/S Drone div/STENDIGER/vrts/'

for k,v in myDict.items():
    for km1name, stack in v.items():
        if os.path.exists(vrtdir + km1name + '.vrt'):
            continue
        vrt = gdal.BuildVRT(vrtdir + km1name + '.vrt', stack, separate=True)





#%%
print(buff10gdf.columns)

def getExtent(geometry):
    coords = geometry.exterior.coords
    xmin = float('inf')
    ymin = float('inf')
    xmax = 0
    ymax = 0
    for (x, y) in coords:
        if x < xmin:
            xmin = x
        if y < ymin:
            ymin = y
        if x > xmax:
            xmax = x
        if y > ymax:
            ymax = y
    return (xmin, ymin, xmax, ymax)





for i, row in buff10gdf.iterrows():

    km1 = row['1kmtile_listname'].split(',')
    km10 = row['10km_plus_1km_name']
    
    km1paths = [vrtdir + x + '.vrt' for x in km1]
    extent = getExtent(row['geometry'])
    gdal.BuildVRT(vrtdir + '10km/' + km10 + '.vrt', km1paths, options=gdal.BuildVRTOptions(outputBounds=extent))



    # my1kms = [x for x in vrtsdir if km1[0] in x]

    
    
    
    

# 




















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


