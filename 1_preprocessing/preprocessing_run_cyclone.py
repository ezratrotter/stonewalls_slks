#%%
import glob
import geopandas as gpd
import shutil
import time
from osgeo import ogr, gdal
import numpy as np
from zobel_filter import zobel_filter

yellow_path = "V:/2022-03-31_Stendiger_EZRA/buteo"
import sys; sys.path.append(yellow_path); sys.path.append(yellow_path + 'buteo/'); sys.path.append(yellow_path + 'buteo/machine_learning/'); sys.path.append(yellow_path + 'buteo/filters/'); sys.path.append(yellow_path + 'buteo/raster/'); sys.path.append(yellow_path + 'buteo/convolutions/')
from buteo.raster.io import *
from scipy import ndimage

#%%

km10 = gpd.read_file(r"\\niras.int\root\PROJ\10\415\217\20_Aflevering/raekkefoelge.gpkg")

############################################################################################

tiles_list = km10[km10['lev_blok'] == 1]['tilename'].tolist()
base_path = '//pc116900/S Drone div/STENDIGER'


dtm_vrt_list = []
hat_vrt_list = []
sobel_vrt_list = []

#
for tile in tiles_list:

    dsm_folder = '{}/DSM/{}_TIF_UTM32-ETRS89/'.format(base_path, tile.replace('10km', 'DSM'))
    dtm_folder = '{}/DTM/{}_TIF_UTM32-ETRS89/'.format(base_path, tile.replace('10km', 'DTM'))
    hat_folder = dtm_folder.replace('DTM', 'HAT')
    sobel_folder = dtm_folder.replace('DTM', 'SOBEL')
    
    # print(hat_folder)
    # print(sobel_folder)
    if not os.path.exists(hat_folder):
        os.makedirs(hat_folder, exist_ok = False)
    if not os.path.exists(sobel_folder):
        os.makedirs(sobel_folder, exist_ok = False)

    dsm_tifs = glob.glob(dsm_folder + '*.tif')    
    dtm_tifs = glob.glob(dtm_folder + '*.tif')

    dsm_tifs.sort()
    dtm_tifs.sort()
    # for dtm, dsm in zip(dtm_tifs, dsm_tifs):
    #     if not (os.path.basename(dtm) == os.path.basename(dsm).replace('DSM', 'DTM')):
    #         ### this exception means that the contents of the folders do not match
    #         raise Exception
    # print('all tiles are present')
    for dsm, dtm in zip(dsm_tifs, dtm_tifs):

        dsm_raster = gdal.Open(dsm)
        dsm_bandarr = dsm_raster.GetRasterBand(1).ReadAsArray()
        dsm_npy = np.array(dsm_bandarr)

        dtm_raster = gdal.Open(dtm)
        dtm_bandarr = dtm_raster.GetRasterBand(1).ReadAsArray()
        dtm_npy = np.array(dtm_bandarr)


        sobel_npy = zobel_filter(
                dtm_npy, size=[5, 5], normalised_sobel=False, gaussian_preprocess=False
            )

        # sobel_npy = ndimage.sobel(dtm_npy, axis=-1, mode='constant', cval=0)

        hat_npy = dsm_npy - dtm_npy

        sobel_path = dtm.replace('DTM', 'SOBEL')
        hat_path = dtm.replace('DTM', 'HAT')

        array_to_raster(sobel_npy, reference=dtm, out_path=sobel_path, creation_options=["COMPRESS=LZW"])
        array_to_raster(hat_npy, reference=dtm, out_path=hat_path, creation_options=["COMPRESS=LZW"])
        
        dtm_vrt_list.append(dtm)
        hat_vrt_list.append(hat_path)
        sobel_vrt_list.append(sobel_path)
        
        print('done with', dtm)

#%%

for dtm, sobel, hat in zip(dtm_vrt_list, sobel_vrt_list, hat_vrt_list,):
    if not (os.path.basename(dtm) == os.path.basename(hat).replace('HAT', 'DTM') == os.path.basename(sobel).replace('SOBEL', 'DTM')):
        ### this exception means that the contents of the folders do not match
        raise Exception("catastrphic error")

    t = os.path.basename(dtm).replace('DTM_', '').replace('.tif', '.vrt')
    vrt = '//pc116900/S Drone div/STENDIGER/vrts/' + t
    print(vrt)  

    gdal.BuildVRT(vrt, [dtm, hat, sobel], options=gdal.BuildVRTOptions(separate=True))

#%%
vrt_dir = '//pc116900/S Drone div/STENDIGER/vrts/'

vrts = glob.glob(vrt_dir + '*.vrt')

ds = gdal.BuildVRT('//pc116900/S Drone div/STENDIGER/vrts/merged.vrt', vrts)
ds.FlushCache()


#%%

# ### run this if the tiles have already been transferred
# import glob
# dest_dir = 'V:/2022-03-31_Stendiger_EZRA/data/'
# dtm_tifs = glob.glob('{}dtm/*.tif'.format(dest_dir))
# dsm_tifs = glob.glob('{}dsm/*.tif'.format(dest_dir))

# dtm_tifs.sort()
# dsm_tifs.sort()

# for dtm, dsm in zip(dtm_tifs, dsm_tifs):
#     if not (os.path.basename(dtm) == os.path.basename(dsm).replace('DSM', 'DTM')):
#         ### this exception means that the contents of the folders do not match
#         raise Exception
    
# print('all tiles are present')

# #%%
# #%%
# ############################################################################################
# ##############CREATE HAT AND SOBEL FILTERS####################
# #######################################################################
# from osgeo import ogr, gdal
# import numpy as np
# from zobel_filter import zobel_filter

# yellow_path = "V:/2022-03-31_Stendiger_EZRA/buteo"
# import sys; sys.path.append(yellow_path); sys.path.append(yellow_path + 'buteo/'); sys.path.append(yellow_path + 'buteo/machine_learning/'); sys.path.append(yellow_path + 'buteo/filters/'); sys.path.append(yellow_path + 'buteo/raster/'); sys.path.append(yellow_path + 'buteo/convolutions/')
# from buteo.raster.io import *
# from scipy import ndimage

# for dsm, dtm in zip(dsm_tifs, dtm_tifs):

#     dsm_raster = gdal.Open(dsm)
#     dsm_bandarr = dsm_raster.GetRasterBand(1).ReadAsArray()
#     dsm_npy = np.array(dsm_bandarr)

#     dtm_raster = gdal.Open(dtm)
#     dtm_bandarr = dtm_raster.GetRasterBand(1).ReadAsArray()
#     dtm_npy = np.array(dtm_bandarr)


#     # sobel_npy = zobel_filter(
#     #         dtm_npy, size=[5, 5], normalised_sobel=False, gaussian_preprocess=False
#     #     )
#     hat_npy = dsm_npy - dtm_npy
#     sobel_npy = ndimage.sobel(dtm_npy, axis=-1, mode='constant', cval=0)

#     sobel_path = dtm.replace('DTM', 'SOBEL1').replace('dtm', 'sobel')
#     hat_path = dtm.replace('DTM', 'HAT').replace('dtm', 'hat')

#     array_to_raster(sobel_npy, reference=dtm, out_path=sobel_path, creation_options=["COMPRESS=LZW"])
#     array_to_raster(hat_npy, reference=dtm, out_path=hat_path, creation_options=["COMPRESS=LZW"])
#     print('done with', dtm)
# #%%









# # # create vrt's
# # 
# #%%
# ### run this if the tiles have already been transferred and hat and sobel have been made
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

# # %%


