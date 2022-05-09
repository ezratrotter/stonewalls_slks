#%%
# import subprocess
# import glob
# import os
# import Scripts.gdal_calc

# """
# V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dsm
# DSM_1km_6052_660.tif
# DTM_1km_6049_685.tif
# """
# tiles = []
# with open("missing_tiles.txt", "r") as file:
#     for tile in file:
#         tiles.append(tile.strip())

# tiles_dtm = []
# for tile in tiles:
#     tile_dtm = 'DTM_' + tile + ".tif"
#     tiles_dtm.append(tile_dtm)
# #%%
# dtm_dir = 'V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dtm/'
# dsm_dir = 'V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dsm/'
# hat_dir = 'V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/hat/'

# dtm_list = glob.glob(r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\dem\dtm\*.tif')
# dsm_list = glob.glob(r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\dem\dsm\*.tif')


# if len(dtm_list) != len(dsm_list):
#     print(len(dtm_list), len(dsm_list))
#     print('DTM and DSM files do not have the same number of files')
   
# # tile_list = [x.split('\\')[-1] for x in tiles]

# #%%
# count = 0
# for t in tiles_dtm:
#     # count += 1
#     # if count ==5: break
#     dtm = dtm_dir + t
#     dsm = dsm_dir + t.replace('DTM', 'DSM')
#     hat_file = hat_dir + t.replace('DTM', 'HAT')
#     if os.path.exists(hat_file):

#         continue

#     try:

#         calccommand = f'gdal_calc -A {dsm} -B {dtm} --outfile={hat_file} --co="COMPRESS=LZW" --NoDataValue=-9999 --type=Float32 --calc=A-B'

#         subprocess.call(calccommand, shell=True)

#         print("HAT file created: ", hat_file)

#     except:

#         print("FAILED: ", hat_file)

# print("checked")

# %%
import Scripts.gdal_calc
# %%
import subprocess

import glob

import os



"""

V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dsm

DSM_1km_6052_660.tif

DTM_1km_6049_685.tif

"""



if __name__ == "__main__":

    ###change this for missing tiles only --> adapt to the tiles you want to process
    tiles = []
    with open("missing_tiles.txt", "r") as file:
        for tile in file:
            tiles.append(tile.strip())

    tiles_dtm = []
    for tile in tiles:
        tile_dtm = 'DTM_' + tile + ".tif"
        tiles_dtm.append(tile_dtm)
    ###
    dtm_dir = "V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dtm/"

    dsm_dir = "V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dsm/"

    hat_dir = "V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/hat/"



    dtm_list = glob.glob(

        r"V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\dem\dtm\*.tif"

    )

    dsm_list = glob.glob(

        r"V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\dem\dsm\*.tif"

    )



    if len(dtm_list) != len(dsm_list):

        print(len(dtm_list), len(dsm_list))

        print("DTM and DSM files do not have the same number of files")



    # tile_list = [x.split("\\")[-1] for x in dtm_list]

    count = 0

    for t in tiles_dtm:

        dtm = dtm_dir + t

        dsm = dsm_dir + t.replace("DTM", "DSM")

        hat_file = hat_dir + t.replace("DTM", "HAT")

        if os.path.exists(hat_file):

            continue

        try:


            calccommand = f'gdal_calc -A {dsm} -B {dtm} --outfile={hat_file} --co="COMPRESS=LZW" --NoDataValue=-9999 --type=Float32 --calc=A-B \n'

            with open("C:/Users/AFER/Desktop/commands.txt","a") as file1:
                file1.writelines(calccommand)
                file1.close()

            # subprocess.call(calccommand, shell=True)

            print("HAT file created: ", hat_file)

        except:

            print("FAILED: ", hat_file)

    print("checked")