#%%
import subprocess
import glob
import os
"""
V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dsm
DSM_1km_6052_660.tif
DTM_1km_6049_685.tif
"""
dtm_dir = 'V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dtm/'
dsm_dir = 'V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dsm/'
hat_dir = 'V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/hat/'

dtm_list = glob.glob(r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\dem\dtm\*.tif')
dsm_list = glob.glob(r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\dem\dsm\*.tif')


if len(dtm_list) != len(dsm_list):
    print(len(dtm_list), len(dsm_list))
    print('DTM and DSM files do not have the same number of files')
   

tile_list = [x.split('\\')[-1] for x in dtm_list]
count = 0
for t in tile_list:
    # count += 1
    # if count ==5: break
    dtm = dtm_dir + t
    dsm = dsm_dir + t.replace('DTM', 'DSM')
    hat_file = hat_dir + t.replace('DTM', 'HAT')
    if os.path.exists(hat_file):
        print('HAT file already exists: ', hat_file)
        continue
    subprocess_string ="gdal_calc.py -A {} -B {} --outfile={} --calc='A-B'".format(dtm, dsm, hat_file)
    print('HAT file created: ', hat_file)
        
print('checked')

# %%
