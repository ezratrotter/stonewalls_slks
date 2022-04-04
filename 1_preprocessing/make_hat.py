import subprocess
import glob
import os

"""
V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dsm
DSM_1km_6052_660.tif
DTM_1km_6049_685.tif
"""

if __name__ == "__main__":

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

    tile_list = [x.split("\\")[-1] for x in dtm_list]
    count = 0
    for t in tile_list:

        dtm = dtm_dir + t
        dsm = dsm_dir + t.replace("DTM", "DSM")
        hat_file = hat_dir + t.replace("DTM", "HAT")
        if os.path.exists(hat_file):
            continue
        try:
            calccommand = f'gdal_calc -A {dsm} -B {dtm} --outfile={hat_file} --co="COMPRESS=LZW" --NoDataValue=-9999 --type=Float32 --calc=A-B'
            subprocess.call(calccommand, shell=True)
            print("HAT file created: ", hat_file)
        except:
            print("FAILED: ", hat_file)
    print("checked")
