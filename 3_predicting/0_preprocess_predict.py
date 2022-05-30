# %%
import os
import numpy as np
from osgeo import gdal
import glob
import geopandas as gpd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# out_dir = 'V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/vrt1km_pilotarea/'
# dtm_dir = 'V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dtm/'
# hat_dir = 'V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/hat/'
# sobel_dir = 'V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/sobel/'

# dtm_tiles = glob.glob(dtm_dir + '/*.tif')
# hat_tiles = glob.glob(hat_dir + '/*.tif')
# sobel_tiles = glob.glob(sobel_dir + '/*.tif')

# %%

# dtm_tiles_test = dtm_tiles[0:3]

# tile_names = [x.split('\\')[1].split('DTM_')[1].split('.')[0] for x in dtm_tiles]

# %% vrt of dtm, hat, sobel for all pilot area


# arrays_pilot_path = glob.glob("V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/npz_tiles_data/pilot_area/*.npz")
# data_tiles = [x.split('\\')[1].split('.')[0] for x in arrays_pilot_path]

# %%
# build vrt of each layer for the all 1km tiles of pilot area
# dtm_pilot_tiles = [x for x in dtm_tiles if x.split('\\')[1].split('DTM_')[1].split('.')[0] in data_tiles]
# dtm_vrt = out_dir + "dtm_pilotarea.vrt"
# gdal.BuildVRT(dtm_vrt, dtm_pilot_tiles, separate=True)

# hat_pilot_tiles = [x for x in hat_tiles if x.split('\\')[1].split('HAT_')[1].split('.')[0] in data_tiles]
# hat_vrt = out_dir + "hat_pilotarea.vrt"
# gdal.BuildVRT(hat_vrt, hat_pilot_tiles, separate=True)

# sobel_pilot_tiles = [x for x in sobel_tiles if x.split('\\')[1].split('SOBELDTM_')[1].split('.')[0] in data_tiles]
# sobel_vrt = out_dir + "sobel_pilotarea.vrt"
# gdal.BuildVRT(sobel_vrt, sobel_pilot_tiles, separate=True)

# %% vrt of dtm, hat, sobel for each 1km tile for pilot area

# missing_tiles = []
# with open("missing_tiles.txt", "r") as file:
#     for tile in file:
#         missing_tiles.append(tile.strip())

# %%
count = 0
for x in data_tiles:
    image_list = []

    count += 1
    # if count ==5: break

    dtm_path = dtm_dir + 'DTM_' + x + '.tif'
    hat_path = hat_dir + 'HAT_' + x + '.tif'
    sobel_path = sobel_dir + 'SOBELDTM_' + x + '.tif'

    if not (os.path.exists(dtm_path) and os.path.exists(hat_path) and os.path.exists(sobel_path)):
        print('DISASTER')
        break

    print(x)
    print(os.path.exists(dtm_path), os.path.exists(
        hat_path), os.path.exists(sobel_path))

    image_list.append(dtm_path)
    image_list.append(hat_path)
    image_list.append(sobel_path)
    print("image_list len:", len(image_list))
    vrt = out_dir + x + '.vrt'
    # print(image_list)
    # print(vrt)
    print(f"{count} ---")

    gdal.BuildVRT(vrt, image_list, options=gdal.BuildVRTOptions(separate=True))


print("all vrt'd!")

# %%

# vrt one 10km tile for test prediction

# km1_pred = gpd.read_file(r'V:\2022-03-31_Stendiger_EZRA\stonewalls_slks\data\pilot_training_data\dki_1km_10km_614_67.gpkg')

# tile_names = list(km1_pred['tilename'])

# dtm_pilot_tiles = [x for x in dtm_tiles if x.split('\\')[1].split('DTM_')[1].split('.')[0] in tile_names]

# print("dtm_tiles: ", len(dtm_pilot_tiles))
# with open("dtm_10km_614_67_1kmlist.txt", "w+") as file:
#     for tile in dtm_pilot_tiles:
#         file.write(tile + '\n')

# hat_pilot_tiles = [x for x in hat_tiles if x.split('\\')[1].split('HAT_')[1].split('.')[0] in tile_names]
# print("hat_tiles: ", len(hat_pilot_tiles))
# hat_list = []

# with open("hat_10km_614_67_1kmlist.txt", "w+") as file:
#     for tile in hat_pilot_tiles:
#         file.write(tile + '\n')

# sobel_pilot_tiles = [x for x in sobel_tiles if x.split('\\')[1].split('SOBELDTM_')[1].split('.')[0] in tile_names]

# sobel_list = []
# print("sobel_tiles: ", len(sobel_pilot_tiles))
# with open("sobel_10km_614_67_1kmlist.txt", "w+") as file:
#     for tile in sobel_pilot_tiles:
#         file.write(tile + '\n')


# %%
# vrt all into 10km tiles and stack them
in_dir = '//pc116900/S Drone div/STENDIGER/vrts/'
vrt_1km = glob.glob('//pc116900/S Drone div/STENDIGER/vrts/*.vrt')
vrt_10km_dir = 'R:/PROJ/10/415/217/20_Aflevering/'

print(len(vrt_1km))
# %%
km1_pilot = gpd.read_file(
    'R:/PROJ/10/415/217/20_Aflevering/geometries/dki_1km_lev1_land.gpkg')
km10_pilot = gpd.read_file('R:/PROJ/10/415/217/20_Aflevering/lev_1_10km.gpkg')


# %%
dict_tiles = {}

for k, row in km1_pilot.iterrows():
    km_10 = row['dki_10km']
    km_1 = row['tilename']
    if km_10 not in dict_tiles:
        dict_tiles[km_10] = []

    dict_tiles[km_10].append(km_1)

# %%

info = []
for k, row in km10_pilot.iterrows():
    name = row['tilename']
    xmin = row['n']
    ymin = row['e']
    tilesize = row['tilesize']

    entry = [name, xmin, ymin, xmin + tilesize, ymin + tilesize]
    info.append(entry)

# %%

vrt_1km_name = [x.split('\\')[1].split('.')[0] for x in vrt_1km]

# %%
# km1_tiles_list = []
# km1_missing_list = []
# x is the name of the 10km tile
for i, x in enumerate(dict_tiles.keys()):

    tilename, xmin, ymin, xmax, ymax = info[i]
    print(tilename, xmin, ymin, xmax, ymax)

    print(
        f"{x} tile with {tilename} bounds with {len(dict_tiles[x])} 1km tiles")
    vrt = vrt_10km_dir + x + '.vrt'
    print(vrt)
    src = [in_dir + y + '.vrt' for y in dict_tiles[x]]
    # for t in dict_tiles[x]:
    #     if not os.path.isfile(out_dir + t +'.vrt'):
    #         km1_missing_list.append(t)
    #     else: continue
    # print(src)
    gdal.BuildVRT(vrt, src, options=gdal.BuildVRTOptions(
        outputBounds=(xmin, ymin, xmax, ymax)))

print("all tiles vrt'd!")
# %%

# with open("km1_missing_tiles.txt", "w+") as file:
#     for tile in km1_missing_list:
#         file.write(tile + '\n')


# %%

# dtm_pilot_tiles = [x for x in dtm_tiles if x.split('\\')[1].split('DTM_')[1].split('.')[0] in tile_names]

# print("dtm_tiles: ", len(dtm_pilot_tiles))
# with open("dtm_10km_614_67_1kmlist.txt", "w+") as file:
#     for tile in dtm_pilot_tiles:
#         file.write(tile + '\n')

# hat_pilot_tiles = [x for x in hat_tiles if x.split('\\')[1].split('HAT_')[1].split('.')[0] in tile_names]
# print("hat_tiles: ", len(hat_pilot_tiles))
# hat_list = []

# with open("hat_10km_614_67_1kmlist.txt", "w+") as file:
#     for tile in hat_pilot_tiles:
#         file.write(tile + '\n')

# sobel_pilot_tiles = [x for x in sobel_tiles if x.split('\\')[1].split('SOBELDTM_')[1].split('.')[0] in tile_names]

# sobel_list = []
# print("sobel_tiles: ", len(sobel_pilot_tiles))
# with open("sobel_10km_614_67_1kmlist.txt", "w+") as file:
#     for tile in sobel_pilot_tiles:
#         file.write(tile + '\n')


# %%


# read vrt for training + pilot

# get vector of our tile / list of vectors of tiles

# clip to that tile/ those tiles

# predict

# put back to gether
