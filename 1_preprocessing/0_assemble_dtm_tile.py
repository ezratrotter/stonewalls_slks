#%%
import geopandas as gpd
import os
import glob
import numpy as np
import shapely
from osgeo import ogr, gdal
#	1. Select 1 10km tile.
km10 = gpd.read_file(r'D:\stonewalls_slks\data\pilot_training_data\training_10km_grid.gpkg')
walls = gpd.read_file('D:/stonewalls_slks/data/stonewalls/denmark')
myTileGeom = km10.loc[0]['geometry']

#%%
#	2. Buffer 10m with x meters.
myTileGeomBuffered = myTileGeom.buffer(11.0)

#%%
#	3. Clip stonewalls to (2)
walls_clip = gpd.clip(walls, myTileGeomBuffered)
#%%

#   4. Extract DTM raster by clipping (2) to the .vrt DTM. buteo.raster.clip
vrt = 'M:/Ekstern_datasamling/Danmark/Frie_DATA/DTM_2019/DTM_Grid_1km_TIFF/_merged.vrt'


# walls_clip.plot()
# walls.plot()
#	4. Extract DTM raster by clipping (2) to the .vrt DTM. buteo.raster.clip

#	5. Done.




#%%
import geopandas as gpd
import os
import glob
import numpy as np
import shapely
from osgeo import ogr, gdal

#%%
km10training = gpd.read_file(r'V:\2022-03-31_Stendiger_EZRA\stonewalls_slks\data\pilot_training_data\training_10km_grid.gpkg')
km1 = gpd.read_file(r'V:\2022-03-31_Stendiger_EZRA\stonewalls_slks\data\grids\dki_1km.gpkg')
# km10 = gpd.read_file(r'V:\2022-03-31_Stendiger_EZRA\stonewalls_slks\data\grids\dki_10km.gpkg')


km10training['geometry'] = km10training.apply(lambda x: x['geometry'].buffer(1), axis=1)

tile_list = gpd.overlay(km1, km10training, how='intersection')['dki_1km'].tolist()

#%%

missing_list = r"V:\2022-03-31_Stendiger_EZRA\code\km1_missing_tiles.txt"

tile_list = []

### adapted this to get the tilis that were missing --> adapt to the data you want to extract from
with open(missing_list, "r") as file:
    for tile in file:
        tile_list.append(tile.strip())

print(len(tile_list), 'tiles....')

#%%
# len(gpd.overlay(km1, km10training, how='intersection')['dki_1km'].unique().tolist())

#%%
import shutil
import os
DTM_source_dir = 'M:/Ekstern_datasamling/Danmark/Frie_DATA/DTM_2019/DTM_Grid_1km_TIFF/'
DSM_source_dir = 'M:/Ekstern_datasamling/Danmark/Frie_DATA/DTM_2019/DSM_Grid_1km_TIFF/'

DTM_destination_dir = 'V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dtm/'
DSM_destination_dir = 'V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dsm/'


print('moving files...', 'count: ', len(tile_list))
for tile in tile_list:
    
    dtm_source_tif = os.path.join(DTM_source_dir, 'DTM_'+ tile +'.tif' )
    dsm_source_tif = os.path.join(DSM_source_dir, 'DSM_'+ tile +'.tif' )

    dtm_dest_tif = os.path.join(DTM_destination_dir, 'DTM_'+ tile +'.tif' )
    dsm_dest_tif = os.path.join(DSM_destination_dir, 'DSM_'+ tile +'.tif' )
    
    if not os.path.exists(dtm_source_tif):
        print("tile does not exist: ", tile)
    else:
        if not os.path.exists(dtm_dest_tif):
            shutil.copy(dtm_source_tif, dtm_dest_tif)
            print('DTM file didnt exist: ', dtm_source_tif)
    if not os.path.exists(dsm_source_tif):
        print("tile does not exist: ", tile)
    else:
        if not os.path.isfile(dsm_dest_tif):
            shutil.copy(dsm_source_tif, dsm_dest_tif)
            print('DSM file didnt exist: ', dsm_source_tif)
    # print('moving: ',tile, '...')
print('finished')
