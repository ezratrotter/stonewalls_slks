
#%%
import geopandas as gpd
import time
import sys; sys.path.append('D:/stonewalls_slks/lib')
from datetime import datetime
from osgeo import ogr, gdal

from core import create_profiles
#%%
#	1. Select 1 10km tile.
km10 = gpd.read_file(r'D:\stonewalls_slks\data\pilot_training_data\training_10km_grid.gpkg')
walls = gpd.read_file('D:/stonewalls_slks/data/stonewalls/denmark')
myTileGeom = km10.loc[0]['geometry']

#%%
#	2. Buffer 10m with x meters.
myTileGeomBuffered = myTileGeom.buffer(11.0)
walls_clip = gpd.clip(walls, myTileGeomBuffered)

#%%
#	3. Clip stonewalls to (2)

#   4. Extract DTM raster by clipping (2) to the .vrt DTM. buteo.raster.clip
vrt = 'M:/Ekstern_datasamling/Danmark/Frie_DATA/DTM_2019/DTM_Grid_1km_TIFF/_merged.vrt'



output_folder = 'D:/OUT'
distance = 10
date = datetime.now().strftime("%m-%d-%Y")


gdf = walls_clip.to_crs(epsg=25832)

# activate and replace first arg of create_profiles for smaller subset
# sub_gdf = gdf[:50]

profiles, walls = create_profiles(gdf, vrt, subwall_distance=distance)

profiles_out = '{}profiles_{}_{}.geojson'.format(output_folder, str(distance)+'m', date)
walls_out = '{}walls_{}_{}.geojson'.format(output_folder, str(distance)+'m', date)

profiles.to_file(profiles_out, driver="GeoJSON")
walls.to_file(walls_out, driver="GeoJSON")

finish = time.time()

print("Time Taken is {0}s".format(finish - start))
print(output_folder)

#%%


# %%
