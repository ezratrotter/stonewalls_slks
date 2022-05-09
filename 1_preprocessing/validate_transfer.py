
import geopandas as gpd
import os
import glob
import numpy as np
import shapely
from osgeo import ogr, gdal
import shutil

def validate_tile_transfer(tile_type, source_dir, dest_dir, tile_list):

    if tile_type != 'DTM' and tile_type != 'DSM':
        print('tile_type: ', tile_type)
        raise ValueError('tile_type must be either DTM or DSM')
    if not os.path.exists(source_dir):
        raise ValueError('source_dir doesnt exist')
    if not os.path.exists(dest_dir):
        raise ValueError('dest_dir doesnt exist')
    if type(tile_list) != list:
        raise ValueError('tile_list must be a list')
    if len(tile_list) == 0:
        raise ValueError('tile_list must not be empty')

    print('validating files...', 'count: ', len(tile_list))
    for tile in tile_list:
        
        source_tif = os.path.join(DTM_source_dir, tile_type + '_' + tile +'.tif' )
        dest_tif = os.path.join(DTM_destination_dir, tile_type + '_' + tile +'.tif' )

        if not os.path.exists(source_tif):
            print('file didnt exist: ', source_tif)
        if not os.path.exists(dest_tif):
            print('file did not transfer: ', dest_tif)
        
    print('finished')

if __name__ == '__main__':

    km10training = gpd.read_file(r'V:\2022-03-31_Stendiger_EZRA\stonewalls_slks\data\pilot_training_data\training_10km_grid.gpkg')
    km10training['geometry'] = km10training.apply(lambda x: x['geometry'].buffer(1), axis=1)
    km1 = gpd.read_file(r'V:\2022-03-31_Stendiger_EZRA\stonewalls_slks\data\grids\dki_1km.gpkg')
    tile_list = gpd.overlay(km1, km10training, how='intersection')['dki_1km'].unique().tolist()



    DTM_source_dir = 'M:/Ekstern_datasamling/Danmark/Frie_DATA/DTM_2019/DTM_Grid_1km_TIFF/'
    DSM_source_dir = 'M:/Ekstern_datasamling/Danmark/Frie_DATA/DTM_2019/DSM_Grid_1km_TIFF/'

    DTM_destination_dir = 'V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dtm/'
    DSM_destination_dir = 'V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/dsm/'

    validate_tile_transfer('DTM', DTM_source_dir, DTM_destination_dir, tile_list)
    validate_tile_transfer('DSM', DSM_source_dir, DSM_destination_dir, tile_list)