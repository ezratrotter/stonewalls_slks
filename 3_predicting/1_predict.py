#%%
import os
from glob import glob
from osgeo import gdal, ogr, gdalconst
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import segmentation_models as sm
import numpy as np
from tensorflow.python.client import device_lib
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# sm.set_framework('tf.keras')

# sm.framework()

# tf.config.list_physical_devices('GPU')

# device_lib.list_local_devices()

####functions###
def read_raster(model_raster_fname):
    model_dataset = gdal.Open(model_raster_fname)
    return model_dataset.ReadAsArray()


def tile_images(images, tile_sz, overlap_multiplier=1):
    """
    Input: List of image numpy arrays
    
    Output: List of tiles and the corresponding index ranges
    """
    im_sz = np.asarray((np.shape(images[0])[0:2]))

    tile_sz = int(tile_sz)
    Ny = int(np.ceil(im_sz[0]/tile_sz))
    Nx = int(np.ceil(im_sz[1]/tile_sz))

    rng_y = np.linspace(0, im_sz[0]-tile_sz, endpoint=True,
                        num=Ny*overlap_multiplier, retstep=False, dtype='int')
    rng_x = np.linspace(0, im_sz[1]-tile_sz, endpoint=True,
                        num=Nx*overlap_multiplier, retstep=False, dtype='int')

    lst = []

    for im in images:
        # Add extra dimension for one channel img
        if len(np.shape(im)) == 2:
            im = im[..., None]

        tiles = []

        for r_i in rng_y:
            for r_j in rng_x:
                tiles.append(im[r_i:r_i+tile_sz, r_j:r_j+tile_sz, :])

        tiles = np.stack(tiles, axis=0)

        lst.append(tiles)
    
    return (lst, rng_x, rng_y, im_sz)


def vrt_tile_generator(vrt_filename, tile_sz, buffer):

    ds = gdal.Open(vrt_filename)

    res_x = ds.RasterXSize
    res_y = ds.RasterYSize

    # buffer = 64
    # tile_sz = 2500

    n_x = int(res_x/tile_sz)
    n_y = int(res_y/tile_sz)

    rng_x = np.linspace(0, res_x-tile_sz, endpoint=True,
                        num=n_x, retstep=False, dtype='int')
    rng_y = np.linspace(0, res_y-tile_sz, endpoint=True,
                        num=n_y, retstep=False, dtype='int')
    for r_i in rng_y:
        for r_j in rng_x:
            start_x = r_j if r_j == 0 else r_j-buffer
            end_x = r_j+tile_sz if r_j == rng_x[-1] else r_j+tile_sz+buffer
            sz_x = end_x - start_x

            start_y = r_i if r_i == 0 else r_i-buffer
            end_y = r_i+tile_sz if r_i == rng_y[-1] else r_i+tile_sz+buffer
            sz_y = end_y - start_y

            im = np.moveaxis(ds.ReadAsArray(int(start_x), int(start_y), int(sz_x), int(sz_y)), 0, -1)
            yield im

def vrt_tile_ranges(vrt_filename, tile_sz, buffer):

    ds = gdal.Open(vrt_filename)

    res_x = ds.RasterXSize
    res_y = ds.RasterYSize

    # buffer = 64
    # tile_sz = 2500

    n_x = int(res_x/tile_sz)
    n_y = int(res_y/tile_sz)

    rng_x = np.linspace(0, res_x-tile_sz, endpoint=True,
                        num=n_x, retstep=False, dtype='int')
    rng_y = np.linspace(0, res_y-tile_sz, endpoint=True,
                        num=n_y, retstep=False, dtype='int')

    ranges = []

    for r_i in rng_y:
        for r_j in rng_x:
            start_x = 0 if r_j == 0 else buffer
            end_x = 0 if r_j == rng_x[-1] else buffer
            sz_x = end_x - start_x

            start_y = 0 if r_i == 0 else buffer
            end_y = 0 if r_i == rng_y[-1] else buffer
            sz_y = end_y - start_y

            ranges.append([r_j, r_i, start_x, end_x, start_y, end_y])
    # returns array that....
    return ranges


def patch_generator(vrt_gen, patch_sz):
    for tile in vrt_gen:
        patches, rng_x, rng_y, im_sz = tile_images([tile], patch_sz, 2)
        yield patches[0], rng_x, rng_y, im_sz

def rasterize_shp_like(shapefiles, model_raster_fname, output_fname, options=[], nodata_val=0, verbose=False):
    """
    Given a list of shapefile names, rasterizes them so they have
    the exact same extent as the given model_raster.

    `options` should be a list that will be passed to GDALRasterizeLayers
    """

    model_dataset = gdal.Open(model_raster_fname)
    x_res = model_dataset.RasterXSize
    y_res = model_dataset.RasterYSize

    srs = model_dataset.GetProjection()
    geo_transform = model_dataset.GetGeoTransform()

    labels = []
    for shpfile in shapefiles:
        if verbose:
            print(shpfile)

        # Open shapefile dataset
        shape_dataset = ogr.Open(shpfile)
        shape_layer = shape_dataset.GetLayer()

        # Create new dataset in memory
        mem_drv = gdal.GetDriverByName('MEM')
        mem_raster = mem_drv.Create(
            '',
            x_res,
            y_res,
            1,
            gdal.GDT_Float32
        )
        mem_raster.SetProjection(srs)
        mem_raster.SetGeoTransform(geo_transform)
        mem_band = mem_raster.GetRasterBand(1)
        mem_band.Fill(nodata_val)
        mem_band.SetNoDataValue(nodata_val)

        # Rasterize shapefile into mem ds
        err = gdal.RasterizeLayer(
            mem_raster,
            [1],
            shape_layer,
            None,
            None,
            [1],
            options
        )

        assert err == gdal.CE_None

        # Store rasterized data in array
        labels.append(mem_raster.ReadAsArray())

    # Save labels in tif
    x_min = geo_transform[0]
    y_min = geo_transform[3]

    pixel_width = geo_transform[1]

    N = len(labels)

    target_ds = gdal.GetDriverByName('GTiff').Create(
        output_fname, x_res, y_res, N, gdal.GDT_Float32)
    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, -pixel_width))
    target_ds.SetProjection(srs)

    for idx, image_data in enumerate(labels):
        print(idx, image_data.shape)
        b = idx+1
        band = target_ds.GetRasterBand(b)
        NoData_value = -1
        band.SetNoDataValue(NoData_value)
        band.FlushCache()

        target_ds.GetRasterBand(b).WriteArray(image_data)

def restore_image_from_tiles(tiles,rng_x,rng_y, dims, n_classes):
    """
    Restore full image from numpy array of tiles generated using tile_images
    """
    tile_sz = tiles.shape[1:3]

    # print(tile_sz)
    
    unq = [v for v in range(n_classes)]

    im = np.zeros( (*dims, len(unq)), dtype=tiles.dtype)

    print(im.shape)
    
    idx = 0

    for r_i in rng_y:
        for r_j in rng_x:


            for u in unq:
                im[r_i:r_i+int(tile_sz[0]) , r_j:r_j+int(tile_sz[1]), u] += (tiles[idx, :,:,0] == u).astype(tiles.dtype)

            idx = idx + 1
    
    return np.argmax(im,axis=-1)

def write_data_as_raster_list(image_data_list, fn_raster, fn_output):

    N = len(image_data_list)

    data = gdal.Open(fn_raster, gdalconst.GA_ReadOnly)
    geo_transform = data.GetGeoTransform()

    x_min = geo_transform[0]
    y_min = geo_transform[3]

    x_res = data.RasterXSize
    y_res = data.RasterYSize

    srs = data.GetProjection()

    pixel_width = geo_transform[1]

    target_ds = gdal.GetDriverByName('GTiff').Create(fn_output, x_res, y_res, N, gdal.GDT_Byte, ['COMPRESS=LZW'])
    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, -pixel_width))
    target_ds.SetProjection(srs)

    for idx,image_data in enumerate(image_data_list):
        print(f"final tile: {idx}, {image_data.shape}")
        band = target_ds.GetRasterBand(1)
        NoData_value = -1
        band.SetNoDataValue(NoData_value)
        band.FlushCache()
        #gdal.RasterizeLayer(target_ds, [1], mb_l,burn_values=[255])
        target_ds.GetRasterBand(idx+1).WriteArray(image_data)
    
    target_ds = None

def clip_tile(y_tile, range_tile):
    #print(y_tile.shape)   

    (y_dim, x_dim) = y_tile.shape
    # print(x_dim, y_dim)

    (rj, ri, left_clip, right_clip, top_clip, bottom_clip) = range_tile
    #(rj, ri, right_clip,left_clip,  bottom_clip, top_clip) = range_tile
    # print(left_clip, right_clip, top_clip, bottom_clip)
    # print(rj, ri)

    y_tile = y_tile[top_clip:y_dim - bottom_clip, left_clip:x_dim - right_clip]

    print(y_tile.shape)

    full[ri:ri+5000, rj:rj+5000] = y_tile
    # print('--')
    
    return y_tile    

###code###
#%%
###upload model
model_path = "V:/2022-03-31_Stendiger_EZRA/code/logs/"

custom_objects = {
    'binary_focal_loss_plus_jaccard_loss': sm.losses.binary_focal_jaccard_loss,
    'iou_score': sm.metrics.iou_score
}

model = load_model(model_path + 'model.25-0.832-0.190-0.984.h5', custom_objects=custom_objects)

#%%
###generate tiles from 10km and patches for prediction
tiles_path = glob('V:/2022-03-31_Stendiger_EZRA/training_data/initial_area/dem/vrt10km_pilotarea/*.vrt')
# tile = tiles_path + '/stack_10km_614_67.vrt'
out_dir = "R:/PROJ/10/415/217/20_Aflevering/pilot_area/"

import gc 
import keras
import keras.backend as K


for tile in tiles_path:
    if not os.path.exists(tile):
        print(f"{tile} does not exist!!")
        break

    if os.path.exists(out_dir + tile.split('\\')[1].split('.')[0] + ".tif"):
        print(f"skipping {tile}, already exists")
        continue

    ds = gdal.Open(tile)
    print(f"generating tiles and patches for {tile}...")
    vrt_gen = vrt_tile_generator(tile, 5000, 64)
    patch_gen = patch_generator(vrt_gen, 64)

    ##for each tile generated, predict for all patches
    print(f"predicting {tile}...")
    all_y = []
    for idx, (patches, rng_x, rng_y, im_sz) in enumerate(patch_gen):
        y = model.predict(patches, batch_size=512, verbose=1)
        K.clear_session()
        
        all_y.append( (y,rng_x, rng_y, im_sz))

        print(idx)

    all_y2 = [ ((v>0.1).astype(np.int32), rng_x, rng_y, im_sz) for (v, rng_x, rng_y, im_sz) in all_y]

    ###restore image tiles from patches
    y_tile = [restore_image_from_tiles(y2, rng_x, rng_y, im_sz, 2) for (y2, rng_x, rng_y, im_sz) in all_y2]

    ##check ranges (buffers), correct tiles size to original tiles (5000x5000), without the buffer
    ##result is array of arrays

    ranges = vrt_tile_ranges(tile, 5000, 64)

    result = []
    full = np.zeros((25000,25000))

    result = [] 
    for idx, _ in enumerate(y_tile):
        # print(idx)
        res = clip_tile(y_tile[idx], ranges[idx])
        result.append(res)

    print(f'done clipping for {tile}')
    # print([v.shape for v in result])

    ##write raster 10km tile from array
   

    #ref raster - vrt 
    # tile = r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\dem\vrt\km10\dtm_10km_614_67.vrt'

    out_tif = tile.split('\\')[1].split('.')[0] + ".tif"
    print(f"writing raster {out_tif}...")
    write_data_as_raster_list([full], tile, out_dir + out_tif)

    ds = None
    del full, y_tile, all_y, all_y2

    gc.collect()
    if not os.path.exists(out_dir + out_tif):
        print("no output created!")
        break


print("all tiles predicted and saved!")


# %%
