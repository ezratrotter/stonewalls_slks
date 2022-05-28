# %%
import scipy
import os
from glob import glob
from osgeo import gdal, ogr, gdalconst
import tensorflow as tf
from keras.layers import LayerNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import segmentation_models as sm
import numpy as np
from tensorflow.python.client import device_lib
from matplotlib import pyplot as plt
import gc
import keras.backend as K
import keras


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# # sm.set_framework('tf.keras')

# # sm.framework()

# tf.config.list_physical_devices('GPU')

# device_lib.list_local_devices()

####functions###


def read_raster(model_raster_fname):
    model_dataset = gdal.Open(model_raster_fname)
    return model_dataset.ReadAsArray()


def write_raster(image, fn_raster, fn_output):

    N = 1
    if len(image.shape) > 2:
        N = image.shape[-1]

    print(f'{N} classes')

    dataset = gdal.Open(fn_raster, gdalconst.GA_ReadOnly)

    x_min, pixel_width, _,  y_min, _, pixel_height = dataset.GetGeoTransform()

    x_res = dataset.RasterXSize
    y_res = dataset.RasterYSize

    srs = dataset.GetProjection()

    target_ds = gdal.GetDriverByName('GTiff').Create(fn_output, x_res, y_res, N, gdal.GDT_Float32, options=[
        'COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'])
    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, -pixel_width))
    target_ds.SetProjection(srs)

    NoData_value = -1
    if N > 1:
        for idx in range(N):
            print(idx)
            band = target_ds.GetRasterBand(idx+1)
            band.SetNoDataValue(NoData_value)
            band.FlushCache()
            band.WriteArray(image[:, :, idx])

    else:
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(NoData_value)
        band.FlushCache()
        band.WriteArray(image[:, :, 0])

    #target_ds.BuildOverviews("NEAREST", [2,4,8,16,32,64,128])
    target_ds = None


def read_from_vrt(ds, x, y, patch_size):
    im = ds.ReadAsArray(xoff=int(x), yoff=int(
        y), xsize=patch_size, ysize=patch_size)
    im = np.moveaxis(im, 0, -1).astype(np.float32)
    return im


def sigmoid_window(window_size, width, steepness, dim=2):

    x = np.linspace(-steepness, steepness, int(np.floor(window_size/width)))
    y = scipy.special.expit(x)

    diff = (window_size//2)-len(y)
    # pad with ones if len(y) < window_size/2
    if diff > 0:
        y = np.concatenate([y, np.ones(int(diff))])
    # trim array if len(y) > window_size/2
    elif diff < 0:
        print('woot')
        y = y[-(window_size//2):]

    # fill in missing one if window_size is odd
    fill = [] if len(y)*2 == window_size else np.ones(window_size-len(y)*2)

    # concat y
    y = np.concatenate([y, fill, y[::-1]])

    # return
    if dim == 1:
        return y

    if dim == 2:
        y = np.expand_dims(y, 1)
        y = y * y.T
        return y


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

            im = np.moveaxis(ds.ReadAsArray(int(start_x), int(
                start_y), int(sz_x), int(sz_y)), 0, -1)

            if np.sum(im) == 0:
                continue

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


def vrt_patch_generator(ds, rngx, rngy, patch_size):
    while(True):
        for x in rngx:
            for y in rngy:
                yield read_from_vrt(ds, x, y, patch_size)


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


def restore_image_from_tiles(tiles, rng_x, rng_y, dims, n_classes):
    """
    Restore full image from numpy array of tiles generated using tile_images
    """
    tile_sz = tiles.shape[1:3]

    # print(tile_sz)

    unq = [v for v in range(n_classes)]

    im = np.zeros((*dims, len(unq)), dtype=tiles.dtype)

    print(im.shape)

    idx = 0

    for r_i in rng_y:
        for r_j in rng_x:

            for u in unq:
                im[r_i:r_i+int(tile_sz[0]), r_j:r_j+int(tile_sz[1]),
                   u] += (tiles[idx, :, :, 0] == u).astype(tiles.dtype)

            idx = idx + 1

    return np.argmax(im, axis=-1)


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

    target_ds = gdal.GetDriverByName('GTiff').Create(
        fn_output, x_res, y_res, N, gdal.GDT_Byte, ['COMPRESS=LZW'])
    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, -pixel_width))
    target_ds.SetProjection(srs)

    for idx, image_data in enumerate(image_data_list):
        print(f"final tile: {idx}, {image_data.shape}")
        band = target_ds.GetRasterBand(1)
        NoData_value = -1
        band.SetNoDataValue(NoData_value)
        band.FlushCache()
        #gdal.RasterizeLayer(target_ds, [1], mb_l,burn_values=[255])
        target_ds.GetRasterBand(idx+1).WriteArray(image_data)

    target_ds = None


def clip_tile(y_tile, range_tile):
    # print(y_tile.shape)

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


def createBuffer(inputfn, outputBufferfn, bufferDist):
    inputds = ogr.Open(inputfn)
    inputlyr = inputds.GetLayer()

    shpdriver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(outputBufferfn):
        shpdriver.DeleteDataSource(outputBufferfn)
    outputBufferds = shpdriver.CreateDataSource(outputBufferfn)
    bufferlyr = outputBufferds.CreateLayer(
        outputBufferfn, geom_type=ogr.wkbPolygon)
    featureDefn = bufferlyr.GetLayerDefn()

    for feature in inputlyr:
        ingeom = feature.GetGeometryRef()
        geomBuffer = ingeom.Buffer(bufferDist)

        outFeature = ogr.Feature(featureDefn)
        outFeature.SetGeometry(geomBuffer)
        bufferlyr.CreateFeature(outFeature)
        outFeature = None

#%%
def smooth_vrt_predictions(model, vrt_filename, output_filename, options):
    print('test')
    if not os.path.isfile(vrt_filename):
        print(vrt_filename)
        raise Exception("VRT file does not exist")
    # Create tf.dataset from generator that iterates over vrt
    ds = gdal.Open(vrt_filename)

    w = ds.RasterXSize
    h = ds.RasterYSize

    patch_size = options['patch_size']
    nx = options['n_bands_x']
    nc = options['n_classes']
    overlap_multplier = options['overlap']
    batch_size = options['batch_size']
    steps_per_predict = options['steps_per_predict']

    rngx, stepx = np.linspace(
        0, w-patch_size, num=int(np.ceil(w*overlap_multplier/patch_size)), retstep=True, dtype=int)
    rngy, stepy = np.linspace(
        0, h-patch_size, num=int(np.ceil(h*overlap_multplier/patch_size)), retstep=True, dtype=int)

    gen = vrt_patch_generator(ds, rngx, rngy, patch_size)

    train_ds = tf.data.Dataset.from_generator(
        lambda: gen,
        output_signature=(
            tf.TensorSpec(shape=(patch_size, patch_size, nx), dtype=tf.float32)
        )
    )
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    # train_ds = train_ds.prefetch(1) # prefetch messes up prediction order

    n_patches = len(rngx)*len(rngy)
    y = []

    # figure out number of steps in each predict iteration
    all_steps = [steps_per_predict for v in range(
        n_patches//(steps_per_predict*batch_size))]
    all_steps.append(
        int(np.ceil((n_patches-sum(all_steps*batch_size))/batch_size)))

    # print(all_steps)
    for idx, steps in enumerate(all_steps):
        tmp = model.predict(train_ds, steps=steps, verbose=1)
        y.append(tmp)

    # Concatenate y and crop to length n_patches
    yh = np.concatenate(y, axis=0)
    if len(yh) > n_patches:
        yh = yh[:n_patches]

    # Create weighted average for all pixels in the image
    # avg = sum(a * weights) / sum(weights)
    _, ph, pw, n_classes = yh.shape

    sigmoid_width = options['sigmoid_width']
    sigmoid_steepness = options['sigmoid_steepness']

    weights = np.expand_dims(sigmoid_window(
        patch_size, sigmoid_width, sigmoid_steepness), -1)
    yh_weighted = yh*weights

    sum_aw = np.zeros((h, w, n_classes), dtype=np.float32)
    sum_w = np.zeros((h, w, 1), dtype=np.float32)

    idx = 0
    for x in rngx:
        for y in rngy:
            sum_w[y:y+ph, x:x+pw, :] = sum_w[y:y+ph, x:x+pw, :]+weights
            sum_aw[y:y+ph, x:x+pw, :] = sum_aw[y:y +
                                               ph, x:x+pw]+yh_weighted[idx, :, :, :]
            idx += 1

    full_weighted = np.divide(sum_aw, sum_w)
    bin_full_weighted = np.where(full_weighted > 0.1, 1, full_weighted)

    write_raster(bin_full_weighted, vrt_filename, output_filename)
#%%

# km10_path = "R:/PROJ/10/415/217/20_Aflevering/raekkefoelge.gpkg"
# driver = ogr.GetDriverByName('gpkg')
# km10 = driver.Open(km10_path)

# if km10 is None:
#     print('Could not open %s' % (km10_path))
# else:
#     layer = km10.GetLayer()
#     featureCount = layer.GetFeatureCount()
#     extent = layer.GetExtent()
#     print("Number of features in %s: %d;" %
#           (os.path.basename(km10_path), featureCount))

# #%%

# for feature in layer:

#     if feature.GetField('lev_blok') != 1:
#         continue

#     tiles = feature.GetField('tilename')
#     tile_geometry = feature.GetGeometryRef()
#     tile_extent = tile_geometry.GetEnvelope()
#     print(f"Tile: {tiles}, extent: {tile_extent}, geometry: {tile_geometry}")

#     [xmin, xmax, ymin, ymax] = tile_extent
#     [bfxmin, bfxmax, bfymin, bfymax] = [xmin - (250*0.4), xmax + (250*0.4), ymin - (250*0.4), ymax + (250*0.4)] #250 (tiles) * tile size (0.4)

#     vrt = '//pc116900/S Drone div/STENDIGER/vrts/merged.vrt'

#     ds = gdal.Open(vrt)

#     band = ds.GetRasterBand(1)

#     geotransform = ds.GetGeoTransform()

#     xinit = geotransform[0]
#     yinit = geotransform[3]

#     xsize = geotransform[1]
#     ysize = geotransform[5]

#     p1 = (bfxmin, bfymax)
#     p2 = (bfxmax, bfymin)
#     row1 = int((p1[1] - yinit)/ysize)
#     col1 = int((p1[0] - xinit)/xsize)

#     row2 = int((p2[1] - yinit)/ysize)
#     col2 = int((p2[0] - xinit)/xsize)

#     data = band.ReadAsArray(col1, row1, col2 - col1, row2 - row1)

#     break


# #%%


# # p1 = (440000.0, 6410000.0)
# # p2 = (900000.0, 6040000.0)

# p1 = (440000.0, 6410000.0)
# p2 = (900000.0, 6040000.0)

# upload model
model_path = "V:/2022-03-31_Stendiger_EZRA/code/logs/"

custom_objects = {
    'binary_focal_loss_plus_jaccard_loss': sm.losses.binary_focal_jaccard_loss,
    'iou_score': sm.metrics.iou_score
}

model = load_model(model_path + 'model.25-0.832-0.190-0.984.h5',
                   custom_objects=custom_objects)

vrt_filename = glob(
    '//pc116900/S Drone div/STENDIGER/vrts/10kmv2/ezra/*.vrt')
#%%
for idx, vrt in enumerate(vrt_filename, start=1):
    
    # if os.path.basename(vrt) != '10km_608_47.vrt':
    #     print(vrt, 'skipped')
    #     continue
    if not os.path.exists(vrt):
        raise Exception(f"{vrt} does not exist")
    output_name = os.path.basename(vrt).replace('vrt', 'tif')
    output_filename = f'R:/PROJ/10/415/217/20_Aflevering/leverance_1/{output_name}'
    if not os.path.isdir('R:/PROJ/10/415/217/20_Aflevering/leverance_1/'):
        raise Exception(f"{output_filename} does not exist")
    options = {
        'n_bands_x': 3,
        'n_classes': 2,
        'patch_size': 64,
        'overlap': 2,
        'batch_size': 128,
        'steps_per_predict': 128,
        'sigmoid_width': 3,
        'sigmoid_steepness': 7
    }
    
    print(f"predicting {output_name}... --> ({idx})")
    try:
        smooth_vrt_predictions(model, vrt, output_filename, options)
    except:
        print(f"{output_name} failed")
        continue
print("Done, all vrts predicted.")

# %%
# ###generate tiles from 10km and patches for prediction
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
