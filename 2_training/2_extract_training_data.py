#%%

from osgeo import gdal
from osgeo import ogr
from osgeo import gdalconst
import numpy as np
import glob
from multiprocessing.pool import ThreadPool

import matplotlib.pyplot as plt


def read_raster(model_raster_fname):
    model_dataset = gdal.Open(model_raster_fname)
    return model_dataset.ReadAsArray()


def tile_images(images, tile_sz, overlap_multiplier=1):
    """Input: List of images, Output: List of tiles"""
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
    return (lst, rng_x, rng_y)


def rasterize_shp_like(shapefiles, model_raster_fname, output_fname, options, nodata_val=0, verbose=False):
    """
    Given a shapefile, rasterizes it so it has
    the exact same extent as the given model_raster

    `dtype` is a gdal type like gdal.GDT_Byte
    `options` should be a list that will be passed to GDALRasterizeLayers
        papszOptions, like ["ATTRIBUTE=vegetation","ALL_TOUCHED=TRUE"]
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


# %%
# Create binary wall tiles
xdir = r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\dem\dtm'
outputdir = r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\walls'

paths = glob.glob(f'{xdir}/*.tif')

vectordata = [
    r'V:\2022-03-31_Stendiger_EZRA\training_data\shp\pilot_area\walls_10km_614_67.shp']
    # r'V:\2022-03-31_Stendiger_EZRA\training_data\shp\absence.shp']


def rasterize_file(fp):
    fn = fp.split('\\')[-1]
    fp_out = f'{outputdir}/{fn}'
    rasterize_shp_like(vectordata, fp, fp_out, [])


# rasterize_file([v for v in paths if '1km_6258_598' in v][0])
#%%

results = ThreadPool(24).imap(rasterize_file, paths)

#%%
tiles = [tile for tile in results]

# %%

#%% extract patches (no overlaps for now)

xdir = r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\dem'
ydir = r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\walls'
outputdir_data = r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\npz_tile_data'
outputdir_nodata = r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\npz_tile_nodata'

#%%
def tile_image(fp):
    # Get filenames
    fn = fp.split('\\')[-1].replace('DTM_','')
    fp_dtm = f'{xdir}/dtm/DTM_{fn}'
    fp_hat = f'{xdir}/hat/HAT_{fn}'
    fp_sobel = f'{xdir}/sobel/SOBELDTM_{fn}'
    fp_wall = f'{ydir}/DTM_{fn}'

    

    # Read image data
    images = [fp_dtm, fp_hat, fp_sobel, fp_wall]
    image_data = [read_raster(v) for v in images]

    image_data = [image_data[0], image_data[1], image_data[2], image_data[3][0], image_data[3][1]] 

    # Tile image
    tiles, rng_x, rng_y = tile_images(image_data, 64)

    x = np.concatenate(tiles[:-2], axis=-1)
    y = tiles[-2]
    absence = tiles[-1]

    # Extract tiles with sum > 0 for y
    idx_data = np.sum(y, axis=(1,2,3)) > 0
    idx_nodata = np.sum(absence, axis=(1,2,3)) == 64*64

    x_data = x[idx_data,:]
    x_nodata = x[idx_nodata,:]

    y_data = y[idx_data,:]
    y_nodata = y[idx_nodata,:]

    # Save tiles
    npz_fn = fn.replace('.tif', '.npz')
    np.savez(f'{outputdir_data}/{npz_fn}', x=x_data, y=y_data )
    np.savez(f'{outputdir_nodata}/{npz_fn}', x=x_nodata, y=y_nodata )

    # return y_data, y_nodata

filepaths = glob.glob(f'{ydir}/*.tif')

# y_data, y_nodata = tile_image(filepaths[0])

#%%

results = ThreadPool(24).imap(tile_image, filepaths)
tiles = [tile for tile in results]

#%%
##tile 10km for test prediction

tifs_dir = r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\dem\vrt'

tifs = glob.glob(f'{tifs_dir}/*.tif')
out_tiles = r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\prediction_tiles'

#%%

def tile_image(fp):
    # Get filenames
    fp_dtm = f'{fp}/dtm_10km_614_67.tif'
    fp_hat = f'{fp}/hat_10km_614_67.tif'
    fp_sobel = f'{fp}/sobel_10km_614_67.tif'

    

    # Read image data
    images = [fp_dtm, fp_hat, fp_sobel]
    image_data = [read_raster(v) for v in images]

    image_data = [image_data[0], image_data[1], image_data[2]] 

    # Tile image
    tiles, rng_x, rng_y = tile_images(image_data, 64)

    x = np.concatenate(tiles, axis=-1)
    # y = tiles[-2]
    # absence = tiles[-1]

    # Extract tiles with sum > 0 for y
    # idx_data = np.sum(y, axis=(1,2,3))
    # idx_nodata = np.sum(absence, axis=(1,2,3)) == 64*64

    # x_data = x[idx_data,:]
    # x_nodata = x[idx_nodata,:]

    # y_data = y[idx_data,:]
    # y_nodata = y[idx_nodata,:]

    
    # Save tiles
  
    np.savez(f'{out_tiles}/stack_10km_614_67.npz', x=x)
    # np.savez(f'{outputdir_nodata}/{npz_fn}', x=x_nodata, y=y_nodata )

    # return y_data, y_nodata


stack_data = tile_image(tifs_dir)

# %%
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
    return (lst, rng_x, rng_y)


fp = r'V:\2022-03-31_Stendiger_EZRA\training_data\initial_area\dem\vrt'
fp_dtm = f'{fp}/dtm_10km_614_67.tif'
fp_hat = f'{fp}/hat_10km_614_67.tif'
fp_sobel = f'{fp}/sobel_10km_614_67.tif'

#%%

# Read image data
images = [fp_dtm, fp_hat, fp_sobel]
image_data = [read_raster(v) for v in images]

image_data = [image_data[0], image_data[1], image_data[2]]

#%%
### make patches of stacked arrays
tiles, rng_x, rng_y = tile_images(image_data, 64)