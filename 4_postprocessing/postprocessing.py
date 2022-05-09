## Python program to demonstrate erosion and

# dilation of images.

from scipy import ndimage

import numpy as np

from osgeo import gdal

from osgeo import ogr

from osgeo import gdalconst



def write_data_as_raster(image_data, fn_raster, fn_output, fn_poly = None):

    data = gdal.Open(fn_raster, gdalconst.GA_ReadOnly)

    geo_transform = data.GetGeoTransform()

   

    x_min = geo_transform[0]

    y_min = geo_transform[3]



    x_res = data.RasterXSize

    y_res = data.RasterYSize



    srs = data.GetProjection()



    pixel_width = geo_transform[1]



    target_ds = gdal.GetDriverByName('GTiff').Create(fn_output, x_res, y_res, 1, gdal.GDT_Byte)

    target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, -pixel_width))

    target_ds.SetProjection(srs)



   

    band = target_ds.GetRasterBand(1)

    NoData_value = -1

    band.SetNoDataValue(NoData_value)

    band.FlushCache()

    #gdal.RasterizeLayer(target_ds, [1], mb_l,burn_values=[255])

    target_ds.GetRasterBand(1).WriteArray(image_data)



    if fn_poly != None:

       

        driver = ogr.GetDriverByName("ESRI Shapefile")

        outDatasource = driver.CreateDataSource(fn_poly)

        outLayer = outDatasource.CreateLayer("polygonized", srs=None)



        fd = ogr.FieldDefn( 'DN', ogr.OFTInteger )

        outLayer.CreateField( fd )



        gdal.Polygonize( target_ds.GetRasterBand(1), None, outLayer, 0)





    target_ds = None



tif = 'D:/stendiger_data/10km_610_57.tif'

ds = gdal.Open(tif)

channel = np.array(ds.GetRasterBand(1).ReadAsArray())



mask =  [[False, True, False], [True, True, True], [False, True, False]]
mask2 =  [[True, True, True], [True, True, True], [True, True, True]]


# channel0 = ndimage.binary_opening(channel, mask2).astype(int)

# channel1 = ndimage.binary_dilation(channel, mask2, iterations=5).astype(int)
# channel1 = ndimage.binary_erosion(channel1, mask2, iterations=5).astype(int)

# channel2 = ndimage.binary_dilation(channel, mask2, iterations=10).astype(int)
# channel2 = ndimage.binary_erosion(channel2, mask2, iterations=10).astype(int)

# channel3 = ndimage.binary_dilation(channel, mask2, iterations=15).astype(int)
# channel3 = ndimage.binary_erosion(channel3, mask2, iterations=15).astype(int)

# channel4 = ndimage.binary_dilation(channel, mask, iterations=5).astype(int)
# channel4 = ndimage.binary_erosion(channel4, mask, iterations=5).astype(int)

# channel5 = ndimage.binary_dilation(channel, mask, iterations=10).astype(int)
# channel5 = ndimage.binary_erosion(channel5, mask, iterations=10).astype(int)

# channel6 = ndimage.binary_dilation(channel, mask, iterations=15).astype(int)
# channel6 = ndimage.binary_erosion(channel6, mask, iterations=15).astype(int)




channel1 = ndimage.convolve(channel, np.ones((3,3)), mode='constant')



# channel2 = ndimage.binary_opening(channel1).astype(int)

# channel2 = ndimage.binary_closing(channel, mask2, iterations=4).astype(int)
# channel2 = ndimage.binary_erosion(channel2, mask2, iterations=1).astype(int)

# channel3 = ndimage.binary_opening(channel, mask2, iterations=1).astype(int)
# channel3 = ndimage.binary_closing(channel3, mask2, iterations=4).astype(int)


# eroded_square = ndimage.binary_erosion(square)
# reconstruction = ndimage.binary_propagation(eroded_square, mask=square)


path = 'D:/stendiger_data/postprocessing/test_dilation1_erosion1_box.tif'
# path1 = 'D:/stendiger_data/postprocessing/test_closing4_erosion1_box.tif'
# path2 = 'D:/stendiger_data/postprocessing/test_opening1_box.tif'

# write_data_as_raster(channel2, tif, path1)
write_data_as_raster(channel1, tif, 'D:/stendiger_data/postprocessing/test_convolution' + '1.tif')
# write_data_as_raster(channel2, tif, 'D:/stendiger_data/postprocessing/test_dilation1_erosion1_box' + '2.tif')
# write_data_as_raster(channel3, tif, 'D:/stendiger_data/postprocessing/test_dilation1_erosion1_box' + '3.tif')
# write_data_as_raster(channel4, tif, 'D:/stendiger_data/postprocessing/test_dilation1_erosion1_box' + '4.tif')
# write_data_as_raster(channel5, tif, 'D:/stendiger_data/postprocessing/test_dilation1_erosion1_box' + '5.tif')
# write_data_as_raster(channel6, tif, 'D:/stendiger_data/postprocessing/test_dilation1_erosion1_box' + '6.tif')
# write_data_as_raster(channel3, tif, path2)

