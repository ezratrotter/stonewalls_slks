#%%
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

tif = r'D:\stonewalls_slks\data\test_data\prediction_614_67.tif'
ds = gdal.Open(tif)
channel = np.array(ds.GetRasterBand(1).ReadAsArray())

mask =  [[False, True, False], [True, True, True], [False, True, False]]




channel = ndimage.binary_erosion(channel, mask, iterations=1)
channel = ndimage.binary_dilation(channel, mask, iterations=1)
channel = ndimage.binary_opening(channel, mask, iterations=1)
channel = ndimage.binary_closing(channel, mask, iterations=1)




path1 = f"C:/Users/EZRA\Desktop/test1.tif"



write_data_as_raster(channel1, tif, path1)


# %%
