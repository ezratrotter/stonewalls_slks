#%%

import sys
import os
from osgeo import gdal, ogr, osr
from glob import glob

gdal.UseExceptions()

leverance_nr = 2

rdrive_base = 'R:/PROJ/10/415/217/20_Aflevering/'
rleverance = rdrive_base + 'leverance_{}/'.format(leverance_nr)

tif_list = glob(rleverance + "*.tif")

for tif in tif_list:

    tif_name = os.path.basename(tif).replace('.tif', '')
    dst_layerpath = rleverance + tif_name

    print(f"Reading {tif_name} raster")
    #if vector file exists skip it
    if os.path.exists(dst_layerpath + ".gpkg"):
                print((tif_name + ".gpkg"), 'already exists!')
                continue

    ds = gdal.Open(tif)

    #get projection from raster
    dest_srs = osr.SpatialReference()
    proj = ds.GetProjectionRef()
    dest_srs.ImportFromWkt(proj)

    #sanity check
    if ds is None:
        print('Unable to open {}'.format(tif_name))
        sys.exit(1)

    try:
        band = ds.GetRasterBand(1)
    except RuntimeError as e:
        print("Unable to read band from {}".format(tif_name))
        sys.exit(1)

    #set driver type
    drv = ogr.GetDriverByName("GPKG")

    #create file and layer
    dst_ds = drv.CreateDataSource(dst_layerpath + '.gpkg')
    dst_layer = dst_ds.CreateLayer(str(tif_name), geom_type=ogr.wkbPolygon, srs=dest_srs)

    #create field for raster value
    fd = ogr.FieldDefn("wall", ogr.OFTInteger)
    dst_layer.CreateField(fd)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("wall")

    #polygonize and include band as mask to not take in 0 values
    print(f"Vectorizing {tif_name} tile...")
    gdal.Polygonize(band, band, dst_layer, dst_field, [], callback=None)

    #clean up
    del ds
    del dst_ds
    del dst_layer
    del dst_field

print(f"All tiles from leverance {leverance_nr} vectorized!")

# %%
import os
import glob
import geopandas as gpd

myfiles = glob.glob("D:/stonewalls_slks/lev/*.gpkg")

print(len(myfiles))

if not os.path.isdir("D:/stonewalls_slks/lev/new/"):
    os.mkdir("D:/stonewalls_slks/lev/new/")

for f in myfiles:
    new_name = f.replace("lev", "lev/new")
    gdf = gpd.read_file(f)
    gdf.to_file(new_name, driver="GPKG")
    #os.rename(f, f[1:])
    #dir = "D:/stonewalls_slks/"
    #base = os.path.basename(f)[1:]
    #os.rename(f, dir + base)
    #print(dir + base)

# %%
