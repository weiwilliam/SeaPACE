from owslib.wms import WebMapService
import lxml.etree as xmltree
import xml.etree.ElementTree as xmlet
import requests
from skimage import io
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def get_OCI_PACE_truecolor(time, bbox):
    """
      time in format of YYYY-MM-DD
    """
    minlon, minlat, maxlon, maxlat = bbox
    #  Construct Geographic projection URL.
    proj4326 = f'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?version=1.3.0&service=WMS&request=GetMap&format=image/png&STYLE=default&bbox={int(minlat)},{int(minlon)},{int(maxlat)},{int(maxlon)}&CRS=EPSG:4326&HEIGHT=600&WIDTH=600&TIME={time}&layers=OCI_PACE_True_Color'
    
    # Request image.
    img = io.imread(proj4326)

    # Display image on map.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    extent = (minlon, maxlon, minlat, maxlat)
    plt.imshow(img, transform = ccrs.PlateCarree(), extent = extent, origin = 'upper')
    
    # Draw grid.
    gl = ax.gridlines(ccrs.PlateCarree(), linewidth = 1, color = 'blue', alpha = 0.3,  draw_labels = True)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = True
    gl.ylines = True
    # gl.xlocator = mticker.FixedLocator([0, 30, -30, 0])
    # gl.ylocator = mticker.FixedLocator([-30, 0, 30])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'blue'}
    gl.ylabel_style = {'color': 'blue'}
    
    plt.show()
    return img