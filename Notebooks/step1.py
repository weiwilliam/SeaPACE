import numpy as np
from owslib.wms import WebMapService
import lxml.etree as xmltree
import xml.etree.ElementTree as xmlet
import requests
from skimage import io

def get_OCI_PACE_truecolor(time, size=(400, 800), bbox=(-180, -90, 180, 90)):
    """
      time: in format of YYYY-MM-DD
      size: (height, width)
      bbox: bounding box (minlon, minlat, maxlon, maxlat)
    """
    height, width = size
    minlon, minlat, maxlon, maxlat = bbox
    #  Construct Geographic projection URL.
    proj4326 = f'https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi?version=1.3.0&service=WMS&request=GetMap&format=image/png&STYLE=default&bbox={int(minlat)},{int(minlon)},{int(maxlat)},{int(maxlon)}&CRS=EPSG:4326&HEIGHT={height}&WIDTH={width}&TIME={time}&layers=OCI_PACE_True_Color'
    
    # Request image.
    img = io.imread(proj4326)
    x = np.linspace(minlon, maxlon, img.shape[0])
    y = np.linspace(minlat, maxlat, img.shape[1])
    print(img.shape)
    # img = img[::-1, :]

    return x, y, img