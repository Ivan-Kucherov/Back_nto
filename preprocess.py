import base64
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
import numpy as np
import io
from PIL.ExifTags import TAGS, GPSTAGS

def get_geo(image):
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for gps_tag in value:
                    sub_decoded = GPSTAGS.get(gps_tag, gps_tag)
                    if sub_decoded == 'GPSLatitude':
                        lat = float(value[gps_tag][0]) + float(value[gps_tag][1])/60 + float(value[gps_tag][2])/3600
                    if sub_decoded == 'GPSLongitude':
                        lng = float(value[gps_tag][0]) + float(value[gps_tag][1])/60 + float(value[gps_tag][2])/3600
    if lat is not None:
        exif_data = (lng,lat)
    else:
        exif_data = None

    return exif_data   


def pil_from_base(base):
    images = [Image.open(io.BytesIO(base64.decodebytes(bytes(i[str(i).find('base64,')+7:],"utf-8")))) for i in base]
    return images
def get_preds(images):
    pass
