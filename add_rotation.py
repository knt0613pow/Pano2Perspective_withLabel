import json
import os
import re
import sys
from PIL import Image, ExifTags

if __name__ == "__main__":
    Region_path = sys.argv[1] 
    imgEx = r'.jpg'
    jsonEx = r'.json'
    img_list = [file for file in os.listdir(Region_path) if file.endswith(imgEx)]
    json_list =  [file for file in os.listdir(Region_path) if file.endswith(jsonEx)]
    json_list.remove('Download.json')

    img_list = sorted(img_list, key = lambda s : int(re.search(r'\d+', s).group()))
    json_list = sorted(json_list, key = lambda s : int(re.search(r'\d+', s).group()))


    for im, js in zip(img_list, json_list):
        breakpoint()
        img = Image.open(os.path.join(Region_path, im))
        exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }

    # with open (Download_json_path, "r") as json_file:
    #     jdata = json.load(json_file)
    # jdata["length"] = float(distance)
    # with open (Download_json_path, "w" ) as json_file:
    #     jdata = json.dump(jdata, json_file)

        
