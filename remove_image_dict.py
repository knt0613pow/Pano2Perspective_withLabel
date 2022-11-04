import json
import os

import sys


if __name__ == "__main__":
    file_path = sys.argv[1] 
    file_list  = os.listdir(file_path)
    file_list = [os.path.join(file_path, path) for path in file_list] 
    for item in file_list:
        if item[-4:] != 'json' : continue
        if item[-13:] == 'Download.json' : continue
        with open (item, "r") as json_file:
            jdata = json.load(json_file)
        print(item)
        try:
            del jdata["imageData"]
        except KeyError:
            pass
        
        with open (item, "w" ) as json_file:
            jdata = json.dump(jdata, json_file)

        
