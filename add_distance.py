import json
import os

import sys


if __name__ == "__main__":
    file_path = sys.argv[1] 
    distance = sys.argv[2]
    Download_json_path = os.path.join(file_path, 'Download.json')
    with open (Download_json_path, "r") as json_file:
        jdata = json.load(json_file)
    jdata["length"] = float(distance)
    with open (Download_json_path, "w" ) as json_file:
        jdata = json.dump(jdata, json_file)

        
