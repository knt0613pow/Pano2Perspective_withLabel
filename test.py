import json
import os

import sys


if __name__ == "__main__":
    json_path  =  './bundang/Download.json'
    with open (json_path, "r") as json_file:
        jdata = json.load(json_file)
    breakpoint()

    
        