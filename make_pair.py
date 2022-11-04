import sys
import os
from make_sequence import Co2path
from make_sequence import dist
import numpy as np
import math
from Pano2PerspecBB import Equirectangular
import glob
import cv2
import re

def id_angle2name(id, angle, idx2lat, idx2lng):
    lat = round(idx2lat[id], 4)
    lng = round(idx2lng[id],4)
    name = str(lat) + '-' + str(lng) + '-' + str(angle)
    return name





if __name__ == "__main__":
    Region_name = sys.argv[1]
    FOV = int(sys.argv[2])/2
    Region_path = os.path.join(os.getcwd(), Region_name)
    Down_json_path = os.path.join(Region_path, 'Download.json')
    Pair_path = os.path.join(Region_path, 'pair_data')

    # os.mkdir(Pair_path)
    if not os.path.exists(Pair_path):
        os.makedirs(Pair_path)
    path = Co2path(Down_json_path)
    Width = path.Road_width

    T1 = list(path.Time2idx.values())[0]
    T2 = list(path.Time2idx.values())[1]

    
    """
    relation[0] = ["P1.jpg", "P2.jpg"]
    relation[1] = ["P3.jpg", "P4.jpg"]
    """
    valid_image_list = {}
    relation = []
    for i, point in enumerate(path.path):
        if point not in T1: continue
        near_P = path.path[i-2:i+2]
        before_P = None
        after_P = None
        for j in range(1,3):
            if ((i-j) >= 0) and path.path[i-j] in T2 :
                before_P = path.path[i-j]
                break
        for j in range(1,3):
            if ((i+j) < len(path.path)) and path.path[i+j] in T2 :
                after_P = path.path[i+j]
                break
        loc_now = (path.idx2lat[point], path.idx2lng[point])
        
        if before_P is not None :
            loc_before = (path.idx2lat[before_P], path.idx2lng[before_P])
            
            dist_1 = dist(loc_before, loc_now)

            valid_angle_1 = np.arctan2(Width ,dist_1)* 180 / np.pi
            BP_angle_1 = np.arange((valid_angle_1 - valid_angle_1%10 +10) - FOV, valid_angle_1 - valid_angle_1 %10 +FOV ,10.0)
            NP_angle_1 = np.arange((170-valid_angle_1 + valid_angle_1%10)+FOV, 180-valid_angle_1 + valid_angle_1 %10 -FOV,-10.0)
            if before_P not in valid_image_list.keys():
                valid_image_list[before_P] = set(BP_angle_1)
                valid_image_list[before_P].update(set(-BP_angle_1))
            else : 
                valid_image_list[before_P].update(set(BP_angle_1))
                valid_image_list[before_P].update(set(-BP_angle_1))

            if point not in valid_image_list.keys():
                valid_image_list[point] = set(NP_angle_1)
                valid_image_list[point].update(set(-NP_angle_1))
            else : 
                valid_image_list[point].update(set(NP_angle_1))
                valid_image_list[point].update(set(-NP_angle_1))

            for nps, bps in zip(NP_angle_1, BP_angle_1):
                relation.append((id_angle2name(point, bps, path.idx2lat, path.idx2lng),
                                id_angle2name(after_P, nps, path.idx2lat, path.idx2lng)))
                relation.append((id_angle2name(point, -bps, path.idx2lat, path.idx2lng),
                                id_angle2name(after_P, -nps, path.idx2lat, path.idx2lng)))
        
        if after_P is not None:
            loc_after = (path.idx2lat[after_P], path.idx2lng[after_P])
            dist_2 = dist(loc_now, loc_after)

            valid_angle_2 = np.arctan2(path.Road_width,dist_2)* 180 / np.pi
            NP_angle_2 = np.arange((valid_angle_2 - valid_angle_2%10 +10) - FOV, valid_angle_2 - valid_angle_2 %10 +FOV ,10.0)
            AP_angle_2 = np.arange((170-valid_angle_2 + valid_angle_2%10)+FOV, 180-valid_angle_2 + valid_angle_2 %10 -FOV,-10.0)
            if after_P not in valid_image_list.keys():
                valid_image_list[after_P] = set(AP_angle_2)
                valid_image_list[after_P].update(set(-AP_angle_2))
            else : 
                valid_image_list[after_P].update(set(AP_angle_2))
                valid_image_list[after_P].update(set(-AP_angle_2))
            if point not in valid_image_list.keys():
                valid_image_list[point] = set(NP_angle_2)
                valid_image_list[point].update(set(-NP_angle_2))
            else : 
                valid_image_list[point].update(set(NP_angle_2))
                valid_image_list[point].update(set(-NP_angle_2))

            for nps, aps in zip(NP_angle_2, AP_angle_2):
                relation.append((id_angle2name(point, nps, path.idx2lat, path.idx2lng),
                                id_angle2name(after_P, aps, path.idx2lat, path.idx2lng)))
                relation.append((id_angle2name(point, -nps, path.idx2lat, path.idx2lng),
                                id_angle2name(after_P, -aps, path.idx2lat, path.idx2lng)))
        imgEx = r'.jpg'
        jsonEx = r'.json'
        img_list = [file for file in os.listdir(Region_path) if file.endswith(imgEx)]
        json_list =  [file for file in os.listdir(Region_path) if file.endswith(jsonEx)]
        json_list.remove('Download.json')

        img_list = sorted(img_list, key = lambda s : int(re.search(r'\d+', s).group()))
        json_list = sorted(json_list, key = lambda s : int(re.search(r'\d+', s).group()))

        for idx, angles in valid_image_list.items():
            Er = Equirectangular(os.path.join(Region_path, img_list[idx]), os.path.join(Region_path,json_list[idx]))
            for angle in angles:
                per = Er.GetPerspective(2*FOV, angle, 0, 720, 1080)
                name = str(id_angle2name(idx, angle, path.idx2lat, path.idx2lng)) + '.jpg'
                label_name = str(id_angle2name(idx, angle, path.idx2lat, path.idx2lng)) + '.json'
            
                persec_path = os.path.join(Pair_path, name)
                label_path = os.path.join(Pair_path, label_name)
                cv2.imwrite(persec_path, per)
                
            
        
        


    
        # for sP in near_P:
        #     if (sP in T2) and  (before_P == None): before_P = sP
        #     if (sP in T2) : after_P = sP


    """
        self.path = path
    self.Time2idx = Time2idx
    self.idx2lat = idx2lat
    self.idx2lng = idx2lng
    """
    #P1 = path.Time2idx.