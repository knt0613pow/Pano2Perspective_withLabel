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
from PIL import Image
import pickle
import json

def id_angle2name(id, angle, idx2lat, idx2lng):
    lat = round(idx2lat[id], 4)
    lng = round(idx2lng[id],4)
    name = str(id)+ '_' +str(lat) + '_' + str(lng) + '_' + str(angle)
    return name


def calcBearing (lat1, long1, lat2, long2): 
    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    bearing = math.atan2(x,y)   # use atan2 to determine the quadrant
    bearing = math.degrees(bearing)

    return bearing %360




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
    Dist_mat = path.dit_mat

    imgEx = r'.jpg'
    jsonEx = r'.json'
    img_list = [file for file in os.listdir(Region_path) if file.endswith(imgEx)]
    json_list =  [file for file in os.listdir(Region_path) if file.endswith(jsonEx)]
    json_list.remove('Download.json')

    img_list = sorted(img_list, key = lambda s : int(re.search(r'\d+', s).group()))
    json_list = sorted(json_list, key = lambda s : int(re.search(r'\d+', s).group()))

    direction_list = []
    
    for img in img_list:
        img_path = os.path.join(Region_name, img) 
        image = Image.open(img_path)
        exifdata = image._getexif()
        direction_list.append(exifdata[34853][17][0])
   
    valid_relation = np.where(Dist_mat<7)
    """
    relation[0] = ["P1.jpg", "P2.jpg"]
    relation[1] = ["P3.jpg", "P4.jpg"]
    """
    valid_image_list = {}
    relation = []
    pair = 0
    for p1, p2 in zip(valid_relation[0], valid_relation[1]):
        check_valid_relation = []

        if p1<= p2 : continue
        # if 
        if (p1 in T1 ) and (p2 in T2): continue
        if (p1 in T2) and (p2 in T2) : continue

        relative_direction = calcBearing(path.idx2lat[p1], path.idx2lng[p1],path.idx2lat[p2], path.idx2lng[p2])
        dist = Dist_mat[p1, p2]
        # theta1 = [-40, -30, -20, -10, 0]
        theta1 = []
        for id in range(0,5): #rectangel 4등분
            theta1.append(math.degrees(math.atan2(dist*id/4, 4)))
        # for id in range(1,5):
        #     temp_theta = math.degrees(math.atan2(4* math.cos(math.radians(id*10)),dist+ 4* math.sin(math.radians(id*10))))
        #     theta1.append(90 - temp_theta)
        theta1 = np.array(theta1)
        theta2 = -theta1[::-1]

        theta1 = np.concatenate((theta1, (180-theta1)[::-1]))
        theta1 = (theta1+360)%360


        theta2 = np.concatenate((theta2, (180-theta2)[::-1]))
        theta2 = (theta2+360)%360

        theta1 -= (90-relative_direction)
        theta2 -= (90-relative_direction)
        
        theta1 = np.round(theta1, -1)
        theta2 = np.round(theta2, -1)

        theta1 = (theta1+360)%360
        theta2 = (theta2+360)%360

        for th1, th2 in zip(theta1, theta2):
            if [th1, th2] in check_valid_relation : continue

            check_valid_relation.append([th1, th2])
            relation.append((id_angle2name(p1, th1, path.idx2lat, path.idx2lng),
                                id_angle2name(p2, th2, path.idx2lat, path.idx2lng)))
            if p1 not in valid_image_list.keys():
                valid_image_list[p1] = set([th1])
            else : 
                valid_image_list[p1].update(set([th1]))
            if p2 not in valid_image_list.keys():
                valid_image_list[p2] = set([th2])
            else : 
                valid_image_list[p2].update(set([th2]))

        # for theta

        # relative_direction = calcBearing()
    breakpoint()
    for idx, angles in valid_image_list.items():
        Er = Equirectangular(os.path.join(Region_path, img_list[idx]), os.path.join(Region_path,json_list[idx]))
        for ang in angles:
            angle = ang-direction_list[idx] 
            per = Er.GetPerspective(2*FOV, angle, 0, 720, 1080)
            name = str(id_angle2name(idx, ang, path.idx2lat, path.idx2lng)) + '.jpg'
            label_name = str(id_angle2name(idx, ang, path.idx2lat, path.idx2lng)) + '.json'
        
            persec_path = os.path.join(Pair_path, name)
            label_path = os.path.join(Pair_path, label_name)
            cv2.imwrite(persec_path, per)

            label = Er.GetPerspective_label(2*FOV, angle, 0, 720, 1080)
            label["imagePath"] = name
            with open (label_path, "w" ) as json_file:
                json.dump(label, json_file)
            

        
    with open(os.path.join(Region_path, Pair_path, "relation.pkl"),"wb") as f:
        pickle.dump(relation, f)
        
        


    
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