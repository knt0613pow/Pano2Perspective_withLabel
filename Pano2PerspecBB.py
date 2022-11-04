from ctypes.wintypes import HACCEL
import os
import sys
import cv2
import numpy as np
import copy
from linesegmentintersections import bentley_ottman


def get_polygon_intersection(polygon, W, H):
    """
    polygon : (N,2)
    """
    inter = []
    rectangle = [[[0,0], [0,H-1]], [[0,H-1], [W-1, H-1]], [[W-1, H-1], [W-1, 0]], [[W-1, 0], [0,0]]]
    inner_idx = None
    for idx ,point in enumerate(polygon):
        
        if not (point[0]>0 and point[0] < W-1 and point[1]> 0 and point[1] < H-1) : continue
        else : 
            inner_idx = idx
            break
    breakpoint()
    poly_sort = np.concatenate((polygon[inner_idx:], polygon[:inner_idx+1]))

    before_inner = False
    for idx2, point in enumerate(poly_sort):
        if (point[0]>0 and point[0] < W-1 and point[1]> 0 and point[1] < H-1) : 
            if not before_inner:
                seg = [[poly_sort[idx2-1], poly_sort[idx2-1]], [point[0], point[1]]]
                for lin in rectangle:
                    rec_seg_intersection = bentley_ottman([lin, seg])
                    if len(rec_seg_intersection) != 0 :
                        inter.append([rec_seg_intersection[0].x, rec_seg_intersection[0].y])
                        break

            inter.append(point)
            before_inner = True
        else:
            if before_inner : 
                seg = [[inter[-1][0], inter[-1][0]], [point[0], point[1]]]
                for lin in rectangle:
                    rec_seg_intersection = bentley_ottman([lin, seg])
                    if len(rec_seg_intersection) != 0 :
                        inter.append([rec_seg_intersection[0].x, rec_seg_intersection[0].y])
                        break

            before_inner = False
    return inter

def clip_over_point(polygon, W, H):
    inter = []
    rectangle = [[[0,0], [0,H-1]], [[0,H-1], [W-1, H-1]], [[W-1, H-1], [W-1, 0]], [[W-1, 0], [0,0]]]
    inner_idx = None
    for idx ,point in enumerate(polygon):
        
        if not (point[0]>0 and point[0] < W-1 and point[1]> 0 and point[1] < H-1) : continue
        else : 
            inner_idx = idx
            break
    if inner_idx == None:
        return []
    polygon[:,0] = np.clip(polygon[:,0], 1 , W-2)
    polygon[:,1] = np.clip(polygon[:,1], 1, H-2)
    return polygon


def is_Inner(point, W, H):
    x , y = point[0], point[1]
    # breakpoint()
    if (x>=0) and (x<W) and (y>=0) and (y<H) : return True
    else : return False

def find_intersec(inner_point, out_point, W, H):
    ts = np.arange(0, 101) /100
    inner2out = out_point -inner_point
    inter = 0
    for t in ts:
        now_point = inner_point + t*inner2out
        if not is_Inner(now_point, W, H): 
            break
        inter += 1
    
    intersec = inner_point + inter*inner2out/100
    return intersec

def clip_over_point_intersection(polygon, W, H):
    inter = []
    rectangle = [[[0,0], [0,H-1]], [[0,H-1], [W-1, H-1]], [[W-1, H-1], [W-1, 0]], [[W-1, 0], [0,0]]]
    inner_idx = None
    for idx ,point in enumerate(polygon):
        if not (point[0]>0 and point[0] < W-1 and point[1]> 0 and point[1] < H-1) : continue
        else : 
            inner_idx = idx
            break
    if inner_idx == None:
        return []
    num_point = polygon.shape[0]

    clip_point = []
    for idx ,point in enumerate(polygon):
        p1 = polygon[idx]
        p2 = polygon[(idx+1)%num_point]
        if not (is_Inner(p1, W, H) or is_Inner(p2, W, H)): continue
        if is_Inner(p1, W, H) and (not is_Inner(p2, W, H)) : 
            clip_point.append(p1)
            clip_point.append(find_intersec(p1, p2, W, H))
        if (not is_Inner(p1, W, H)) and ( is_Inner(p2, W, H)) :
            clip_point.append(find_intersec(p2, p1, W,H))
        if is_Inner(p1, W, H) and is_Inner(p2, W, H) :
            clip_point.append(p1)

    polygon = np.array(clip_point)
    polygon[:,0] = np.clip(polygon[:,0], 1 , W-2)
    polygon[:,1] = np.clip(polygon[:,1], 1, H-2)
    return polygon




def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out

def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out 
    
def Panorama2lonlat(label, shape):
    """
    shape : (H, W, 3)
    label : (N,2) # of (x, y) coordinate or panorama image
    x is right direction coordinate
    y is down direction coordinate
    return : (2, N)
    """
    lon = (label[:,0]/(shape[1]) -0.5)*(2*np.pi)
    lat = (label[:,1]/(shape[0])-0.5)*(np.pi)
    lonlat = [lon, lat]
    return np.array(lonlat)

    
def lonlat2xyz(lonlat):
    """
    lonlat : (2,N) 
    xyz : len(xyz) == 3
    return : (N,3)
    """
    lon, lat = lonlat
    # breakpoint()
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)

    return np.stack((x,y,z), axis = 0).T
    
    
def valid_label(lonlat, THETA):

    # breakpoint()
    lon, lat = lonlat
    theta = THETA / np.pi
    left = (THETA-90)* np.pi/180
    right = (THETA+90) * np.pi/180
    if THETA <= -90:
        idx = np.where( (2*np.pi+left)<lon or lon <right)
    elif THETA < 90:
        idx = np.where(left < lon and lon<right)
    else:
        idx = np.where(left < lon or (-2*np.pi +right) > lon)
    return lonlat[:, idx]

class Equirectangular:
    def __init__(self, img_name, label):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        # breakpoint()
        with open(label, 'r') as f:
            json_data = json.load(f)
        self._label = json_data
        # breakpoint()
        [self._height, self._width, _] = self._img.shape
        #cp = self._img.copy()  
        #w = self._width
        #self._img[:, :w/8, :] = cp[:, 7*w/8:, :]
        #self._img[:, w/8:, :] = cp[:, :7*w/8, :]
    

    def GetPerspective(self, FOV, THETA, PHI, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0,  1],
            ], np.float32)
        K_inv = np.linalg.inv(K)
        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)  
        z = np.ones_like(x)
        xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
        xyz = xyz @ K_inv.T

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        xyz = xyz @ R.T
        lonlat = xyz2lonlat(xyz) 
        XY = lonlat2XY(lonlat, shape=self._img.shape).astype(np.float32)
        persp = cv2.remap(self._img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

        return persp
    def GetPerspective_label(self, FOV, THETA, PHI, height, width):
        f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        x_axis = np.array([1.0, 0.0, 0.0], np.float32)
        R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
        R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
        R = R2 @ R1
        R_inv = np.linalg.inv(R)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array([
                [f, 0, cx],
                [0, f, cy],
                [0, 0, 1],
            ], np.float32)    

        perpec_label = copy.deepcopy(self._label)
        remov_obj = []
        for i, obj_polygon in enumerate(self._label['shapes']):
            points = obj_polygon['points']
            points = np.array(points)
            lonlat = Panorama2lonlat(points, shape = self._img.shape)
            # lonlat = valid_label(lonlat, THETA)
            xyz = lonlat2xyz(lonlat)
            camera_xyz = xyz@R_inv.T
            # breakpoint()
            # if i == 10 : breakpoint()
            if np.sum(camera_xyz[:,2] != np.abs(camera_xyz[:,2])) != 0:
                remov_obj.append(i)
                continue
            # breakpoint()
            camera_xyz =camera_xyz / (camera_xyz[:,2].reshape(-1,1))
            camera_xyz = camera_xyz @ K.T
            # breakpoint()
            camera_xy = camera_xyz[:,:2]
            clip_camera_xy = clip_over_point_intersection(camera_xy, width, height)


            

            if len(clip_camera_xy) ==0: 
                remov_obj.append(i)
                continue
            object_area = (max(camera_xy[:, 0]) -min(camera_xy[:,0])) * (max(camera_xy[:, 1]) -min(camera_xy[:,1]))
            clip_area = (max(clip_camera_xy[:, 0]) -min(clip_camera_xy[:,0])) * (max(clip_camera_xy[:, 1]) -min(clip_camera_xy[:,1]))
    
            if 0.6 * object_area >clip_area : 
                remov_obj.append(i)
                continue
            
            perpec_label['shapes'][i]['points'] = camera_xy.tolist()

        # get index list in reverse order
        drop_inds = sorted(remov_obj, reverse=True)
        # drop elements in-place
        for idx in drop_inds:
            del perpec_label['shapes'][idx]

        # breakpoint()
        return perpec_label
        
        
        
        
        
        
import json
        
        
if __name__ == '__main__':
    img_path = 'R1/Street View_인덕원_0_2018_11_00_37.397825_126.982295.jpg'
    label = 'R1/Street View_인덕원_0_2018_11_00_37.397825_126.982295.json'
    Er = Equirectangular(img_path, label)
    per = Er.GetPerspective(90, -60, 0, 720, 1080)
    
    
    
    label_idx = Er.GetPerspective_label(90, -60, 0, 720, 1080)
    breakpoint()
    # per = cv2.circle(per,(int(label_idx[0,0]), int(label_idx[0,1])),8,(0,0,255),3)
    cv2.imwrite('test_result.jpg', per)
    #should be ~345,30
