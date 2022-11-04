import json
import argparse 
import geopy
from geopy import distance
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import floyd_warshall
import matplotlib.pyplot as plt


class Co2path:
  def __init__(self, download_json):
    with open(download_json, 'r') as f:
      self.download_json = json.load(f)
    self.Road_width = self.download_json["length"]
    json_data = self.download_json
    PN_info = json_data['successResults']
    total_N = len(json_data["uniquePanoIds"])
    idx2PanoId = []
    idx2lat = []
    idx2lng = []
    idx2Time = []
    Time2idx = {}
    PanoId2idx = {}
    semi_idx = 0
    for i, PN in enumerate(PN_info):
      if PN['panoId'] not in PanoId2idx.keys():
        PanoId2idx[PN['panoId']] = i
        idx2PanoId.append(PN['panoId'])
        idx2lat.append(PN['location']['lat'])
        idx2lng.append(PN['location']['lng'])
        PN['date'] = (str(PN['date']['year']) +"-" +str(PN['date']['month']))
        idx2Time.append(PN['date'])
        if PN['date'] in Time2idx.keys():
          Time2idx[PN['date']].append(semi_idx)
        else :
          Time2idx[PN['date']] = [semi_idx]
        semi_idx+=1
      

    idx2lat = np.array(idx2lat)
    idx2lng = np.array(idx2lng)

    distance_matrix = np.zeros((len(idx2lat), len(idx2lat)))
    for i in range(total_N):
      for j in range(total_N):
        distance_matrix[i,j] = dist((idx2lat[i], idx2lng[i]), (idx2lat[j], idx2lng[j]))
    self.dit_mat = distance_matrix.copy()
    start, end = np.argmax(distance_matrix)%total_N, np.argmax(distance_matrix)//total_N
    # breakpoint()
    path = [start]
    np.place(distance_matrix, distance_matrix  == 0, float("inf"))
    
    for _ in range(total_N-1):
      before = path[-1]
      next_node = np.argmin(distance_matrix[before])
      path.append(next_node)
      
      distance_matrix[before,:] =float("inf")
      distance_matrix[:, before] = float("inf")
    
    for time, idx_per_time in Time2idx.items():
      plt.scatter(idx2lat[idx_per_time], idx2lng[idx_per_time])
    
    for i in range(len(idx2lat)):
      plt.annotate(i, (idx2lat[i], idx2lng[i]), fontsize = 8)

      
    # plt.plot(idx2lat[path], idx2lng[path])
      
      
    plt.show()
    breakpoint()
    plt.savefig(str(download_json) +".jpg")



    self.path = path
    self.Time2idx = Time2idx
    self.idx2lat = idx2lat
    self.idx2lng = idx2lng




def dist(loc1, loc2):
  """
  loc : tuple(lat, lon)
  """
  return distance.distance(loc1, loc2).m  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('datapath', help ='datapath')
  args = parser.parse_args()
  Co2path(args.datapath)
  