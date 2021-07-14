import numpy as np
import open3d as o3d

from open3d.visualization import Visualizer

if __package__ is '':
  import sys
  from os import path
  print(path.dirname( path.dirname( path.abspath(__file__) ) ))
  sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
  from dataset_utils import get_3D_Net_models
else:
  from .dataset_utils import get_3D_Net_models

"""Visualizer
http://www.open3d.org/docs/0.9.0/python_api/open3d.visualization.Visualizer.html#open3d.visualization.Visualizer
"""

def load_point_cloud_to_geometry(pc_file):
  pcd = o3d.io.read_point_cloud(pc_file)
  return pcd

    
def show_window(vis):
  vis.create_window(left=600, top=50)
  vis.run()
  
def show_visualizer():
  vis = o3d.visualization.Visualizer()
  vis.create_window(left=600)
  vis.run()

class PointCloudVisualizer:
  def __init__(self, point_cloud_file_list, custom_color=False):
    self.object_file_list = point_cloud_file_list
    self.custom_color = custom_color
    
    self.object_idx = -1
    self.object_num = len(self.object_file_list)
    self.current_pcd = self.get_next_pcd()
    
    # initialize visualizer
    self.vis = o3d.visualization.VisualizerWithKeyCallback()
    self.vis.create_window(left=600) # margin
    
    # add current PointCloud to window
    self.vis.add_geometry(self.current_pcd)
    
    # load render option from file : RenderOption
    self.render_option = self.vis.get_render_option()
    self.render_option.load_from_json("renderoption.json")
    
    # register callback function 
    self.vis.register_animation_callback(self.rotate_view)
    self.vis.register_key_callback(ord("N"), self.update_pcd)

    # activate window
    self.vis.run()
  
  def get_next_pcd(self):
    self.object_idx += 1
    if not self.object_idx < self.object_num:
      print("No more Objects ==> Exit Program!!")
      exit()
    
    # next object file from list    
    point_cloud_file = self.object_file_list[self.object_idx]
    print("Get Next Object: {}".format(point_cloud_file))
    
    # create PointCloud
    pcd = o3d.io.read_point_cloud(point_cloud_file)
  
    # if there is custom color
    if self.custom_color:
      self.update_point_cloud_color()

    return pcd
  
  def update_point_cloud_color(self):
    #TODO: if there is additional color
    pass
    
  def update_pcd(self, vis):
    vis.clear_geometries()
    self.current_pcd = self.get_next_pcd()
    vis.add_geometry(self.current_pcd)
    vis.reset_view_point(True)
    
    return False
  
  def rotate_view(self, vis):
    ctr = vis.get_view_control()
    ctr.rotate(5.0, 0.0)
    return False



if __name__=="__main__":
  object_point_cloud = get_3D_Net_models()
  whole_point_cloud = []
  for object_name in object_point_cloud.keys():
    whole_point_cloud += object_point_cloud[object_name]
  PointCloudVisualizer(whole_point_cloud)
  
  
    