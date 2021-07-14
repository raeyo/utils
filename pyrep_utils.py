import os
from os.path import join, isdir, isfile
from os import listdir
import math
from numpy.lib.npyio import savez_compressed
import copy

from pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects import Shape

import numpy as np

# relative import
if __package__ is '' or __package__ is None:
  import sys
  from os import path
  print(path.dirname( path.dirname( path.abspath(__file__) ) ))
  sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
  from file_utils import *
  from dataset_utils import get_3D_Net_models
else:
  from .file_utils import *
  from .dataset_utils import get_3D_Net_models



class BaseEnv:
  def __init__(self, scene_file="", headless=False):
    self.pr = PyRep()
    # Launch the application with a scene file in headless mode
    self.pr.launch(scene_file, headless=headless) 
    
    self.pr.start()  # Start the simulation

  def step(self):
    self.pr.step()

  def stop(self):
    self.pr.stop()
    self.pr.shutdown()

  def load_scene_object_from_file(self, file_path):
    respondable = self.pr.import_model(file_path)
    visible = respondable.get_objects_in_tree(exclude_base=True)[0]
    return SceneObject(respondable_part=respondable, visible_part=visible)


class SceneObject(object):
  def __init__(self, respondable_part: Shape, visible_part: Shape):
    self.respondable = respondable_part
    self.visible = visible_part
    name = self.respondable.get_name().replace("_respondable", "")
    try:
      int(name[-1])
      name = name[:-1]
    except:
      pass
    self.name = name

    self.visible.set_pose(self.respondable.get_pose())
    self.visible.set_parent(self.respondable)

    """Shape Property
    Shapes are collidable, measurable, detectable and renderable objects. This means that shapes:

    collidable: can be used in collision detections against other collidable objects.
    measurable: can be used in minimum distance calculations with other measurable objects.
    detectable: can be detected by proximity sensors.
    renderable: can be detected by vision sensors.
    
    Dynamic shapes will be directly influenced by gravity or other constraints
    Respondable shapes influence each other during dynamic collision

    """
    self._is_collidable = True 
    self._is_measurable = True
    self._is_detectable = True
    self._is_renderable = True
    self._is_dynamic = False
    self._is_respondable = False

    self.initialize_respondable()
    self.initialize_visible()

  def initialize_visible(self):
    self.visible.set_collidable(False)
    self.visible.set_measurable(False)
    self.visible.set_detectable(True)
    self.visible.set_renderable(True)
    self.visible.set_dynamic(False)
    self.visible.set_respondable(False)
    self.set_emission_color(self.visible, [1, 1, 1])

  def initialize_respondable(self):
    self.respondable.set_collidable(True)
    self.respondable.set_measurable(True)
    self.respondable.set_detectable(False)
    self.respondable.set_renderable(False)
    self.respondable.set_dynamic(True)
    self.respondable.set_respondable(True)
    self.set_transparency(self.respondable, [0])

  def set_collidable(self, is_collidable):
    self.respondable.set_collidable(is_collidable)
    self._is_collidable = is_collidable

  def set_measurable(self, is_measurable):
    self.respondable.set_measurable(is_measurable)
    self._is_measurable = is_measurable
  
  def set_detectable(self, is_detectable):
    self.visible.set_detectable(is_detectable)
    self._is_detectable = is_detectable
  
  def set_renderable(self, is_renderable):
    self.visible.set_renderable(is_renderable)
    self._is_renderable = is_renderable
  
  def set_respondable(self, is_respondable):
    self.respondable.set_respondable(is_respondable)
    self._is_respondable = is_respondable

  def set_dynamic(self, is_dynamic):
    self.respondable.set_dynamic(is_dynamic)
    self._is_dynamic = is_dynamic

  def set_position(self, position, relative_to=None):
    self.respondable.set_position(position, relative_to)
  def get_position(self, relative_to=None):
    return self.respondable.get_position(relative_to)
  
  def set_orientation(self, orientation, relative_to=None):
    self.respondable.set_orientation(orientation, relative_to)
  def get_orientation(self, relative_to=None):
    return self.respondable.get_orientation(relative_to)
  
  def set_pose(self, pose, relative_to=None):
      self.respondable.set_pose(pose, relative_to)
  def get_pose(self, relative_to=None):
      return self.respondable.get_pose(relative_to)

  def set_name(self, name):
    self.visible.set_name("{}_visible".format(name))
    self.respondable.set_name("{}_respondable".format(name))

  def get_name(self):
    return self.respondable.get_name(), self.visible.get_name()

  def get_handle(self):
    return self.respondable.get_handle()

  def check_distance(self, object):
    return self.respondable.check_distance(object)

  def remove(self):
    self.visible.remove()
    self.respondable.remove()

  def save_model(self, save_path):
    self.respondable.set_model(True)
    self.respondable.save_model(save_path)

  @staticmethod
  def set_emission_color(object, color):
    """set object emission color

    Args:
        object (Shape): [PyRep Shape class]
        color (list): [3 value of rgb] 0 ~ 1
    """
    sim.simSetShapeColor(
    object.get_handle(), None, sim.sim_colorcomponent_emission, color)
  
  @staticmethod
  def set_transparency(object, value):
    """set object transparency

    Args:
        object (Shape): [PyRep Shape class]
        value (list): [list of 1 value] 0 ~ 1
    """
    sim.simSetShapeColor(
    object.get_handle(), None, sim.sim_colorcomponent_transparency, value)


def convex_decompose(obj):
  return obj.get_convex_decomposition(morph=False,
                                      use_vhacd=True)

def convert_YCB_to_SceneObject(ycb_root):
  # initialize coppeliasim environment
  env = BaseEnv(headless=True)
  env.step()
  
  ycb_list = [join(ycb_root, p) for p in os.listdir(ycb_root)]

  for target_ycb_folder in ycb_list:
    env.step()
    
    # In coppeliasim "-" sometimes make trouble
    target_name = target_ycb_folder.split('/')[-1]
    target_name = target_name.replace("-", "_")
    
    # tsdf file has row quality
    if "google_16k" in os.listdir(target_ycb_folder):
      mesh_type = "google_16k"
    else:
      mesh_type = "tsdf"
    
    save_root = "./ycb_models"
    save_path = join(save_root, mesh_type, "{}.ttm".format(target_name))
    
    if mesh_type == "google_16k":
      mesh_file = os.path.join(target_ycb_folder, "google_16k", "textured.dae")
    elif mesh_type == "tsdf":
      continue
      mesh_file = os.path.join(target_ycb_folder, "tsdf", "textured.obj")
    
    print("Save {} \n\t===> {}".format(target_name, save_path))
    
    visible = Shape.import_shape(filename=mesh_file,
                                scaling_factor=1)
    
    try:
      respondable = convex_decompose(visible)
    except TimeoutError:
      print("Too long to convex decomposition {}".format(target_name))
      continue
    
    
    obj = SceneObject(visible, respondable)
    
    obj.set_name(target_name)
    obj.save_model(save_path)
    
    obj.remove()
    del obj

    env.step()

  env.stop()

def convert_3DNet_to_SceneObject():
  object_point_cloud = get_3D_Net_models()
  
  conversion_info = {}
  
  save_root = "3DNet_scene_model"
  check_and_create_dir(save_root)
  
  # initialize coppeliasim environment
  env = BaseEnv(headless=True)
  env.step()
  
  for object_name in object_point_cloud.keys():
    conversion_info.setdefault(object_name, [])
    
    point_cloud_list = object_point_cloud[object_name]
    target_root = join(save_root, object_name)
    check_and_create_dir(target_root)
    
    for idx, point_cloud_file in enumerate(point_cloud_list):
      target_name = "{}_{}".format(object_name, idx)
      save_path = join(target_root, "{}.ttm".format(target_name))
      if convert_mesh_to_scene_object(env, point_cloud_file, save_path, target_name=target_name):
        conversion_info[object_name].append({
          "orginal_file": point_cloud_file,
          "model_file": relative_path_to_abs_path(save_path)
        })
  
  env.stop()
  save_dic_to_json(conversion_info, "3DNet_conversion_info.json")
  

def convert_mesh_to_scene_object(env, mesh_file, save_path, target_name=None):
  if target_name is None:
    target_name = get_file_name(mesh_file)
  
  print("Save {} \n\t===> {}".format(target_name, save_path))
  
  visible = Shape.import_shape(filename=mesh_file,
                              scaling_factor=1)
  
  try:
    respondable = convex_decompose(visible)
  except TimeoutError:
    print("Too long to convex decomposition {}".format(target_name))
    return False
  
  obj = SceneObject(visible, respondable)
  
  obj.set_name(target_name)
  obj.save_model(save_path)
  
  obj.remove()
  del obj
  env.step()
  
  return True

def visualize_scene_objects(model_list):
  """Load Scene Objects from saved model files
  """
  print("Show {} Scene Models".format(len(model_list)))
  
  # set whole grid
  grid_size = math.sqrt(len(model_list))
  if grid_size.is_integer():
    grid_size = int(grid_size)
  else:
    grid_size = int(grid_size) + 1

  grid = range(grid_size)
  grid_x, grid_y = np.meshgrid(grid, grid)
  object_xy_idx = zip(grid_x.flatten(), grid_y.flatten())

  env = BaseEnv()
  start_xy = (-2.25, -2.25)
  step_size = 5 / grid_size
  for model_path, xy_idx in zip(model_list, object_xy_idx):
    obj = env.load_scene_object_from_file(model_path)
    obj.set_dynamic(False)
    obj.set_renderable(False)

    obj_name = os.path.splitext(model_path)[0].split('/')[-1]
    print("Load {}".format(obj_name))

    obj_pos = obj.get_position()
    obj_pos[0] = start_xy[0] + xy_idx[0]*step_size
    obj_pos[1] = start_xy[1] + xy_idx[1]*step_size
    obj.set_position(obj_pos)
    
    env.step()

  try:
    while True:
      env.step()
  except:
    env.stop()
  


if __name__=="__main__":
  convert_3DNet_to_SceneObject()
  