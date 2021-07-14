

# relative import
if __package__ is '':
  import sys
  from os import path
  print(path.dirname( path.dirname( path.abspath(__file__) ) ))
  sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))
  from file_utils import get_dir_list, get_file_list, get_dir_name, get_file_name
else:
  from .file_utils import get_dir_list, get_file_list, get_dir_name, get_file_name



def get_3D_Net_models():
  point_cloud_root = "/home/raeyo/dataset/3D_Net/"

  data_dir_list = get_dir_list(point_cloud_root)
  model_dir_list = []
  test_dir_list = []

  for data_dir in data_dir_list:
    if "Model" in data_dir:
      model_dir_list.append(data_dir)
    elif "Test" in data_dir:
      test_dir_list.append(data_dir)

  object_point_cloud = {}

  for model_dir in model_dir_list:
    object_dir_list = get_dir_list(model_dir)
    for object_dir in object_dir_list:
      object_name = get_dir_name(object_dir)
      object_point_cloud.setdefault(object_name, [])
      for file_path in get_file_list(object_dir):
        if file_path.endswith(".ply"):
          if not file_path in object_point_cloud[object_name]:
            object_point_cloud[object_name].append(file_path) 

  for object_name in object_point_cloud.keys():
    print("Object: {} has {} files".format(object_name, len(object_point_cloud[object_name])))

  print("Whole Data Set has {} files".format(sum(map(len, object_point_cloud.values()))))
  
  return object_point_cloud  
  
if __name__=="__main__":
  object_files = get_3D_Net_models()
  