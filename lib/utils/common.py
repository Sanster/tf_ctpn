import os

def check_dir(dir_path):
  """
  create dir if dir not exist
  """
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
