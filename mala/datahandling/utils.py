import os
import pickle
import time
from mala.common.parameters import printout

def sanitize(s: str):
  return s.replace('/', '<sl>').replace('\\', '<bsl>')

# decorator that caches the pickled result of a function call in a file
def pickle_cache(folder_name=None):
  def decorator(func):
    nonlocal folder_name
    if folder_name is None:
      folder_name = f"{func.__name__}_cached_values"
    def inner(*args, **kwargs):
      # stringify args and kwargs
      args_str = "_".join([str(arg) for arg in args])
      kwargs_str = "_".join([f"{key}={value}" for key, value in kwargs.items()])
      # sanitize strings
      args_str = sanitize(args_str)
      kwargs_str = sanitize(kwargs_str)
      
      file_name = f"{args_str}_{kwargs_str}.pkl"
      max_file_name_len = 250
      if len(file_name) > max_file_name_len:
        hashed_file_name = str(hash(file_name))
        hash_len = len(hashed_file_name)
        file_name = f'{file_name[:max_file_name_len-hash_len]}_{hashed_file_name}'
      file_name = f"{folder_name}/{file_name}.pkl"
      try:
        with open(file_name, 'rb') as f:
          # time the loading
          start = time.time()
          value = pickle.load(f)
          end = time.time()
          printout(
            f"Loaded cached value in {end-start:.2f} seconds from file {file_name}", min_verbosity=2
          )
          return value
      except FileNotFoundError:
        value = func(*args, **kwargs)
        # create directory if not exists
        os.makedirs(folder_name, exist_ok=True)
        with open(file_name, "wb") as f:
          pickle.dump(value, f)
        return value
      except Exception as e:
        print(f"Exception {e} occurred while trying to load cached value from {file_name}.")
        value = func(*args, **kwargs)
        # create directory if not exists
        os.makedirs(folder_name, exist_ok=True)
        with open(file_name, "wb") as f:
          pickle.dump(value, f)
        return value
    return inner
  return decorator

def binary_cache(folder_name=None):
  raise Exception("Not implemented yet.")
  def decorator(func):
    if folder_name is None:
      folder_name = f"{func.__name__}_cached_values"
    def inner(*args, **kwargs):
      # stringify args and kwargs
      args_str = "_".join([str(arg) for arg in args])
      kwargs_str = "_".join([f"{key}={value}" for key, value in kwargs.items()])
      # sanitize strings
      args_str = sanitize(args_str)
      kwargs_str = sanitize(kwargs_str)
      
      file_name = f"{args_str}_{kwargs_str}.pkl"
      max_file_name_len = 250
      if len(file_name) > max_file_name_len:
        hashed_file_name = str(hash(file_name))
        hash_len = len(hashed_file_name)
        file_name = f'{file_name[:max_file_name_len-hash_len]}_{hashed_file_name}'
      file_name = f"{folder_name}/{file_name}.pkl"
      try:
        with open(file_name, 'rb') as f:
          value = pickle.load(f)
          return value
      except FileNotFoundError:
        value = func(*args, **kwargs)
        # create directory if not exists
        os.makedirs(folder_name, exist_ok=True)
        with open(file_name, "wb") as f:
          pickle.dump(value, f)
        return value
      except Exception as e:
        print(f"Exception {e} occurred while trying to load cached value from {file_name}.")
        value = func(*args, **kwargs)
        # create directory if not exists
        os.makedirs(folder_name, exist_ok=True)
        with open(file_name, "wb") as f:
          pickle.dump(value, f)
        return value
    return inner
  return decorator

