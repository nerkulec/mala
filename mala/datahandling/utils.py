import os
import pickle

def sanitize(s: str):
  return s.replace('/', '<sl>').replace('\\', '<bsl>')

# decorator that caches the pickled result of a function call in a file
def pickle_cache(func):
  folder_name = f"{func.__name__}_cached_values"
  def inner(*args, **kwargs):
    # stringify args and kwargs
    args_str = "_".join([str(arg) for arg in args])
    kwargs_str = "_".join([f"{key}={value}" for key, value in kwargs.items()])
    # sanitize strings
    args_str = sanitize(args_str)
    kwargs_str = sanitize(kwargs_str)
    
    file_name = f"{folder_name}/{args_str}_{kwargs_str}.pkl"
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
  return inner

