# agents/_cache.py

import time
from functools import wraps

_cache: dict[str, tuple[float, dict]] = {}

def cached(ttl_seconds: int = 1800):
  def decorator(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
      key = f"{fn.__name__}:{args}:{kwargs}"
      now = time.time()
      if key in _cache:
        ts, val = _cache[key]
        if now - ts < ttl_seconds:
          return val
      val = fn(*args, **kwargs)
      _cache[key] = (now, val)
      return val
    return wrapper
  return decorator