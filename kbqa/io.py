import os
import hashlib
import pathlib
import pickle


def mkdir_p(dir_path):
    pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def save_pickle(data, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


class CacheManager(object):

    def __init__(self, cache_dir, enabled=True):
        self._cache_dir = cache_dir
        self._enabled = enabled

    def _get_cache_path(self, name):
        return os.path.join(
            self._cache_dir,
            "{}.pickle".format(hashlib.sha256(str.encode(name)).hexdigest()),
        )

    def load(self, name, fn, *args, **kwargs):
        if not self._enabled:
            return fn(*args, **kwargs)

        cache_path = self._get_cache_path(name)
        if not os.path.exists(cache_path):
            result = fn(*args, **kwargs)
            mkdir_p(self._cache_dir)
            save_pickle(result, cache_path)
        else:
            result = load_pickle(cache_path)

        return result
