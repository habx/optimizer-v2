from os import makedirs
from os.path import isfile, join, exists
import hashlib
import functools
import pickle


class DecoratorCache(object):
    """
    A Decorator used to add cache capability to a function
    """
    def __init__(self, required_files, cache_folder):
        """

        :param required_files: list the python files that we want to take into account to
        know if we should recalculate the results of the function or cache them
        :param cache_folder: where to cache the results or where to retrieve them from
        """
        self.required_files = required_files
        self.cache_folder = cache_folder

    def __call__(self, func):

        @functools.wraps(func)
        def decorated(*args, **kwargs):
            """
            Decorates a function in order to add caching capabilities
            :param args: arguments of the function
            :param kwargs:  keywords arguments of the function
            :return: the function results
            """

            cache_file_path = self.get_file_path(func.__name__, args, kwargs)

            # check if a cache file exists for identical arguments and identical required files
            if not isfile(cache_file_path):
                results = func(*args, **kwargs)
                self.set_to_cache(results, cache_file_path)
            else:
                results = self.get_from_cache(cache_file_path)

            return results

        return decorated

    def get_file_path(self, func_name, func_args, func_kwargs):
        """
        Computes a file path from the hash of the function arguments and the content of the required files.
        To compute the hash we use a pickle dumps byte object

        :param func_name: name of the function that is decorated
        :param func_args: list of the arguments passed to the decorated function
        :param func_kwargs: list of the keywords arguments passed to the decorated function
        :return:
        """

        _hash = ""

        for elt in self.required_files:
            if isinstance(elt, str) and isfile(elt):
                with open(elt, "r", encoding="utf-8") as f:
                    data = f.read()
                    f.close()
                _hash += make_hash_from_dump(data)

        _hash += make_hash_from_dump(func_args)
        _hash += make_hash_from_dump(func_kwargs)

        file_path = make_hash_from_dump(_hash) + "_" + func_name + ".cache"

        return join(self.cache_folder, file_path)

    def set_to_cache(self, results, cache_file_path):
        """
        Saves the results to a cache file. If the cache folder does not exists, it is created.
        :param results: what should be cached
        :param cache_file_path: path of the file used to cache the results
        :return: the results (for convenience and consistency)
        """

        print('Saving to cache')

        if not exists(self.cache_folder):
            makedirs(self.cache_folder)

        with open(cache_file_path, 'wb') as cache:
            serializer = pickle.Pickler(cache)
            serializer.dump(results)

        return results

    @staticmethod
    def get_from_cache(cache_file_path):
        """
        Retrieves results from a specific cache file
        :param cache_file_path: where to look for the cached results
        :return: the retrieved results
        """

        print('Retrieving from cache')

        with open(cache_file_path, 'rb') as cache:
            deserializer = pickle.Unpickler(cache)
            results = deserializer.load()

        return results

# module functions


def hash_sha1(elt):
    return hashlib.sha1(str(elt).encode('utf-8')).hexdigest()


def make_hash_from_dump(obj):
    """
    Computes a hash of an object by hashing the bytestring computed by the pickle module
    :param obj: object to hash
    :return: a sha1 hash of the object bytestring
    """
    m = hashlib.sha1()
    m.update(pickle.dumps(obj))
    return m.hexdigest()
