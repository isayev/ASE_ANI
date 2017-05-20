import h5py
import numpy as np
import platform

PY_VERSION = int(platform.python_version().split('.')[0]) > 3

class datapacker(object):
    def __init__(self, store_file, mode='w-', complib='gzip', complevel=6):
        """Wrapper to store arrays within HFD5 file
        """
        # opening file
        self.store = h5py.File(store_file, mode=mode)
        self.clib = complib
        self.clev = complevel

    def store_data(self, store_loc, **kwargs):
        """Put arrays to store
        """
        #print(store_loc)
        g = self.store.create_group(store_loc)
        for k, v, in kwargs.items():
            #print(type(v[0]))

            #print(k)
            if type(v[0]) is np.str_ or type(v[0]) is str:
                v = [a.encode('utf8') for a in v]

            g.create_dataset(k, data=v, compression=self.clib, compression_opts=self.clev)

    def cleanup(self):
        """Wrapper to close HDF5 file
        """
        self.store.close()


class anidataloader(object):

    ''' Contructor '''
    def __init__(self, store_file):
        self.store = h5py.File(store_file)

    ''' Default class iterator (iterate through all data) '''
    def __iter__(self):
        for g in self.store.values():
            dt = dict()
            for e in g.items():
                dt['name'] = e[0]
                dt['parent'] = g.name
                for k in e[1]:
                    v = e[1][k].value

                    if type(v[0]) is np.bytes_:
                        v = [a.decode('ascii') for a in v]
                    dt[k]=v
                yield dt

    getnextdata = __iter__

    ''' Iterates through a file stored as roman stores it '''
    def get_roman_data(self):
        for g in self.store.values():
            yield dict((d, g[d].value) for d in g)

    ''' Returns a list of all groups in the file '''
    def get_group_list(self):
        return [g for g in self.store.values()]

    ''' Allows interation through the data in a given group '''
    def iter_group(self,g):
        for e in g.items():
            dt = dict()
            dt['name'] = e[0]
            dt['parent'] = g.name
            for k in e[1]:
                v = e[1][k].value

                if type(v[0]) is np.bytes_:
                    v = [a.decode('ascii') for a in v]
                dt[k] = v
            yield dt

    ''' Returns the number of groups '''
    def size(self):
        return len(self.get_group_list())

    ''' Close the HDF5 file '''
    def cleanup(self):
        self.store.close()

