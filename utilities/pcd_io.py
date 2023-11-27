# Copyright (c) 2018, Daniel Maturana and contributors
#
# This file originated from the `pypcd` package:
#
#   https://github.com/dimatura/pypcd

import numpy as np
import re
import warnings
import struct
# import copy
import numpy as np
import warnings
import lzf

def _build_dtype(metadata):
    """ build numpy structured array dtype from pcl metadata.
    note that fields with count > 1 are 'flattened' by creating multiple
    single-count fields.
    TODO: allow 'proper' multi-count fields.
    """
    fieldnames = []
    typenames = []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
            typenames.extend([np_type]*c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype

numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
numpy_type_to_pcd_type = dict(numpy_pcd_type_mappings)
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)



def parse_header(lines):
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        ln = ln.replace(' _ ',' s ',1)
        ln = ln.replace(' _ ',' m ',1)
        match = re.match('(\w+)\s+([\w\s\.]+)', ln)
        if not match:
            warnings.warn("warning: can't understand line: %s" % ln)
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = list(map(int, value.split()))
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = list(map(float, value.split()))
        elif key == 'data':
            metadata[key] = value.strip().lower()
        # TODO apparently count is not required?
    # add some reasonable defaults
    if 'count' not in metadata:
        metadata['count'] = [1]*len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata


def point_cloud_from_fileobj(f):
    """ parse pointcloud coming from file object f
    """
    header = []
    while True:
        ln = f.readline().strip()
        if not isinstance(ln, str):
            ln = ln.decode('utf-8')
        header.append(ln)
        if ln.startswith('DATA'):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break
    if metadata['data'] == 'ascii':
        pc_data = parse_ascii_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary':
        pc_data = parse_binary_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary_compressed':
        pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
    else:
        print('DATA field is neither "ascii" or "binary" or\
                "binary_compressed"')
    return metadata, pc_data

def parse_ascii_pc_data(f, dtype, metadata):
    return np.loadtxt(f, dtype=dtype, delimiter=' ')


def parse_binary_pc_data(f, dtype, metadata):
    rowstep = metadata['points']*dtype.itemsize
    # for some reason pcl adds empty space at the end of files
    buf = f.read(rowstep)
    return np.fromstring(buf, dtype=dtype)


def parse_binary_compressed_pc_data(f, dtype, metadata):
    """ Parse lzf-compressed data.
    Format is undocumented but seems to be:
    - compressed size of data (uint32)
    - uncompressed size of data (uint32)
    - compressed data
    - junk
    """
    fmt = 'II'
    compressed_size, uncompressed_size =\
        struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    compressed_data = f.read(compressed_size)
    # TODO what to use as second argument? if buf is None
    # (compressed > uncompressed)
    # should we read buf as raw binary?
    buf = lzf.decompress(compressed_data, uncompressed_size)
    if len(buf) != uncompressed_size:
        raise IOError('Error decompressing data')
    # the data is stored field-by-field
    pc_data = np.zeros(metadata['width'], dtype=dtype)
    ix = 0
    for dti in range(len(dtype)):
        dt = dtype[dti]
        bytes = dt.itemsize * metadata['width']
        column = np.fromstring(buf[ix:(ix+bytes)], dt)
        pc_data[dtype.names[dti]] = column
        ix += bytes
    return pc_data

def load_pcd(fname, with_header=False):
    with open(fname, 'rb') as f:
        metadata, pc_data = point_cloud_from_fileobj(f)
        points = np.array([pc_data["x"], pc_data["y"], pc_data["z"]], dtype=np.float32).T
        return points

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d, Axes3D
    points = load_pcd('data/bunny.pcd')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c='r')
    plt.show()

