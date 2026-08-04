import os
import glob
import h5py

from scripts.h5py_test import cut_dict

src = '../aloha_data/aloha_stationary/1.twist_one+looking/*.hdf5'
dst = '../aloha_data/cut_data'
os.makedirs(dst, exist_ok=True)

def make_path(src, dst):
    src_list = glob.glob(src)
    for src_path in src_list:
        name = os.path.basename(src_path)
        dst_path = os.path.join(dst, name)
        revise_data(src_path, dst_path)

def visit_dataset(fin, prefix=''):
    cut_dict = {}
    for key in fin:
        item = fin[key]
        key_name = f'{prefix}/{key}' if prefix else key
        if isinstance(item, h5py.Group):
            cut_dict.update(visit_dataset(item, key_name)) 
        if isinstance(item, h5py.Dataset):
            if item.shape[0] > 50:
                data = item[:-50]
            else:
                data = item        
            cut_dict[key_name] = data
    return cut_dict

def revise_data(src_path, dst_path):
    with h5py.File(src_path, 'r') as fin, h5py.File(dst_path, 'w') as fout:
        cut_dict = visit_dataset(fin, prefix='')
        for key, data in cut_dict.items():
            key_path, key_name = os.path.split(key)
            if key_path:
                group = fout.require_group(key_path) 
            else:
                group = fout
            group.create_dataset(key_name, data=data)
make_path(src, dst)