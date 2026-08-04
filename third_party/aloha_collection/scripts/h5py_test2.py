import glob
import h5py
import os

file_list = glob.glob('../aloha_data/aloha_stationary/1.twist_many+looking/*.hdf5')

save_dir = '../aloha_data/cut_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
input_path = file_list[0]


def visit_group(group, prefix=''):
    cut_dict = {}  # 使用字典存储数据
    for key in group:
        item = group[key]
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5py.Group):
            sub_dict = visit_group(item, full_key)
            cut_dict.update(sub_dict)
        elif isinstance(item, h5py.Dataset):
            # 读取数据并处理
            if item.shape[0] > 50:
                # 如果第一维长度 > 50，截取前 N-50 个
                data = item[:-50]
            else:
                # 否则保留全部数据
                data = item[:]
            # 将处理后的数据存储到字典中，使用 full_key 作为键
            cut_dict[full_key] = data
            # print(full_key)
    return cut_dict
with h5py.File(input_path, 'r') as fin, h5py.File(new_path, 'w') as fout:
    cut_dict = visit_group(fin, prefix='')
    for key in cut_dict:
        print(key)
        # print(f"{key}:{cut_dict[key]}")
        group_path, group_name = os.path.split(key)
        if group_path:
            group = fout.require_group(group_path)
        else:
            group = fout
        fout.create_dataset(group_name, cut_dict[key])
