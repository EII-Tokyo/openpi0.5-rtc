from hmac import new
import h5py

file_list = '../aloha_data/cut_data/2025.11.14_twist_two/episode_1.hdf5'
save_dir = ''

with h5py.File(file_list, 'r') as f_in:
    ds = f_in['action']
    cut_data = ds[:-120]
    print(cut_data.shape)

    for i in f_in.keys():
        print(i,type(f_in[i]))

    obs = f_in['observations']
    print(obs.keys())

    for k in obs.keys():
        print(k,type(obs[k]))
    img = obs['images']
    print(img.keys())

    for j in img.keys():
        print(j,type(img[j]))
        print(f'shape:{img[j].shape}')


# 裁减dataset
def cut_dataset(ds):
    if ds.shape[0] > 120:
        cut_data = ds[:-120]
    else:
        cut_data = ds[:]
    return cut_data

def copy_cut_dataset(out_file, name, ds):
    new_data = cut_dataset(ds)
    out_file.create_dataset(name, data=new_data)

print("准备写文件 output.hdf5 ...")
with h5py.File(file_list, 'r') as f_in, h5py.File('output.hdf5', 'w') as f_out:
    print("已经打开 output.hdf5")
    for key in f_in.keys():
        if isinstance(f_in[key], h5py.Dataset):
            copy_cut_dataset(f_out, key, f_in[key])
        else:
            new
print("写文件结束！")
