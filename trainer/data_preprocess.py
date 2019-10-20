"""
Recommended partitioning of images into training, validation, testing sets.
Images 1-162770 are training, 162771-182637 are validation, 182638-202599 are testing.
"""

import os

environments = ['local', 'cloud']
env = environments[0]
use_tiny = True

if env == 'local':
    # Path for local env.
    dir_path = os.path.join(os.path.expanduser("~"), "Dataset/celeba")
    partition_path = os.path.join(dir_path, "list_eval_partition.csv")
    img_dir_path = os.path.join(dir_path, "img_align_celeba")
    img_url = "~/Dataset/celeba/img_align_celeba/"
    train_flist_path = os.path.join(dir_path, "train.flist")
    valid_flist_path = os.path.join(dir_path, "valid.flist")
else:
    # Path for cloud env.
    gs_path = "gs://inpaint-dataset/celeba"
    dir_path = os.path.join(os.path.expanduser("~"), "Dataset/celeba")
    partition_path = os.path.join(dir_path, "list_eval_partition.csv")
    img_dir_path = "inpaint-dataset/celeba/img_align_celeba/"
    img_url = "https://storage.cloud.google.com/" + img_dir_path
    train_flist_path = os.path.join(dir_path, "celeba_train_cloud.flist")
    valid_flist_path = os.path.join(dir_path, "celeba_valid_cloud.flist")

data_lists = [[], [], []]

with open(partition_path, 'r', encoding='UTF-8') as f:
    next(f)
    for line in f:
        lines = line.strip().split(',')
        absolute_path = os.path.join(img_url, lines[0])
        data_lists[int(lines[1])].append(absolute_path)

with open(train_flist_path, 'w', encoding='UTF-8') as f:
    if use_tiny:
        for line in data_lists[0][0:20000]:
            f.write(line + '\n')
    else:
        for line in data_lists[0]:
            f.write(line + '\n')

with open(valid_flist_path, 'w', encoding='UTF-8') as f:
    if use_tiny:
        for line in data_lists[1][0:2000]:
            f.write(line + '\n')
        for line in data_lists[2][0:2000]:
            f.write(line + '\n')
    else:
        for line in data_lists[1]:
            f.write(line + '\n')
        for line in data_lists[2]:
            f.write(line + '\n')


