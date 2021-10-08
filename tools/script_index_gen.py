import os

import cv2
import numpy as np


data_root_path = '/data/root/path/'
in_path = data_root_path + 'datasets/processed/UCSD_P2_256/'
frame_file_type = 'jpg';
clip_len = 16
overlap_rate = 0
skip_step = 1
clip_rng = clip_len*skip_step-1
overlap_shift = clip_len - 1
sub_dir_list = ['Train', 'Test'];

for sub_dir_idx in range(len(sub_dir_list)):
    sub_dir_name = sub_dir_list[sub_dir_idx]
    print(sub_dir_name)
    sub_in_path = in_path + sub_dir_name + '/'
    idx_out_path = in_path + sub_dir_name + '_idx/'
    os.mkdir(idx_out_path)

    v_list = os.listdir(sub_in_path + sub_dir_name + '*')
    for i in range(len(v_list)):
        v_name = v_list[0].split('/')[-1]
        print(v_name)
        frame_list = os.listdir(sub_in_path + v_name + '/*.' + frame_file_type)
        frame_num = len(frame_list)
        s_list = list(range(1, frame_num + 1, (clip_rng + 1 - overlap_shift)))
        e_list = s_list + clip_rng
        idx_val = e_list <= frame_num
        s_list = s_list[idx_val]
        e_list = e_list[idx_val]
        video_sub_dir_out_path = idx_out_path + v_name + '/'
        os.mkdir(video_sub_dir_out_path)
        for j in range(len(s_list)):
            idx = list(range(s_list[j], e_list[j] + 1, skip_step))
            np.savez(video_sub_dir_out_path + '_i' + str(j) + '.npz', v_name = v_name, idx = idx)