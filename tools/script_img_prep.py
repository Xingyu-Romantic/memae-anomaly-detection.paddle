import os
import argparse

import cv2
import numpy as np

def im2double(im):

    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def trans_img2img(inpath, outpath, opts):
    os.mkdir(outpath)
    img_list = os.listdir(inpath + '/' + '*.' + opts.img_type)
    img_name = img_list[0].split('/')[-1]
    img = cv2.imread(inpath + '/' + img_name);
    H, W = img.shape[:2]

    print(inpath)

    img_list = os.listdir(inpath + '/*.' + opts.img_type)

    for i in range(len(img_list)):
        img_name = img_list[0].split('/')[-1]
        img = cv2.imread(inpath + '/' + img_name)
        if opts.is_gray and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if not opts.is_gray and img.shape[2] == 3:
            img = np.kron(np.ones((1, 1, 3)), img)
        if opts.outsize != None:
            img = cv2.resize(img, opts.outsize)
        cv2.imwrite(img, [outpath, str(i), '.jpg'])

def trans_img2label(inpath, idx, outpath):
    gt_in_path = inpath + 'Test' + str(idx) + '_gt/'
    print(gt_in_path)
    os.mkdir(gt_in_path)
    file_list = os.listdir(gt_in_path + '/*.bmp')
    l = np.zeros((1, len(file_list)))
    for j in range(len(file_list)):
        name = file_list[0].split('/')[-1]
        file_path = gt_in_path + name
        img = cv2.imread(file_path)
        img = im2double(img)
        f = np.sum(img[:])
        if f < 1:
            l[j] = 0
        else:
            l[j] = 1
    np.save(outpath + 'Test' + str(idx), '.npy', l)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def main(opts):

    data_root_path = './'
    in_path = data_root_path + 'datasets/UCSD_Anomaly_Dataset.v1p2/UCSDped2/'
    out_path = data_root_path + 'datasets/processed/UCSD_P2_256/'
    

    os.mkdir(out_path)

    sub_dir_list = ['Train', 'Test'];
    file_num_list = [16, 12];

    for subdir_idx in range(len(sub_dir_list)):
        subdir_file_num = file_num_list[subdir_idx]
        subdir_name = sub_dir_list[subdir_idx]
        subdir_in_path = in_path + subdir_name + '/'
        subdir_out_path = out_path + subdir_name + '/'
        for i in range(subdir_file_num):
            v_name = subdir_name + str(i)
            v_path = subdir_in_path + v_name + '/'
            v_out_path = subdir_out_path + v_name + '/'
            os.mkdir(v_out_path)
            print(v_path)
            print(v_out_path)
            trans_img2img(v_path, v_out_path, opts)
    gt_in_path = in_path + 'Test/'
    gt_out_path = out_path + 'Test_gt/'

    os.mkdir(gt_out_path)
    for i in range(file_num_list[1]):
        trans_img2label(gt_in_path, gt_out_path)


if __name__ == '__main__':
    opt = argparse.ArgumentParser()
    opt.add_argument('--is_gray', help='rgb 2 gray', type=str2bool, default=True)
    opt.add_argument('--maxs', type = int, default='320')
    opt.add_argument('--outsize', type=list, default=[256, 256])
    opt.add_argument('--img_type', type=str, default='tif')


    main(opt)