import os
import json
import numpy as np
import StringIO


def gen_config(args):

    if args.seq != '':
        # generate config from a sequence name
        seq_home = '../../tracker_benchmark-master/data'
        save_home = '../result_fig'
        result_home = '../result'

        seq_name = args.seq
        img_dir = os.path.join(seq_home, seq_name, 'img')
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        if seq_name == "David":
            img_list = [os.path.join(img_dir,x) for x in img_list[299:770]]
        elif seq_name == "Football1":
            img_list = [os.path.join(img_dir, x) for x in img_list[0:74]]  # actually is 0-73
        elif seq_name == "Freeman3":
            img_list = [os.path.join(img_dir, x) for x in img_list[0:460]]
        elif seq_name == "Freeman4":
            img_list = [os.path.join(img_dir, x) for x in img_list[0:283]]
        elif seq_name == "Diving":
            img_list = [os.path.join(img_dir, x) for x in img_list[0:215]]
        elif seq_name == "Tiger1":
            img_list = [os.path.join(img_dir, x) for x in img_list[5:354]]  # 349 frames is 5-353
        else:
            img_list = [os.path.join(img_dir, x) for x in img_list]

        # gt = np.loadtxt(gt_path,delimiter=',')
        print(args.seq)
        s = open(gt_path).read().replace('\t', ',')
        s = s.replace(' ', ',')
        gt = np.loadtxt(StringIO.StringIO(s), delimiter=',')
        init_bbox = gt[0]
        
        savefig_dir = os.path.join(save_home, seq_name)
        result_dir = os.path.join(result_home)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, 'result.json')

    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json,'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None
        
    if args.savefig:
        if not os.path.exists(savefig_dir):
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''

    return img_list, init_bbox, gt, savefig_dir, args.display, result_path
