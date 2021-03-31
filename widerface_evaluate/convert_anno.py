import mmcv
import numpy as np
import os
import xml.etree.ElementTree as ET
import argparse
from nms import py_weighted_nms
import tqdm
import time

parser = argparse.ArgumentParser(description='Convert MMDet prediction format to WiderFace one.')
parser.add_argument('config', default='widerface_cfg/tinaface.py', help='model config name')
parser.add_argument('--nms', default=False)
parser.add_argument('--nms_thres', default=0.45, type=float)
parser.add_argument('--vote_thres', default=0.5, type=float)
args = parser.parse_args()

model_name = args.config.split('/')[-1].split('.py')[0]
output_dir = 'widerface_evaluate/widerface_txt/'
os.makedirs(output_dir, exist_ok=True)

anno_path = 'work_dirs/%s/result.pkl' % model_name
det_result = mmcv.load(anno_path)

anno_path = 'data/WIDERFace/val.txt'
with open(anno_path, 'r') as f:
    anno_list = f.readlines()

def parse_each_annotation(param):
    start_time = time.time()
    anno_name = param[0].strip(); det = param[1]
    xml_folder_path = 'data/WIDERFace/WIDER_val/Annotations/'
    xml_path = xml_folder_path + anno_name + '.xml'
    tree = ET.parse(xml_path)
    root = tree.getroot()
    folder_name = root.find('folder').text
    os.makedirs(output_dir + folder_name, exist_ok=True)
    dets = det[0]
    origin_shape = dets.shape
    if args.nms:
        dets = py_weighted_nms(dets, args.nms_thres, args.vote_thres)
    save_path = output_dir + folder_name + '/' + anno_name + '.txt'
    bbox_list = []
    for ii in range(dets.shape[0]):
        box = dets[ii]
        x = int(box[0])
        y = int(box[1])
        w = int(box[2]) - int(box[0])
        h = int(box[3]) - int(box[1])
        confidence = str(box[4])
        line = f'{x} {y} {w} {h} {confidence}\n'
        bbox_list.append(line)
    with open(save_path, 'w') as f:
        f.write(anno_name + '.jpg' + '\n')
        f.write(str(len(bbox_list)) + '\n')
        for each_det in bbox_list:
            f.write(each_det)
    end_time = time.time()
    print("Parsing ", origin_shape, " Using %g s" % (end_time - start_time))
    return None


def parse_annotation(anno_list, det_result):
    from multiprocessing import cpu_count
    from multiprocessing.pool import Pool
    pool = Pool(2 * cpu_count() // 3)
    _ = pool.map(parse_each_annotation, zip(anno_list, det_result))
    pool.close()
    # for (anno, det) in zip(anno_list, det_result):
    #     anno = anno.strip()
    #     parse_each_annotation(anno, det)

parse_annotation(anno_list, det_result)