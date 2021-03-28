import mmcv
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mutils
from pycocotools.coco import COCO
import cv2

pred_path = '/ghome/zhuqi/mmlab/mmdetection/test.pkl'
anno_path = '/ghome/zhuqi/mmlab/mmdetection/data/DarkFace_coco_0.666/annotations/val_annotations.json'
img_path = '/ghome/zhuqi/mmlab/mmdetection/data/DarkFace_coco_0.666/val/'
num_class = 1

pred_data = mmcv.load(pred_path)
anno_data = mmcv.load(anno_path)
coco = COCO(anno_path)

color_list = [[0, 153, 51], [0, 153, 255], [255, 255, 0], [204, 102, 255],\
              [255, 0, 0], [204, 0, 255], [102, 255, 255], [255, 102, 102]]
ratio = 1
def decode_mask(mask_encode):
    mask_decode = mutils.decode(mask_encode)
    return mask_decode

def apply_mask(img, mask, color, alpha=.5):
    for c in range(3):
        img[:, :, c] = np.where(mask==1, img[:, :, c] *
                                (1 - alpha) + alpha * color[c],
                                img[:, :, c])
    return img

def draw_mask(img, mask_data, index):
    global color_list
    index %= 5
    mask_mat = decode_mask(mask_data)
    img = apply_mask(img, mask_mat, color_list[index], .5)
    return img

def draw_bbox(img, bbox_score_data, index, show_score=False):
    global color_list
    index %= 5
    print(bbox_score_data[:4])
    print(bbox_score_data[:4])
    test = list(map(lambda x: int(x), bbox_score_data[:4]))
    print(test)  # [742]
    print('8888888888888888')

    img = cv2.rectangle(img, (test[0], test[1]), (test[2], test[3]), color_list[index], 2)
    if show_score:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = ""
        if show_score:
            text += str(bbox_score_data[4])[:4]
        rec_width = len(text) * 6
        img = cv2.rectangle(img, (test[0], test[1]), (test[0]+rec_width, test[1]+10), color_list[index], -1)
        cv2.putText(img, text, (test[0], test[1]+8), font, 0.35, (0, 0, 0), 1)
    return img

def parse_img(img, anno):

    bbox_data = anno[0]
    print("#############")
    print(bbox_data)  # 这个地方没有读到信息

    for bbox in bbox_data:#, mask_data[cid]):
        #if bbox[4] > 0.05:
            #img = draw_bbox(img, bbox, cid, show_score=True)
            #img = draw_mask(img, mask, cid)
        print(bbox[4])
        if bbox[4] > 0.3:
            print('555555555555555555')
            img = draw_bbox(img, bbox, 1, show_score=True)
    return img


for index, (anno, pred) in enumerate(zip(anno_data['images'], pred_data)):

    img_name = anno['file_name'] #+ '.png'这里读到的file_name已经有后缀了
    print(img_name)
    print(anno)  # anno只包含annotationa中的image参数
    print(pred)  # 预测到的五个值[7.42111328e+02, 4.01652344e+02, 7.80128418e+02, 4.42174316e+02,9.34274435e-01],这些值应该是四个坐标

    img = cv2.imread(img_path + img_name) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(len(pred)) 
    print("#########################")

    img = parse_img(img, pred)


    save_path = '/ghome/zhuqi/mmlab/mmdetection/work_dirs/darkface_coco_0.66/batch8_gpu2_lr0.01/save_img/%s' % img_name
    plt.imsave(save_path, img)
    if index % 100 == 0:
        print("Saving %d/%d" % (index, len(pred_data)))