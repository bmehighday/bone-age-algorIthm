from multiprocessing.pool import Pool

from PIL import ImageDraw, Image
import numpy as np
import os

from tqdm import tqdm

from crop.random_crop import *
from crop.crop_bone import CropBone
import sys
import traceback

def get_bbox(pts):
    xmin = np.min([x[0] for x in pts])
    xmax = np.max([x[0] for x in pts])
    ymin = np.min([x[1] for x in pts])
    ymax = np.max([x[1] for x in pts])
    return np.array([[xmin, ymin], [xmax, ymax]])


def draw_pts(draw, cpts, color):
    apts = np.concatenate([cpts, cpts[:1]])
    for i, j in zip(apts[:-1], apts[1:]):
        draw.line([tuple(i), tuple(j)], fill=color, width=5)


def one_crop(crop_inputs):
    ######ADHOC, tired to change it elegently#####
    adhoc_dict = {}
    adhoc_dict['CHI_GU'] = 0
    adhoc_dict["RAO_GU"] = 0
    adhoc_dict["DI_I_ZHANG_GU"] = 0
    adhoc_dict["DI_III_ZHANG_GU"] = 0
    adhoc_dict["DI_V_ZHANG_GU"] = 1
    adhoc_dict['DI_I_YUAN_DUAN_ZHI_GU'] = 0
    adhoc_dict['DI_I_JIN_DUAN_ZHI_GU'] = 1
    adhoc_dict['DI_III_YUAN_DUAN_ZHI_GU'] = 2
    adhoc_dict['DI_III_ZHONG_JIAN_ZHI_GU'] = 3
    adhoc_dict['DI_III_JIN_DUAN_ZHI_GU'] = 4
    adhoc_dict['DI_V_YUAN_DUAN_ZHI_GU'] = 5
    adhoc_dict['DI_V_ZHONG_JIAN_ZHI_GU'] = 6
    adhoc_dict['DI_V_JIN_DUAN_ZHI_GU'] = 7
    adhoc_dict['GOU_GU'] = 0
    adhoc_dict['SAN_JIAO_GU'] = 0
    adhoc_dict['TOU_GU'] = 0
    adhoc_dict['YUE_GU'] = 0
    adhoc_dict['ZHOU_ZHUANG_GU'] = 0
    adhoc_dict['DA_DUO_JIAO_GU'] = 0
    adhoc_dict['XIAO_DUO_JIAO_GU'] = 0
    #### end of ADHOC #####
    points, pic_path, args = crop_inputs
    img = Image.open(pic_path)
    cropper = CropBone(img, points)
    # ["zi", "wan", "chi", "rao", "zhang35", "zhang1", "zhi"]
    crop_result = [cropper.get_name_gu(args.type)[adhoc_dict[args.idx]]]
    
    pic_name = pic_path.split('/')[-1]
    enchance_results = []
    ratio_list = []
    for i, one_bone_crop in enumerate(crop_result):
        cimg, cpts, ratio = one_bone_crop
        ratio_list.append(ratio)
        bbox = get_bbox(cpts)
        oimg = cimg.crop(np.round(bbox).astype(np.int).flatten())
        oimg = oimg.resize(tuple(np.round(np.array(args.size, dtype=np.float)).astype(np.int32)),
                           Image.BICUBIC)  # LANCZOS

        save_folder = args.output
        try:
            os.mkdir(save_folder)
        except:
            pass
        save_name = pic_name + '-CROP-{}.png'.format(args.idx)
        save_path = os.path.join(save_folder, save_name)
        oimg.save(save_path)
        one_bone_rtn = [(save_path, cimg, cpts)]
        enchance_results.append(one_bone_rtn)
    return enchance_results, ratio_list

def one_crop_(crop_inputs):
    try:
        one_crop(crop_inputs)
    except Exception as e:
        exc_type, exc_value, exc_traceback_obj = sys.exc_info()
        traceback.print_tb(exc_traceback_obj)
        print(e,1)

def batch_crop(crop_inputs, pool_size=8):
    pool = Pool(pool_size)
    pbar = tqdm(total=len(crop_inputs))
    for _ in pool.imap_unordered(one_crop_, crop_inputs):
        pbar.update(1)

    pool.close()
    pool.join()
