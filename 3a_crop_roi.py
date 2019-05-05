from common.head import *
from common.util import init_logging, load_config, try_make_dir,load_csv,load_list, map_csv_to_dict, save_csv
from crop.random_crop import UniformDist
from crop.cropper import batch_crop
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, Process, Queue, cpu_count
import traceback
type_dict = {
    'zhang1' :  (160, 128),
    'zhang35':  (128, 128),
    'zhi':      (128, 128),
    'chi':      (100, 120),
    'rao':      (160, 128),
    'wan':      (160, 128),
    'zi':       (128, 128)
}

def get_crop_args():
    #init
    args = argparse.Namespace()
    args.count = 1
    args.rotate = UniformDist(0, 0)
    args.stretch = UniformDist(1, 1)
    args.translate = UniformDist(0, 0)

    return args

def main(cui_args, config):
    # load args

    # init
    idx_type_map = config['crop']['common']['index_type_map']

    bone_names = list(idx_type_map.keys())
    img_name_list = [row['img_name'] for row in load_csv(config['annos']['img_name_csv'])]
    align_point_json_csv = load_csv(config['annos']['align_point_json_csv'])
    align_point_json_dict = map_csv_to_dict(align_point_json_csv, ['img_name'])
    point_align_id_list = load_list(config['annos']['point_align_id_list'])

    crop_img_csv = []
    crop_img_csv_title = ["img_name", "bone_name", "crop_img_path"]
    for bone_name in bone_names:
        logging.info("CROP @ {}".format(bone_name))
        args = get_crop_args()
        args.idx = bone_name
        crop_output_dir = os.path.join(config['crop']['common']['dir'], bone_name)
        args.output = crop_output_dir
        try_make_dir(args.output)
        args.type = idx_type_map[bone_name]
        args.size = type_dict[args.type]
        crop_inputs = []
        for img_name in tqdm(img_name_list, desc='load jsons'):
            pic_path = os.path.join(config['input_img_dir'], img_name)
            json_path = align_point_json_dict[img_name]['json_path']
            points = []
            with open(json_path) as f:
                j = json.load(f)
            for point_align_id in point_align_id_list:
                points.append((j[point_align_id]['x'], j[point_align_id]['y']))
            crop_inputs.append((points, pic_path, args))
            crop_img_csv.append({'img_name': img_name, "bone_name": bone_name,
                                       'crop_img_path': os.path.join(crop_output_dir, img_name + '-CROP-{}.png'.format(bone_name))})
        with ProcessPoolExecutor(max_workers=128) as executor:
            try:
                executor.submit(batch_crop, crop_inputs, pool_size=48)
            except Exception as e:
                print(e)
    crop_img_csv_path = os.path.join(config['crop']['common']['dir'], config['crop']['result']['crop_img_csv'])
    save_csv(crop_img_csv_path, crop_img_csv, crop_img_csv_title)


def get_cui_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, default='./config.json',
                        help="config json")
    parser.add_argument("--debug", action="store_true",
                        help="debug mode")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_cui_args()
    config = load_config(args.config)
    init_logging(args.debug)
    main(args, config)
