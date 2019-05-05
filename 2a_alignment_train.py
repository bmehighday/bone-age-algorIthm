from common.head import *
from common.bone_alignment import BoneAligner
from common.util import init_logging, load_config, try_make_dir,load_csv,load_list


def main(args, config):
    common_dir = config['alignment']['common']['dir']
    logging.debug("set common_dir at: {}".format(common_dir))
    try_make_dir(common_dir)
    bone_aligner = BoneAligner(point_num=config['alignment']['common']['point_num'],
                               img_shape=config['alignment']['common']['image_shape'],
                               common_dir=common_dir)

    bone_aligner.init_model()

    image_name_list_path = config['annos']['img_name_csv']
    align_point_json_csv_path = config['annos']['align_point_json_csv']
    point_align_id_list_path = config['annos']['point_align_id_list']
    bone_aligner.load_data(image_name_csv=load_csv(image_name_list_path),
                           img_dir=config['input_img_dir'],
                           point_jsons_csv=load_csv(align_point_json_csv_path),
                           point_align_id_list=load_list(point_align_id_list_path))

    bone_aligner.train(config['alignment']['train'])


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
