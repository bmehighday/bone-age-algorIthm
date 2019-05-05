from common.head import *
from common.bone_classifier import BoneClassifier
from common.util import init_logging, load_config, try_make_dir,load_csv


def train_single(args, config, bone_name):
    bone_setting = config['bone_classifier']['common']['bone_setting'][bone_name]
    common_dir = os.path.join(config['bone_classifier']['common']['dir'], bone_name)
    logging.debug("set common_dir at: {}".format(common_dir))
    try_make_dir(common_dir)
    bone_classifier = BoneClassifier(bone_name=bone_name,
                                     cls_num=bone_setting['class_num'],
                                     img_shape=bone_setting['image_shape'],
                                     common_dir=common_dir)

    bone_classifier.init_model()

    image_name_list_path = config['annos']['img_name_csv']
    bone_label_csv_path = config['annos']['bone_label_csv']
    crop_image_csv = os.path.join(config['crop']['common']['dir'], config['crop']['result']['crop_img_csv'])

    bone_classifier.load_data(image_name_csv=load_csv(image_name_list_path),
                              crop_image_csv=load_csv(crop_image_csv),
                              label_csv=load_csv(bone_label_csv_path))

    bone_classifier.train(config['bone_classifier']['train'])

def main_train(args, config):
    bone_names = list(config['bone_classifier']['common']['bone_setting'].keys())
    for bone_name in bone_names:
        logging.info("train @ {}".format(bone_name))
        train_single(args, config, bone_name)


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
    main_train(args, config)
