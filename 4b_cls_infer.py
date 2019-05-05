from common.head import *
from common.bone_classifier import BoneClassifier
from common.util import init_logging, load_config, try_make_dir, save_csv, load_csv, CsvAppender


def infer_single(args, config, bone_name):
    bone_setting = config['bone_classifier']['common']['bone_setting'][bone_name]
    common_dir = os.path.join(config['bone_classifier']['common']['dir'], bone_name)
    logging.debug("set common_dir at: {}".format(common_dir))
    try_make_dir(common_dir)
    bone_classifier = BoneClassifier(bone_name=bone_name,
                                     cls_num=bone_setting['class_num'],
                                     img_shape=bone_setting['image_shape'],
                                     common_dir=common_dir)

    weight_path = os.path.join(common_dir, config['bone_classifier']['train']['model_path'])
    bone_classifier.init_model(weight_path)

    image_name_list_path = config['annos']['img_name_csv']
    bone_label_csv_path = config['annos']['bone_label_csv']
    crop_image_csv = os.path.join(config['crop']['common']['dir'], config['crop']['result']['crop_img_csv'])
    bone_classifier.load_data(image_name_csv=load_csv(image_name_list_path),
                              crop_image_csv=load_csv(crop_image_csv),
                              label_csv=None)

    infer_result = bone_classifier.infer()

    title = ['img_name', 'bone_name', "label"]
    infer_result_csv = [{'img_name': img_name, 'bone_name': bone_name, 'label': infer_result[img_name]}
                        for img_name in infer_result]
    result_csv_path = os.path.join(common_dir, config['bone_classifier']['infer']['result_csv_path'])
    save_csv(result_csv_path, infer_result_csv, title)
    return infer_result_csv


def main_infer(args, config):
    total_result_csv_path = os.path.join(config['bone_classifier']['common']['dir'],
                                         config['bone_classifier']['infer']['result_csv_path'])
    csv_appender = CsvAppender(filepath=total_result_csv_path, title=['img_name', 'bone_name', "label"])
    bone_names = list(config['bone_classifier']['common']['bone_setting'].keys())
    for bone_name in bone_names:
        logging.info("infer @ {}".format(bone_name))
        infer_result_csv = infer_single(args, config, bone_name)
        csv_appender.append_rows(infer_result_csv)


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
    main_infer(args, config)
