from common.head import *
from common.bone_alignment import BoneAligner
from common.util import init_logging, load_config, try_make_dir, save_csv, load_csv, CsvAppender, load_list
import json

def main(args, config):
    common_dir = config['alignment']['common']['dir']
    logging.debug("set common_dir at: {}".format(common_dir))
    try_make_dir(common_dir)
    bone_aligner = BoneAligner(point_num=config['alignment']['common']['point_num'],
                               img_shape=config['alignment']['common']['image_shape'],
                               common_dir=common_dir)

    weight_path = os.path.join(common_dir, config['alignment']['train']['model_path'])
    bone_aligner.init_model(weight_path)

    image_name_list = load_csv(config['annos']['img_name_csv'])
    point_align_id_list_path = config['annos']['point_align_id_list']
    bone_aligner.load_data(image_name_csv=image_name_list,
                           img_dir=config['input_img_dir'],
                           point_jsons_csv=None,
                           point_align_id_list=load_list(point_align_id_list_path))

    infer_result = bone_aligner.infer()
    result_json_dir = config['alignment']['infer']['result_json_dir']
    result_json_csv_path = config['alignment']['infer']['result_json_csv']
    result_json_tail = config['alignment']['infer']['tail']
    try_make_dir(result_json_dir, clear=True)

    title = ["img_name", "json_path"]
    result_json_csv = []
    for i in range(len(image_name_list)):
        image_name = image_name_list[i]['img_name']
        json_path = os.path.join(result_json_dir, image_name + result_json_tail)
        with open(json_path, "w") as fp:
            json.dump(infer_result[image_name], fp, indent=2)
        result_json_csv.append({"img_name": image_name, "json_path": json_path})
    logging.info("save result_json_csv at {}".format(result_json_csv_path))
    save_csv(result_json_csv_path, result_json_csv, title)

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
