from common.head import *
from common.util import try_make_dir, load_config, init_logging, check_dir, check_file, load_csv


def main(args, config):
    try_make_dir(config['common']['data_dir'])
    check_file(config['annos']['img_name_csv'])
    check_file(config['annos']['bone_label_csv'])
    check_file(config['annos']['gender_csv'])
    check_dir(config['input_img_dir'])
    img_name_csv = load_csv(config['annos']['img_name_csv'])
    for row in tqdm(img_name_csv, desc="check image"):
        check_file(os.path.join(config['input_img_dir'], row['img_name']))


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
