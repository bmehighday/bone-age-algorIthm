from common.head import *
from common.util import load_csv, load_csv_map, init_logging, load_config, save_csv, try_make_dir
from tqdm import tqdm


def get_imgname_bonename_dict(inf_result_csv):
    inf_result_dict = {}
    for row in inf_result_csv:
        img_name = row['img_name']
        bone_name = row['bone_name']
        label = row['label']
        if img_name not in inf_result_dict:
            inf_result_dict[img_name] = {}
        inf_result_dict[img_name][bone_name] = int(label)
    return inf_result_dict


def calc_bone_age(args, config):
    img_name_csv_path = config['annos']['img_name_csv']
    img_name_list = [row['img_name'] for row in load_csv(img_name_csv_path)]

    gender_csv_path = config['annos']['gender_csv']
    gender_dict = load_csv_map(gender_csv_path, ['img_name'])

    inf_result_csv_path = os.path.join(config['bone_classifier']['common']['dir'],
                                       config['bone_classifier']['infer']['result_csv_path'])
    inf_result_csv = load_csv(inf_result_csv_path)
    inf_result_dict = get_imgname_bonename_dict(inf_result_csv)

    calc_table = config['calc_age']['tw3_table']

    result_csv = []
    title = ['img_name', "gender", "RUS_age", "Carpal_age"]
    for img_name in tqdm(img_name_list):
        sex = {'M': 'male', 'F': 'female'}[gender_dict[img_name]['gender']]
        to_append = {'img_name': img_name, 'gender': sex}
        the_inf_dict = inf_result_dict[img_name]
        for age_meth in ['RUS', 'Carpal']:
            use_features = calc_table[age_meth + '_features']
            total_score = 0
            for feat in use_features:
                total_score += calc_table['bone_scores'][sex][feat][the_inf_dict[feat]]
            use_score_table = calc_table['score_2_age'][sex][age_meth]
            near_i = 0
            for i in range(len(use_score_table['score'])):
                if abs(use_score_table['score'][i] - total_score) < abs(use_score_table['score'][near_i] - total_score):
                    near_i = i
            age = use_score_table['age'][near_i]
            to_append[age_meth + '_age'] = age
        result_csv.append(to_append)

    try_make_dir(config['calc_age']['common']['dir'])
    calc_result_csv_path = os.path.join(config['calc_age']['common']['dir'],
                                        config['calc_age']['common']['bone_age_result'])
    save_csv(calc_result_csv_path, result_csv, title)


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
    calc_bone_age(args, config)
