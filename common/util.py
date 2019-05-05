from common.head import *

def check_file(filepath):
    if not os.path.isfile(filepath):
        logging.error("Error! {} does not exist!".format(filepath))
        exit(1)

def check_dir(dirpath):
    if not os.path.isdir(dirpath):
        logging.error("Error! {} does not exist!".format(dirpath))
        exit(1)

def load_csv(filepath):
    return [row for row in csv.DictReader(open(filepath, 'r'))]

def load_list(filepath):
    return [row.strip() for row in open(filepath, "r").readlines()]

def map_csv_to_dict(cdata, keys):
    if isinstance(keys, list):
        tuple_flag = len(keys) > 1
    else:
        tuple_flag = True
        keys = [keys]
    if not tuple_flag:
        return {row[keys[0]]: row for row in cdata}
    else:
        return {tuple([row[key] for key in keys]): row for row in cdata}


def load_csv_map(filepath, keys):
    return  map_csv_to_dict(load_csv(filepath), keys)

class CsvAppender:
    def __init__(self, filepath, title=None):
        self.filepath = filepath
        self.title = title
        if self.title is None:
            with open(self.filepath, "w") as fp:
                fp.write("")
        else:
            with open(self.filepath, "w") as fp:
                csv_writer = csv.DictWriter(fp, fieldnames=self.title, lineterminator='\n')
                csv_writer.writeheader()

    def append_rows(self, rows):
        to_append = []
        if self.title is not None:
            for row in rows:
                to_append.append([row[ti] for ti in self.title])
        else:
            to_append = rows
        with open(self.filepath, "a+") as fp:
            csv_writer = csv.writer(fp, lineterminator='\n')
            csv_writer.writerows(to_append)

    def append_row(self, row):
        self.append_rows([row])

def load_image(filepath, output_shape=None):
    read_img = cv2.imread(filepath, 0)
    if output_shape is not None:
        ret_img = cv2.resize(read_img, (output_shape[1], output_shape[0]))
        return ret_img
    else:
        return read_img

def init_logging(debug_mode=False):
    format_string = '%(asctime)s(%(funcName)s:%(lineno)d)[%(levelname)s]: %(message)s'
    date_string = '%Y/%m/%d-%H:%M:%S'
    if debug_mode:
        logging.basicConfig(level=logging.DEBUG,
                            format=format_string,
                            datefmt=date_string)
    else:
        logging.basicConfig(level=logging.INFO,
                            format=format_string,
                            datefmt=date_string)


def load_config(config_path):
    return json.load(open(config_path, "r"))


def try_make_dir(dirpath, clear=False):
    if os.path.isdir(dirpath) and clear:
        shutil.rmtree(dirpath)
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)


def save_csv(filepath, data, title):
    with open(filepath, "w") as fp:
        csv_writer = csv.DictWriter(fp, fieldnames=title, lineterminator='\n', extrasaction='ignore')
        csv_writer.writeheader()
        csv_writer.writerows(data)

def get_max_i(nums):
    max_i = 0
    for i in range(len(nums)):
        if nums[i] > nums[max_i]:
            max_i = i
    return max_i