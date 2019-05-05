from common.head import *
from common.resnet50 import resnet50_align
from common.util import map_csv_to_dict, load_image, try_make_dir
from keras.callbacks import CSVLogger, ModelCheckpoint
from tqdm import tqdm

class BoneAligner:
    def __init__(self, point_num, img_shape, common_dir):
        self.point_num = point_num
        self.img_shape = img_shape
        self.common_dir = common_dir
        self.point_align_id = None
        self.model = None
        self.image_names = None
        self.input_images = None
        self.input_points = None
        self.weight_path = None
        self.image_org_shape = None

        try_make_dir(self.common_dir)

    def load_data(self, image_name_csv, img_dir, point_jsons_csv, point_align_id_list):
        self.image_names = [row['img_name'] for row in image_name_csv]
        self.point_align_id = deepcopy(point_align_id_list)
        assert len(self.point_align_id) == self.point_num

        logging.info("load images: {}".format(len(self.image_names)))
        self.input_images = []
        self.image_org_shape = []
        for image_name in tqdm(self.image_names):
            img_path = os.path.join(img_dir, image_name)
            img = load_image(img_path)
            self.image_org_shape.append(img.shape)
            self.input_images.append(cv2.resize(img, tuple(self.img_shape)))

        if point_jsons_csv is not None:
            point_jsons_dict = map_csv_to_dict(point_jsons_csv, ['img_name'])
            logging.info("load jsons: {}".format(len(point_jsons_csv)))
            self.input_points = []
            for image_name in tqdm(self.image_names):
                json_path = point_jsons_dict[image_name]['json_path']
                with open(json_path, "r") as fp:
                    jdata = json.load(fp)
                self.input_points.append([jdata[point_id] for point_id in self.point_align_id])
            assert len(self.input_points) == len(self.input_images)

    def init_model(self, weight_path=None):
        if self.model is None:
            logging.debug("init model: resnet50")
            self.model = resnet50_align(self.img_shape, self.point_num)
        if weight_path is not None:
            self.weight_path = weight_path

    def train(self, train_configs):
        assert self.input_images is not None
        assert self.input_points is not None
        assert self.model is not None

        opt = keras.optimizers.RMSprop(lr=train_configs['init_lr'], decay=0.0)
        self.model.compile(optimizer=opt,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

        csv_path = os.path.join(self.common_dir, train_configs['csv_name'])
        ckpt_path = os.path.join(self.common_dir, train_configs['ckpt_dir'], "weights.{epoch:03d}.hdf5")
        try_make_dir(os.path.join(self.common_dir, train_configs['ckpt_dir']), clear=False)
        call_back_list = [CSVLogger(csv_path, separator=',', append=False),
                          ModelCheckpoint(ckpt_path)]
        logging.info("Start train the model")
        flat_labels = []
        for i in range(len(self.input_points)):
            points = self.input_points[i]
            to_append = []
            for p in points:
                to_append += [p['x'] * 1.0 / self.image_org_shape[i][0],
                              p['y'] * 1.0 / self.image_org_shape[i][1]]
            flat_labels.append(to_append)
        input_train_data = np.array(self.input_images)[:,:,:, np.newaxis]
        logging.debug("input_train_data.shape = {}".format(input_train_data.shape))
        self.model.fit(input_train_data, np.array(flat_labels),
                       epochs=train_configs['max_epochs'],
                       batch_size=train_configs['batch_size'],
                       callbacks=call_back_list)
        logging.info("done")

        model_save_path = os.path.join(self.common_dir, train_configs['model_path'])
        logging.info("Save model at {}".format(model_save_path))
        self.model.save(model_save_path)

    def infer(self):
        assert self.input_images is not None
        assert self.model is not None
        assert self.weight_path is not None

        self.model.load_weights(self.weight_path)
        logging.info("load weight from {}".format(self.weight_path))

        input_infer_data = np.array(self.input_images)[:,:,:, np.newaxis]
        logging.debug("input_infer_data.shape = {}".format(input_infer_data.shape))
        align_result = self.model.predict(input_infer_data)
        ret_result = {}
        for i in range(len(self.image_names)):
            to_append = {}
            for j in range(len(self.point_align_id)):
                to_append[self.point_align_id[j]] = {'x': align_result[i][j * 2] * self.image_org_shape[i][0],
                                                     'y': align_result[i][j * 2 + 1] * self.image_org_shape[i][1]}
            ret_result[self.image_names[i]] = to_append
        return ret_result
