from common.head import *
from common.resnet50 import resnet50_cls
from common.util import map_csv_to_dict, load_image, try_make_dir,get_max_i
from keras.callbacks import CSVLogger, ModelCheckpoint


class BoneClassifier:
    def __init__(self, bone_name, cls_num, img_shape, common_dir):
        self.bone_name = bone_name
        self.cls_num = cls_num
        self.img_shape = img_shape
        self.common_dir = common_dir
        self.model = None
        self.image_names = None
        self.input_images = None
        self.labels = None
        self.onehot_label = None
        self.weight_path = None

        if not os.path.isdir(self.common_dir):
            os.makedirs(self.common_dir, exist_ok=True)

    def load_data(self, image_name_csv, crop_image_csv, label_csv):
        self.image_names = [row['img_name'] for row in image_name_csv]
        crop_image_dict = map_csv_to_dict(crop_image_csv, ['img_name', 'bone_name'])
        self.input_images = []
        logging.info("load images: {}".format(len(self.image_names)))
        for image_name in tqdm(self.image_names):
            crop_image_path = crop_image_dict[(image_name, self.bone_name)]['crop_img_path']
            self.input_images.append(load_image(crop_image_path, output_shape=self.img_shape))

        if label_csv is not None:
            label_dict = map_csv_to_dict(label_csv, ['img_name', 'bone_name'])
            self.labels = [int(label_dict[(img_name, self.bone_name)]['label']) for img_name in self.image_names]
            self.onehot_label = np.zeros((len(self.labels), self.cls_num))
            for i in range(len(self.labels)):
                self.onehot_label[i][self.labels[i]] = 1

    def init_model(self, weight_path=None):
        if self.model is None:
            logging.debug("init model: resnet50")
            self.model = resnet50_cls(self.img_shape, self.cls_num)
        if weight_path is not None:
            self.weight_path = weight_path

    def train(self, train_configs):
        assert self.input_images is not None
        assert self.labels is not None
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
        input_train_data = np.array(self.input_images)[:,:,:, np.newaxis]
        logging.debug("self.shape = {}".format(self.img_shape))
        logging.debug("input_train_data.shape = {}".format(input_train_data.shape))
        logging.info("Start train the model")
        self.model.fit(input_train_data, self.onehot_label,
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
        predict_result = self.model.predict(input_infer_data)
        cls_result = [get_max_i(nums) for nums in predict_result]
#        cls_result = keras.backend.argmax(predict_result)
        ret_result = {}
        for i in range(len(self.image_names)):
            ret_result[self.image_names[i]] = cls_result[i]
        return ret_result
