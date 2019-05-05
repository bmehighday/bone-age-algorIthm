from common.head import *
from keras.applications.resnet50 import ResNet50

def resnet50_align(input_shape, points_num):
    if len(input_shape) == 2:
        input_shape = tuple(list(input_shape) + [1])
    return ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=input_shape,
                    pooling=None, classes=points_num * 2)

def resnet50_cls(input_shape, classes_num):
    if len(input_shape) == 2:
        input_shape = tuple(list(input_shape) + [1])
    return ResNet50(include_top=True, weights=None, input_tensor=None, input_shape=input_shape,
                    pooling=None, classes=classes_num)