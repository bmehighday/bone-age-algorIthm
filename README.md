# bone-age_python_e2e

本代码分支是论文《_Diagnostic Performance of Convolutional Neural Network-Based TW3 Bone Age Assessment system_》中的实现代码。

代码主要包含以下模块：
* 预处理模块
* X光图片关键点检测模块，包含训练及预测
* 骨骺ROI截取模块
* 骨骺ROI分级模块，包含训练及预测
* 骨龄值计算模块

## 依赖项
运行环境主要包含的程序和第三方库有：
* 运行环境：
    * Python 3.7.3
    * anaconda3-4.3.14
* 第三方库：
    * image 1.5.27
    * Keras 2.2.4
    * opencv 4.1.0.25
    * tensorflow 1.13.1
    * tqdm 4.31.1

## 使用方式
将数据按一定格式存放到数据目录后，依次运行脚本。

其中配置文件为config.json，相关文件的索引路径可通过修改此文件而修改。

### 数据存放方式

#### 输入图片目录

输入图片放置的目录,放置位置为：config.json -> input_img_dir

#### 输入列表

输入的图片的文件名的列表,放置位置为：config.json -> annos -> img_name_csv

表格的形式为：

| img_name |
| :--- : |
| 文件名1 |
| 文件名2 |
| ...... |

#### 输入图片性别列表

输入的图片，对应儿童的性别映射表,放置位置为：config.json -> annos -> gender_csv

表格的形式为：

| img_name | gender |
| :---: | :---:|
| 文件名1 | 性别1（M为男，F为女）|
| 文件名2 | 性别2 |
| ...... | ...... |

#### 训练数据骨骺分级表

训练时，训练数据中图片各个TW3标准所需骨骺的分级表。放置位置为：config.json -> annos -> bone_label_csv

表格的形式为：

| img_name | bone_name | label |
| :---: | :---: | :---: |
| 图片文件名1 | 骨骺编号A | 图片1上骨骺A对应的分数 |
| 图片文件名1 | 骨骺编号B | 图片1上骨骺B对应的分数 |
| ...... | ...... | ...... |

其中，分数为数字形式，对应为0-A, 1-B,...

#### 关键标注查找表

训练时，训练数据中图片关键点标注文件定位表。放置位置为：config.json -> annos -> align_point_json_csv

表格的形式为：

| img_name | json_path |
| :---: | :---: |
| 图片文件名1 | 对应关键点json路径1 |
| 图片文件名2 | 对应关键点json路径2 |
| ...... | ...... |


#### 关键点编号列表

关键点编号名，同　crop/point_align　即可。

### 脚本运行顺序

数据放置完毕且依赖安装完毕后，按顺序运行脚本即可：

- 准备环境： python 1a_build_wksp.py
- 定位模块：
    - 训练定位模块： python 2a_alignment_train.py
    - 使用定位模块生成定位结果： python 2b_alignment_infer.py
- 根据定位结果截取ROI： python 3a_crop_roi.py
- 分级模块：
    - 训练分级模块： python 4a_cls_train.py
    - 使用分级模块生成分级结果： python 4b_cls_infer.py
- 根据分级结果和性别信息，计算骨龄值： python 5a_calc_bone_age.py

其中，common文件夹内包含一些公有代码，crop文件夹内包含一些crop用的代码