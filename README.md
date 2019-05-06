# Bone-age-algorithm

This repository contains the code used for the paper- Diagnostic Performance of Convolutional Neural Network-Based TW3 Bone Age Assessment system.

The code mainly contains the following modules:
* Pre-processing module
* Positioning module for key points detection of an X-ray image, including training and prediction
* Cropping module for epiphysis ROIs 
* Classifying module for epiphysis ROIs, including training and prediction
* Bone age calculating module

## Dependencies
The main programs and third-party libraries included in the operating environment:
* Operating environment:
    * Python 3.7.3
    * anaconda3-4.3.14
* Third-party libraries:
    * image 1.5.27
    * Keras 2.2.4
    * opencv 4.1.0.25
    * tensorflow 1.13.1
    * tqdm 4.31.1

## How to use it?
Run the script in turn after the data is stored in the data directory in a certain format.

The configuration file is named as config.json, and the index path of the related files can be modified by modifying this file.

### Data storage 

* Input image directory

Input the directory where the image is placed, and the placement location is：config.json -> input_img_dir

* Input name list

Input the name list of the image, and the placement location is：config.json -> annos -> img_name_csv

As the following format:

| img_name |
| :--- : |
| file name 1 |
| file name 2 |
| ...... |

* Input gender list

Input the children's gender mapping table corresponding to the input images, and the placement location is: config.json -> annos -> gender_csv

As the following format:

| img_name | gender |
| :---: | :---:|
| file name 1 | gender 1（M for male and F for female）|
| file name 2 | gender 2 |
| ...... | ...... |

* Epiphysis classifying table

This is the epiphysis classifying table corresponding to TW3-method for the training dataset, and the placement location is: config.json -> annos -> bone_label_csv

As the following format:

| img_name | bone_name | label |
| :---: | :---: | :---: |
| image name 1 | epiphysis A | score of epiphysis A|
| image name 1 | epiphysis B | score of epiphysis B |
| ...... | ...... | ...... |

The score is in the form of a number, such as 0-A, 1-B...

* Key annotation lookup table

This is the key points annotation positioning table for the training dataset, and the placement location is: config.json -> annos -> align_point_json_csv

As the following format:

| img_name | json_path |
| :---: | :---: |
| image name 1 | json path 1 |
| image name 2 | json path 2 |
| ...... | ...... |


* Key points numbering list

Key points numbering name can be same as './crop/point_align'.

### Script running sequence

After the data is properly placed and the dependencies installation is complete, run the script in order：

- Prepare operating environment: python 1a_build_wksp.py
- Positioning module：
    - Train the module: python 2a_alignment_train.py
    - Use the module to get positioning results: python 2b_alignment_infer.py
- Crop ROIs based on positioning results： python 3a_crop_roi.py
- Classifying module：
    - Train the module： python 4a_cls_train.py
    - Use the module to get classifying results: python 4b_cls_infer.py
- Calculate bone age values based on classifying results and gender information： python 5a_calc_bone_age.py

It should be noted that, the folder named 'common' contains some public code and the folder named 'crop' contains some code for cropping purpose. 
