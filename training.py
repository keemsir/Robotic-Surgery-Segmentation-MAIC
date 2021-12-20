import os
from typing import List

# Utils ..
def maybe_mkdir_p(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def subdirs(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None, sort: bool = True) -> List[str]:
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res
##
def Group_frame(surgery_NUM: int, surgery_path='/mnt/dataset/train-valid_set'):
    '''
    surgery_path(default) : '/mnt/dataset/train-valid_set/'
    surgery_NUM category :
    '''

    surgery_DIC = {0: 'Prograsp (da Vinci) (GRSR-UCDV)',
                   1: 'R-LND (da Vinci) (NDDR-LADV)',
                   2: 'R-Maryland (da Vinci) (BFCR-MADV)',
                   3: 'R-Scissors (da Vinci) (MSCR-UCDV)',
                   4: 'Suction Irrigator (SIRL-UCUK)',
                   5: 'Graspers (straight) (OLYMPUS) (GRSL-UCOL)',
                   6: 'Needle Holder (AESCULAP) (NDHL-UCAE)',
                   7: 'Metal Clip Applier (Medtronic) (CLAL-M3MT)',
                   8: 'Polymer Clip Applier (CLAL-UCUK)'}
    print('Print surgery : ' + (surgery_DIC[surgery_NUM]))

    SG_listpath = os.path.join(surgery_path, surgery_DIC[surgery_NUM])
    SG_listname = subfiles(SG_listpath, join=False, suffix='.json')

    Frame_class = []

    for sg in SG_listname:
        name_, ext_ = os.path.splitext(sg)
        name1, name2, name3, name4, name5, name6 = name_.split('_')
        Frame_class.append('{}_{}_{}_{}_{}'.format(name1, name2, name3, name4, name5))

    set_class = list(set(Frame_class))

    return set_class


#
# Group_frame(0)

##
import json
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

task_dir = '/home/ncc/PycharmProjects/Robotic/task'
maybe_mkdir_p(task_dir)


def png2nifti(training_folder: str = '/mnt/dataset/train-valid_set', test_folder: str = '/mnt/dataset/test_set',
              save_folder: str = task_dir):
    # training_folder : training file path '/mnt/dataset/train-valid_set'
    # test_folder : test file path '/mnt/dataset/test_set'
    # save_folder : [imagesTr, imagesTs] Save Folder path

    # Surgery Catalog
    surgery_DIC = {0: 'Prograsp (da Vinci) (GRSR-UCDV)',
                   1: 'R-LND (da Vinci) (NDDR-LADV)',
                   2: 'R-Maryland (da Vinci) (BFCR-MADV)',
                   3: 'R-Scissors (da Vinci) (MSCR-UCDV)',
                   4: 'Suction Irrigator (SIRL-UCUK)',
                   5: 'Graspers (straight) (OLYMPUS) (GRSL-UCOL)',
                   6: 'Needle Holder (AESCULAP) (NDHL-UCAE)',
                   7: 'Metal Clip Applier (Medtronic) (CLAL-M3MT)',
                   8: 'Polymer Clip Applier (CLAL-UCUK)'}

    maybe_mkdir_p(os.path.join(save_folder, 'imagesTr'))
    maybe_mkdir_p(os.path.join(save_folder, 'imagesTs'))
    maybe_mkdir_p(os.path.join(save_folder, 'labelsTr'))
    print('Creating "{}" Image & Label ..'.format(os.path.basename(os.path.normpath(save_folder))))

    # training_files = os.listdir(training_folder)

    # training_files = subfiles(training_folder, join=False, suffix='.json')
    for i in range(len(surgery_DIC)):

        # training_files = subfiles(os.path.join(training_folder, surgery_DIC[i]), join=False, suffix='.json')
        # training_files : suffix = '.json'

        for gf in Group_frame(i):

            training_files = subfiles(os.path.join(training_folder, surgery_DIC[i]), join=False, prefix=gf,
                                      suffix='.json')

            for training_file in training_files:
                training_path = os.path.join(training_folder, surgery_DIC[i], training_file)

                training_file_name = os.path.splitext(training_file)[i]  # split suffix, Only Name

                # png image
                png_data = plt.imread(os.path.join(training_folder, training_path[:-5] + '.png'))

                # json label
                f_json = open(os.path.join(training_folder, surgery_DIC[i], training_file), 'rt', encoding='UTF8')
                jsonData = json.load(f_json)
                jsonPoints = jsonData.get('annotations')[0].get('points')
                jsonImgSize = jsonData.get('images')

                points = np.array(np.round(jsonPoints), dtype=np.int)
                json_height = jsonImgSize.get('height')
                json_width = jsonImgSize.get('width')
                label_json = np.empty((json_height, json_width), dtype=np.uint8)
                cv2.fillPoly(label_json, [points], 1)

                # image

                SLICE_SIZE_X, SLICE_SIZE_Y, SLICE_COUNT = png_data.shape

                images = np.empty([SLICE_SIZE_X, SLICE_SIZE_Y, 0], dtype=np.single)

                image_png = np.expand_dims(png_data[:, :, 0], axis=2)
                images = np.append(images, image_png, axis=2)

                label_json = np.expand_dims(label_json, axis=2)
                labels = np.empty([json_height, json_width, 0], dtype=np.uint8)
                # print(labels.shape, label_json.shape)
                labels = np.append(labels, label_json, axis=2)

            niim = nib.Nifti1Image(images, affine=np.eye(4))
            nib.save(niim, os.path.join(save_folder, 'imagesTr/{}.nii.gz'.format(gf)))

            del images

            nila = nib.Nifti1Image(labels, affine=np.eye(4))
            nib.save(nila, os.path.join(save_folder, 'labelsTr/{}.nii.gz'.format(gf)))

            del labels

    print('"{}" Image & Label Completed !!'.format(os.path.basename(os.path.normpath(save_folder))))
    print('Image Patient : {}'.format(len(os.listdir(input_dir))))


## backup 11/26

def png2nifti(training_folder: str = '/mnt/dataset/train-valid_set', test_folder: str = '/mnt/dataset/test_set',
              save_folder: str = task_dir):
    # training_folder : training file path '/mnt/dataset/train-valid_set'
    # test_folder : test file path '/mnt/dataset/test_set'
    # save_folder : [imagesTr, imagesTs] Save Folder path

    # Surgery Catalog
    surgery_DIC = {0: 'Prograsp (da Vinci) (GRSR-UCDV)',
                   1: 'R-LND (da Vinci) (NDDR-LADV)',
                   2: 'R-Maryland (da Vinci) (BFCR-MADV)',
                   3: 'R-Scissors (da Vinci) (MSCR-UCDV)',
                   4: 'Suction Irrigator (SIRL-UCUK)',
                   5: 'Graspers (straight) (OLYMPUS) (GRSL-UCOL)',
                   6: 'Needle Holder (AESCULAP) (NDHL-UCAE)',
                   7: 'Metal Clip Applier (Medtronic) (CLAL-M3MT)',
                   8: 'Polymer Clip Applier (CLAL-UCUK)'}

    maybe_mkdir_p(os.path.join(save_folder, 'imagesTr'))
    maybe_mkdir_p(os.path.join(save_folder, 'imagesTs'))
    maybe_mkdir_p(os.path.join(save_folder, 'labelsTr'))
    print('Creating "{}" Image & Label ..'.format(os.path.basename(os.path.normpath(save_folder))))

    # training_files = os.listdir(training_folder)

    # training_files = subfiles(training_folder, join=False, suffix='.json')

    training_files = subfiles(os.path.join(training_folder, surgery_DIC[0]), join=False, suffix='.json')
    # training_files : suffix = '.json'

    for gf in Group_frame(0):

        training_files = subfiles(os.path.join(training_folder, surgery_DIC[0]), join=False, prefix=gf, suffix='.json')

        for training_file in training_files:
            training_path = os.path.join(training_folder, surgery_DIC[0], training_file)

            training_file_name = os.path.splitext(training_file)[0]  # split suffix, Only Name

            # png image
            png_path = training_path[:-5]
            png_data = plt.imread(os.path.join(training_folder, training_path[:-5] + '.png'))

            # json label

            f_json = open(os.path.join(training_folder, surgery_DIC[0], training_file[:-5] + '.json'), 'rt',
                          encoding='UTF8')
            jsonData = json.load(f_json)
            jsonPoints = jsonData.get('annotations')[0].get('points')
            jsonImgSize = jsonData.get('images')

            points = np.array(np.round(jsonPoints), dtype=np.int)
            labels = np.zeros((jsonImgSize.get('height'), jsonImgSize.get('width')), dtype=np.uint8)
            cv2.fillPoly(labels, [points], 1)

            # image

            SLICE_SIZE_X, SLICE_SIZE_Y, SLICE_COUNT = png_data.shape

            images = np.empty([SLICE_SIZE_X, SLICE_SIZE_Y, 0], dtype=np.single)

            image_png = np.expand_dims(png_data[:, :, 0], axis=2)
            images = np.append(images, image_png, axis=2)

            niim = nib.Nifti1Image(images, affine=np.eye(4))
            nib.save(niim, os.path.join(save_folder, 'imagesTr/{}.nii.gz'.format(training_file[:-5])))

            nila = nib.Nifti1Image(labels, affine=np.eye(4))
            nib.save(nila, os.path.join(save_folder, 'labelsTr/{}.nii.gz'.format(training_file[:-5])))

    test_files = subfiles(test_folder)

    for test_file in test_files:
        test_png = np.expand_dims(png_data_ts, axis=3)
        tests = np.append(tests, test_png, axis=3)

        nits = nib.Nifti1Image(tests, affine=np.eye(4))
        nib.save(nits, os.path.join(save_folder, 'imagesTs/{}.nii.gz'.format(training_file[:-5])))

    print('"{}" Image & Label Completed !!'.format(os.path.basename(os.path.normpath(save_folder))))
    print('Image Patient : {}'.format(len(os.listdir(input_dir))))