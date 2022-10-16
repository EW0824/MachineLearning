
# # To be changed accordingly, if it is required.
# config["img_dir"] = "./data/model/brats20/TrainingData"
# config["label_dir"] = "./data/model/brats20/TrainingData"
# config["test_dir"] = "./data/model/brats20/ValidationData" # or change 'ValidationData' --> 'TestData' when you predict for test data
# config["num_test_files"] = 125 # Currently, this is number of validation files, change it to number of test files when you predict for test files
#
# config["data_file"] = "./data/model/brats20_data.h5"
# config["data_file_test"] = "./data/model/brats20_data_test.h5"
# config["model_file"] = "./data/model/isensee_2017_model.h5"
#
# config["training_file"] = "./data/model/training_ids.pkl"
# config["validation_file"] = "./data/model/validation_ids.pkl"
# config["test_file"] = "./data/model/test_ids.pkl"
#
# config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.


# IMG_DIR = '/Users/OAA/Desktop/MICCAI_BraTS2020_TrainingData'
LABEL_DIR = '/Users/OAA/Desktop/MICCAI_BraTS2020_ValidationDataData'

DATA_FILE = "/Users/OAA/Desktop/example/data.h5"


IMG_DIR = "/Users/OAA/Desktop/example/TrainingData"

MODALITIES = ['t1', 't1ce', 't2', 'flair']

import os
import glob

from data import write_data_to_file, open_data_file


def fetch_training_data_files(return_subject_ids=False):
    training_data_files = list()
    subject_ids = list()
    # processed_dir = config["preprocessed"]
    processed_dir = IMG_DIR
    for idx, subject_dir in enumerate(glob.glob(os.path.join(processed_dir, "*"))):
        #if idx == 5: break
        subject_ids.append(os.path.basename(subject_dir))
        subject_files = list()
        for modality in MODALITIES + ["seg"]:
            # subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
            subject_files.append(os.path.join(subject_dir, os.path.basename(subject_dir) + '_' + modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files



# if os.path.exists(config["model_file"])


training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)
print(training_files, subject_ids)

write_data_to_file(training_files, DATA_FILE, image_shap=(128, 128, 128), subject_ids=subject_ids)

