DATADIR = "/mnt/Terry/data/caltech_256/256_ObjectCategories"
IMAGE_SIZE = (256,256,3)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


import os
import glob

_class_dirs = sorted(glob.glob(os.path.join(DATADIR, "**/")))
_class_dirs = [os.path.basename(os.path.dirname(p)) for p in _class_dirs]

CLASSNUM_TO_CLASSNAME = {
    int(subdir.split('.')[0]) : subdir.split('.')[1] for subdir in _class_dirs
}

