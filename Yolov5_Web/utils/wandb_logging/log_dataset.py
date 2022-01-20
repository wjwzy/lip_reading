import argparse
from pathlib import Path

import yaml

from Yolov5_Web.utils.wandb_logging.wandb_utils import WandbLogger
from Yolov5_Web.utils.datasets import LoadImagesAndLabels

WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def create_dataset_artifact(opt):
    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)  # web_data dict
    logger = WandbLogger(opt, '', None, data, job_type='create_dataset')
    nc, names = (1, ['item']) if opt.single_cls else (int(data['nc']), data['names'])
    names = {k: v for k, v in enumerate(names)}  # to index dictionary
    logger.log_dataset_artifact(LoadImagesAndLabels(data['train']), names, name='train')  # trainset
    logger.log_dataset_artifact(LoadImagesAndLabels(data['val']), names, name='val')  # valset

    # Update web_data.yaml with artifact links
    data['train'] = WANDB_ARTIFACT_PREFIX + str(Path(opt.project) / 'train')
    data['val'] = WANDB_ARTIFACT_PREFIX + str(Path(opt.project) / 'val')
    path = opt.data if opt.overwrite_config else opt.data.replace('.', '_wandb.')  # updated web_data.yaml path
    data.pop('download', None)  # download via artifact instead of predefined field 'download:'
    with open(path, 'w') as f:
        yaml.dump(data, f)
    print("New Config file => ", path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--web_data', type=str, default='web_data/coco128.yaml', help='web_data.yaml path')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--project', type=str, default='YOLOv5', help='name of W&B Project')
    parser.add_argument('--overwrite_config', action='store_true', help='overwrite web_data.yaml')
    opt = parser.parse_args()

    create_dataset_artifact(opt)
