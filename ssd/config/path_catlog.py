import os


class DatasetCatalog:
    DATA_DIR = '/content/drive/MyDrive/Cross_Domain_project/dataset'
    DATASETS = {
        '''
        #---ADDED DATASET INFORMATION ---#
        '''
        'clipart_train': {
            "data_dir": "clipart",
            "split": "train"
        },
        'clipart_test': {
            "data_dir": "clipart",
            "split": "test"
        },
        'dt_voc_2007_trainval': {
            "data_dir": "dt_clipart/VOC2007",
            "split": "trainval"
        },
        'dt_voc_2012_trainval': {
            "data_dir": "dt_clipart/VOC2012",
            "split": "trainval"
        },
        '''
        ##------------------------##
        '''
        'voc_2007_train': {
            "data_dir": "VOC2007",
            "split": "train"
        },
        'voc_2007_val': {
            "data_dir": "VOC2007",
            "split": "val"
        },
        'voc_2007_trainval': {
            "data_dir": "VOC2007",
            "split": "trainval"
        },
        'voc_2007_test': {
            "data_dir": "VOC2007",
            "split": "test"
        },
        'voc_2012_train': {
            "data_dir": "VOC2012",
            "split": "train"
        },
        'voc_2012_val': {
            "data_dir": "VOC2012",
            "split": "val"
        },
        'voc_2012_trainval': {
            "data_dir": "VOC2012",
            "split": "trainval"
        },
        'voc_2012_test': {
            "data_dir": "VOC2012",
            "split": "test"
        },
        'coco_2014_valminusminival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_valminusminival2014.json"
        },
        'coco_2014_minival': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_minival2014.json"
        },
        'coco_2014_train': {
            "data_dir": "train2014",
            "ann_file": "annotations/instances_train2014.json"
        },
        'coco_2014_val': {
            "data_dir": "val2014",
            "ann_file": "annotations/instances_val2014.json"
        },
    }

    @staticmethod
    def get(name):
        if "voc" in name:
            voc_root = DatasetCatalog.DATA_DIR
            if 'VOC_ROOT' in os.environ:
                voc_root = os.environ['VOC_ROOT']

            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(voc_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="VOCDataset", args=args)
            '''
            ADDED CLIPART DATASET GETTER
            '''
        elif "clipart" in name:
            clipart_root = DatasetCatalog.DATA_DIR
            if 'CLIPART_ROOT' in os.environ:
                clipart_root = os.environ['CLIPART_ROOT']
            
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(clipart_root, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(factory="CLIPDataset", args=args)
    
        raise RuntimeError("Dataset not available: {}".format(name))
