# mmseg/datasets/custom_datasets.py
import os.path as osp
from mmseg.datasets import CityscapesDataset
from mmseg.datasets.builder import DATASETS


@DATASETS.register_module()
class CityscapesCoarseDataset(CityscapesDataset):
    def __init__(self, **kwargs):
        self.gt_suffix = "_gtCoarse_labelTrainIds.png"  # 关键修改
        super().__init__(**kwargs)

    def load_annotations(self, img_dir, img_suffix='_leftImg8bit.png',
                         ann_dir=None, seg_map_suffix='_gtCoarse_labelTrainIds.png',  # 确保这里匹配
                         split=None):
        return super().load_annotations(
            img_dir, img_suffix, ann_dir, seg_map_suffix, split)