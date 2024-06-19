from torch._C import set_flush_denormal
from det3d import torchie
import numpy as np
import torch

from ..registry import PIPELINES
from OpenPCSeg.pcseg.data.dataset.semantickitti.laserscan import SemLaserScan
from copy import deepcopy
from torchvision.transforms import functional as F

class DataBundle(object):
    def __init__(self, data):
        self.data = data


@PIPELINES.register_module
class TTAReformat(object):
    def __init__(self, cfg, **kwargs):
        self.tta_flag = cfg.get('tta_flag', False)
        self.num_tta_tranforms = cfg.get('num_tta_tranforms', -1)
        self.A = SemLaserScan()
    
    def get_range_image(self):

        def prepare_input_label_semantic_with_mask(sample):
            scale_x = np.expand_dims(np.ones([self.A.proj_H, self.A.proj_W]) * 50.0, axis=-1).astype(np.float32)
            scale_y = np.expand_dims(np.ones([self.A.proj_H, self.A.proj_W]) * 50.0, axis=-1).astype(np.float32)
            scale_z = np.expand_dims(np.ones([self.A.proj_H, self.A.proj_W]) * 3.0,  axis=-1).astype(np.float32)
            scale_matrx = np.concatenate([scale_x, scale_y, scale_z], axis=2)

            each_input = [
                sample['xyz'] / scale_matrx,
                np.expand_dims(sample['intensity'], axis=-1), 
                np.expand_dims(sample['range_img']/80.0, axis=-1),
                np.expand_dims(sample['xyz_mask'], axis=-1),         
            ]
            input_tensor = np.concatenate(each_input, axis=-1)
            
            semantic_label = sample['semantic_label'][:, :]
            semantic_label_mask = sample['xyz_mask'][:, :]

            return input_tensor, semantic_label, semantic_label_mask
        
        
        # def generate_label(semantic_label):
        #     original_label=np.copy(semantic_label)
        #     label_new=self.sem_label_transform(original_label)

        #     return label_new
    
        # prepare attributes
        dataset_dict = {}

        dataset_dict['xyz'] = self.A.proj_xyz
        dataset_dict['intensity'] = self.A.proj_remission
        dataset_dict['range_img'] = self.A.proj_range
        dataset_dict['xyz_mask'] = self.A.proj_mask
        
        
        # TODO: KHANH add create random label for range image
        # because we will not use them
        semantic_label = torch.rand(self.A.proj_H, self.A.proj_W)
        # semantic_train_label = generate_label(semantic_label)
        # dataset_dict['semantic_label'] = semantic_train_label
        dataset_dict['semantic_label'] = semantic_label
        # KHANH COMMENT

        scan, label, mask = prepare_input_label_semantic_with_mask(dataset_dict)
        
        # TODO: KHANH pass proj_x, proj_y for later shuffle with point cloud data and 
        # create range_pxpy
        data_dict = {
            'scan_rv': F.to_tensor(scan) if torch.is_tensor(scan) == False else scan,
            'range_pxpy': np.hstack([self.A.proj_x.reshape(-1, 1), self.A.proj_y.reshape(-1, 1)])
        }
        
        return data_dict


    def __call__(self, res, info):
        meta = res["metadata"]
        points = res["lidar"]["points"]
        voxels = res["lidar"]["voxels"]
        all_points = res["lidar"]["all_points"]


        data_bundle = dict(
            metadata=meta,
            points=points,
            voxels=voxels["voxels"],
            shape=voxels["shape"],
            num_points=voxels["num_points"],
            num_voxels=voxels["num_voxels"],
            coordinates=voxels["coordinates"],
            all_points=all_points, 
        )

        if res["mode"] == "train":
            data_bundle.update(res["lidar"]["targets"])
        elif res["mode"] == "val":
            data_bundle.update(dict(metadata=meta, ))

            # TODO: KHANH ADD
            # 把img相关的信息添加进去
            if "images" in res:
                data_bundle["images"] = res["images"]
    
            if "points_cuv" in res["lidar"]:
                data_bundle["points_cuv"] = res["lidar"]["points_cuv"]
            if "points_cp" in res["lidar"]:
                data_bundle["points_cp"] = res["lidar"]["points_cp"]
    
            # TODO: KHANH ADD reformatting data before input to the model
            if "range data" in res:
                data_bundle['range data'] = res['range data']
            # KHANH ADD
            
            if self.tta_flag:
                data_bundle_list = [data_bundle]
                assert self.num_tta_tranforms > 1
                for i in range(1, self.num_tta_tranforms):
                    point_key_i = "tta_%s_points" %i
                    voxel_key_i = "tta_%s_voxels" %i
                    images_key_i = 'tta_%s_images' %i

                    tta_points = res["lidar"][point_key_i]
                    tta_voxels = res["lidar"][voxel_key_i]
                    tta_images = res['lidar'][images_key_i]
                    
                    tta_data_bundle = dict(
                        metadata=meta,
                        points=tta_points,
                        voxels=tta_voxels["voxels"],
                        shape=tta_voxels["shape"],
                        num_points=tta_voxels["num_points"],
                        num_voxels=tta_voxels["num_voxels"],
                        coordinates=tta_voxels["coordinates"],

                        points_cuv = tta_images['points_cuv'],
                        images = res['images'],
                    )

                    if "range data" in res:
                        points_tta_range = deepcopy(tta_points)
                        self.A.set_points(points_tta_range[:, 0:3], points_tta_range[:, 3]) # points, intensity
                        tta_data_bundle['range data'] = self.get_range_image()
                         
                    data_bundle_list.append(tta_data_bundle)

                return data_bundle_list, info


        return data_bundle, info



