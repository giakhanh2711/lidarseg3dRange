from .. import builder
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint
from OpenPCSeg.pcseg.model import build_network
import numpy as np
import torch

@DETECTORS.register_module
class SegMSeg3DNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        img_backbone,
        img_head,
        point_head,
        range_backbone,
        neck=None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        **kwargs,
    ):
        super(SegMSeg3DNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained=None
        )
        
        # TODO: IMAGE BACKBONE -> SAM
#         sam_checkpoint = '/home/nnthao02/Linh_Khanh/lidarseg3dSamPT/sam_vit_h_4b8939.pth'
#         model_type = 'vit_h'
        
#         device = 'cuda'

#         sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
#         sam.to(device=device)
#         sam.eval()
#         self.mask_generator = SamAutomaticMaskGenerator(sam)
#         self.img_backbone = self.mask_generator.predictor
        # KHANH ADD
        
        self.img_backbone = builder.build_img_backbone(img_backbone)
        self.img_head = builder.build_img_head(img_head)
        self.point_head = builder.build_point_head(point_head)
        
        # TODO: KHANH ADD range backbone
        self.range_backbone = build_network(
            model_cfgs=range_backbone,
            num_class=range_backbone['NUM_CLASS'],
        )
        self.batchsize=kwargs.get('batchsize')
        self.tta = kwargs.get('tta')
        # KHANH ADD
        


    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))


    def extract_feat(self):
        assert False


    def forward(self, example, return_loss=True, **kwargs):
        """
        """
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]
        #print('num_voxels:', num_voxels) 
        batch_size = len(num_voxels)
        # ensure that the points just including [bs_idx, x, y, z]
        points = example["points"][:, 0:4]
        # print('batch size: ', batch_size)
        # print('points example shape: ', points.shape, points[:10, 0], max(points[:, 0]), min(points[:, 0]))  #torch.Size([416640, 6]) torch.Size([138912, 6])
        # print('voxels shape:', voxels.shape)  #torch.Size([180345, 5, 5]) torch.Size([59722, 5, 5])

        # camera branch
        # images: [batch, num_cams, num_ch, h, w] like [1, 5, 3, 640, 960]
        images = example["images"]         
        # print('images example shape: ', example['images'].shape)
        num_cams, hi, wi = images.shape[1], images.shape[3], images.shape[4] 
        # images: (batch, num_cams=5, 3, h, w) -> (batch*num_cams=5, 3, h, w)
        images = images.view(-1, 3, hi, wi)

        
        
        #TODO: COMMENT KHANH
        # img_backbone_return: (batch*num_cams=5, c, ho, wo)
        img_backbone_return = self.img_backbone(images)  
        img_data = dict(
            inputs=img_backbone_return,
            batch_size=batch_size,
        )

        # TODO: KHANH COMMENT
#         image_features_a = []
        
#         with torch.no_grad():
#             for i in range(images.shape[0]):
#                 self.img_backbone.set_image(images[i])
#                 image_features_a.append(self.img_backbone.get_image_embedding())
            
#         image_features = torch.cat(image_features_a, dim=0).unsqueeze(0)

#         img_data = dict(
#             inputs = image_features,
#             batch_size = batch_size
#         )

        if return_loss:
            images_sem_labels = example["images_sem_labels"]
            images_sem_labels = images_sem_labels.view(-1, hi, wi).unsqueeze(1)
            img_data["images_sem_labels"] = images_sem_labels
        
        img_data = self.img_head(batch_dict=img_data, return_loss=return_loss)
        # get image_features from the img_head
        image_features = img_data["image_features"]
        _, num_chs, ho, wo = image_features.shape

        # # TODO: KHANH ADD
        # if self.tta is not None:
        #     image_features = torch.repeat_interleave(image_features, repeats=self.tta, dim=0)
            
        # KHANH ADD
        #print(f'image_features shape: {image_features.shape}, {batch_size}') # torch.Size([24, 48, 160, 240]), 4
        image_features = image_features.view(batch_size, num_cams, num_chs, ho, wo)
        #print(f'image features shape after: {image_features.shape}') # torch.Size([4, 6, 48, 160, 240])


        # TODO: KHANH Get range image data before passing to GF-Phase and SF-Phase
        range_data = example['range data']
        r = []
        range_pxpy = []

        for i in range(len(range_data)):
            r.append(range_data[i]['scan_rv'])
            
            rxy = range_data[i]['range_pxpy']
    
            rxy = torch.from_numpy(rxy.astype(np.float32)).to('cuda')

            # # TODO: KHANH ADD
            # if self.tta is not None:
            #     rxy = rxy.repeat(self.tta, *rxy.shape[1:])
            # # KHANH ADD
            
            range_pxpy.append(rxy)

        range_data_input = {
            'scan_rv':torch.stack(r).to('cuda')
            }

        range_features = self.range_backbone(range_data_input)['range features']

        # # TODO: KHANH ADD
        # if self.tta is not None:
        #     range_features = torch.repeat_interleave(range_features, repeats=self.tta, dim=0)
        # # KHANH ADD
        
        # KHANH ADD


        # lidar branch
        # construct a batch_dict like pv-rcnn
        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            voxel_coords=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
            points=points,
        )

        # VFE
        input_features = self.reader(data["features"], data["num_voxels"], data["voxel_coords"])
        data["voxel_features"] = input_features
        
        # backbone
        data = self.backbone(data)

        # prepare labels for training
        if return_loss:
            data["voxel_sem_labels"] = example["voxel_sem_labels"]
            data["point_sem_labels"] = example["point_sem_labels"]

        # fusion and segmentation in point head
        data["points_cuv"] = example["points_cuv"]
        
        # print(f'data["points_cuv"] shape: {data["points_cuv"].shape}') # torch.Size([138880, 4])
        # print(f'example["points_cp"]: {example["points_cp"].shape}') # torch.Size([138880, 3])
        data["image_features"] = image_features
        data["camera_semantic_embeddings"] = img_data.get("camera_semantic_embeddings", None)
        data["metadata"] = example.get("metadata", None)

        # TODO: KHANH ADD stores data before passing to PointSegMSeg3DHead class
        data['range_features'] = range_features
        data['range_pxpy'] = range_pxpy
        # KHANH ADD

        data = self.point_head(batch_dict=data, return_loss=return_loss)


        if return_loss:
            seg_loss_dict = {}
            point_loss, point_loss_dict = self.point_head.get_loss()

            # compute the img head loss
            img_loss, point_loss_dict = self.img_head.get_loss(point_loss_dict)


            # this item for Optimizer, formating as loss per task
            total_loss = point_loss + img_loss
            opt_loss = [total_loss]
            seg_loss_dict["loss"] = opt_loss

            # reformat for text logger
            for k, v in point_loss_dict.items():
                repeat_list = [v for i in range(len(opt_loss))]
                seg_loss_dict[k] = repeat_list

            return seg_loss_dict

        else:
            return self.point_head.predict(example=example, test_cfg=self.test_cfg)
