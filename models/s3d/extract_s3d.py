import sys
from typing import Dict

import numpy as np
import torch
import torchvision
from models._base.base_extractor import BaseExtractor
from models.s3d.s3d_src.s3d import S3D
from models.transforms import CenterCrop, Resize, ToFloatTensorInZeroOne
from torchvision.io.video import read_video
from utils.io import reencode_video_with_diff_fps
from utils.utils import form_slices, show_predictions_on_dataset, form_slices_32

class ExtractS3D(BaseExtractor):

    def __init__(self, args) -> None:
        # init the BaseExtractor
        super().__init__(
            feature_type=args.feature_type,
            on_extraction=args.on_extraction,
            tmp_path=args.tmp_path,
            output_path=args.output_path,
            keep_tmp_files=args.keep_tmp_files,
            device=args.device,
        )
        # (Re-)Define arguments for this class
        self.stack_size = 64 if args.stack_size is None else args.stack_size
        self.step_size = 64 if args.step_size is None else args.step_size
        self.extraction_fps = 25 if args.extraction_fps is None else args.extraction_fps
        self.extraction_fps = None
        # normalization is not used as per: https://github.com/kylemin/S3D/issues/4
        self.transforms = torchvision.transforms.Compose([
            ToFloatTensorInZeroOne(),
            Resize(224),
            CenterCrop((224, 224))
        ])
        self.show_pred = args.show_pred
        self.output_feat_keys = [self.feature_type]
        self.name2module = self.load_model()

    def augment_rgb_frame(self, rgb, start_idx, end_idx):
        # Select frames from start_idx to end_idx (inclusive) along the frame dimension
        selected_frames = rgb[:, :, start_idx:end_idx + 1, :, :]

        # Interleave original frames with duplicated frames
        num_frames = selected_frames.shape[2]
        interleaved_frames = torch.empty(1, 3, 2 * num_frames, 224, 224)
        interleaved_frames[:, :, ::2, :, :] = selected_frames
        interleaved_frames[:, :, 1::2, :, :] = selected_frames

        return interleaved_frames

    @torch.no_grad()
    def extract(self, video_path: str) -> Dict[str, np.ndarray]:
        """Extracts features for a given video path.

        Arguments:
            video_path (str): a video path from which to extract features

        Returns:
            Dict[str, np.ndarray]: feature name (e.g. 'fps' or feature_type) to the feature tensor
        """
        # take the video, change fps and save to the tmp folder
        #print('video_path:', video_path)
        if self.extraction_fps is not None:
            video_path = reencode_video_with_diff_fps(video_path, self.tmp_path, self.extraction_fps)

        # read a video
        if (type(video_path) == str):
            rgb, audio, info = read_video(video_path, pts_unit='sec')
        else:
            rgb = torch.from_numpy(video_path)
            audio = None
            info = 30

        print("shape: ", rgb.shape)
        #import sys
        #sys.exit()

        # prepare data (first -- transform, then -- unsqueeze)
        rgb = self.transforms(rgb)  # could run out of memory here
        rgb = rgb.unsqueeze(0)
        # slice the stack of frames
        slices = form_slices(rgb.size(2), self.stack_size, self.step_size)
        ## to make a 32 stack feature vector for UCF Crime
        slices = form_slices_32(rgb.size(2), rgb.size(2)//32, rgb.size(2)//32)

        print("Slices: ", slices)
        #sys.exit(1)
        vid_feats = []

        for stack_idx, (start_idx, end_idx) in enumerate(slices):
            # inference
            rgb_stack = rgb[:, :, start_idx:end_idx, :, :].to(self.device)

            # below is to creates a duplicate frame for each frame for data augmentation and to
            # avoid errors with too small stack sizes
            # Select frames from start_idx to end_idx (inclusive) along the frame dimension
            if (end_idx - start_idx < 10):
                rgb_stack = self.augment_rgb_frame(rgb, start_idx, end_idx)

            output = self.name2module['model'](rgb_stack, features=True)
            vid_feats.extend(output.tolist())
            self.maybe_show_pred(rgb_stack, start_idx, end_idx)

        feats_dict = {
            self.feature_type: np.array(vid_feats),
        }

        return feats_dict


    def load_model(self) -> Dict[str, torch.nn.Module]:
        """Defines the models, loads checkpoints, sends them to the device.

        Raises:
            NotImplementedError: if a model is not implemented.

        Returns:
            Dict[str, torch.nn.Module]: model-agnostic dict holding modules for extraction and show_pred
        """
        s3d_kinetics400_weights_torch_path = './models/s3d/checkpoint/S3D_kinetics400_torchified.pt'
        model = S3D(num_class=400, ckpt_path=s3d_kinetics400_weights_torch_path)
        model = model.to(self.device)
        model.eval()

        return {
            'model': model,
        }

    def maybe_show_pred(self, rgb_stack: torch.Tensor, start_idx: int, end_idx: int):
        if self.show_pred:
            logits = self.name2module['model'](rgb_stack, features=False)
            print(f'At frames ({start_idx}, {end_idx})')
            show_predictions_on_dataset(logits, 'kinetics')
