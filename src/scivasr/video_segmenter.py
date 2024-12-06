import os
import cv2
import pdb
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from transformers import logging
from kts_src.kts_utils import cpd_auto, l2_normalize_np_array
import json
import tqdm
import logging
from utils import *

logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(self, args):
        self.device = args.feature_extractor_device
        self.beta = args.beta
        self.processor = CLIPProcessor.from_pretrained(args.feature_extractor)
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            args.feature_extractor
        ).to(self.device)

    def extract_video_features(self, video_path, video_id, output_dir):

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length = frame_count / fps
        sample_rate = int(fps) * self.beta

        save_path = os.path.join(output_dir, video_id + ".npz")
        if os.path.exists(save_path):
            data = np.load(save_path)
            clip_features = data["features"]
            return clip_features, video_length

        clip_features = []
        with tqdm.tqdm(total=frame_count) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_fame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                pbar.update(1)
                if current_fame_idx % sample_rate == 0:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    inputs = self.processor(
                        images=image, return_tensors="pt"
                    ).pixel_values
                    inputs = inputs.to(self.device)

                    with torch.no_grad():
                        feat = self.model(inputs)["image_embeds"]
                        clip_features.append(feat.cpu().numpy())

        clip_features = np.concatenate(clip_features, axis=0)
        np.savez_compressed(save_path, features=clip_features)

        return clip_features, video_length, fps

    def extract_frame_features(self, video_path, seg_windows):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i, (start, end) in enumerate(seg_windows):
            cut_point = start + (end - start) * 0.95
            middle_frame_idx = int(cut_point * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            ret, frame = cap.read()
            frames.append(frame)
        frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        inputs = self.processor(images=frames, return_tensors="pt").pixel_values.to(
            self.device
        )
        feat = self.model(inputs)["image_embeds"]
        return feat.cpu()


class VideoSegmenter:
    def __init__(self, args):
        self.feature_extractor = FeatureExtractor(args)
        self.alpha = args.alpha
        self.beta = args.beta
        self.dedup_threshold = args.dedup_threshold
        self.max_dedup_iter = args.max_dedup_iter

    def __call__(self, video_path, video_id, output_dir):

        output_name = os.path.join(output_dir, video_id)
        if check_existence(output_name + ".scene.json"):
            seg_windows = read_json(output_name + ".scene.json")
            logger.info(f"Num of Scenes: {len(seg_windows)}")
            scene_image_files = self._save_scene_images(
                video_path, seg_windows, output_name
            )
            return seg_windows, scene_image_files
        seg_windows = self._segment(video_path, video_id, output_dir)
        write_json(seg_windows, output_name + ".full_scene.json")
        logger.info(f"Num of Scenes: {len(seg_windows)}")
        seg_windows = self._deduplicate_segments(video_path, seg_windows)
        logger.info(f"Num of Scenes after Deduplication: {len(seg_windows)}")
        write_json(seg_windows, output_name + ".scene.json")
        scene_image_files = self._save_scene_images(
            video_path, seg_windows, output_name
        )
        return seg_windows, scene_image_files

    def _save_scene_images(self, video_path, seg_windows, output_name):

        # Check whether all the scene images are already saved
        scene_image_files = []
        for i, (start, end) in enumerate(seg_windows):
            if not check_existence(f"{output_name}_{i}.jpg"):
                break
            scene_image_files.append(f"{output_name}_{i}.jpg")
        if len(scene_image_files) == len(seg_windows):
            return scene_image_files

        # Save the segment windows into images
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        scene_image_files = []
        for i, (start, end) in enumerate(seg_windows):
            cut_point = start + (end - start) * 0.95
            sampled_frame_idx = int(cut_point * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, sampled_frame_idx)
            ret, frame = cap.read()
            cv2.imwrite(f"{output_name}_{i}.jpg", frame)
            scene_image_files.append(f"{output_name}_{i}.jpg")
        return scene_image_files

    def _deduplicate_segments(self, video_path, seg_windows):

        for i in range(self.max_dedup_iter):
            features = self.feature_extractor.extract_frame_features(
                video_path, seg_windows
            )
            if features.shape[0] < 2:
                return seg_windows
            similar_slides = (
                torch.arange(1, features.shape[0])
                .to(features.device)[
                    torch.cosine_similarity(features[:-1], features[1:])
                    > self.dedup_threshold
                ]
                .tolist()
            )
            if len(similar_slides) == 0:
                return seg_windows
            # Merge the consuctive seg_windows if they are indicated as similar slides
            seg_windows_with_tag = list(
                map(lambda x: (x[0] in similar_slides, x[1]), enumerate(seg_windows))
            )
            for i in list(range(len(seg_windows_with_tag)))[::-1]:
                tag, (start, end) = seg_windows_with_tag[i]
                if tag:
                    seg_windows_with_tag[i - 1][1][1] = end
            seg_windows = list(
                map(lambda u: u[1], filter(lambda x: not x[0], seg_windows_with_tag))
            )

        assert all(
            [
                (x[1] - y[0]) == 0
                for x, y in list(zip(seg_windows[:-1], seg_windows[1:]))
            ]
        ), "Discontinuous segments are detected."

        return seg_windows

    def _segment(self, video_path, video_id, output_dir):

        video_features, video_length, fps = (
            self.feature_extractor.extract_video_features(
                video_path=video_path, video_id=video_id, output_dir=output_dir
            )
        )
        K = l2_normalize_np_array(video_features)
        K = np.dot(K, K.T)
        clip_num = K.shape[0]
        max_seg_num = clip_num // self.alpha

        # cps, _ = cpd_auto(K, max_seg_num - 1, 0)
        cps, _ = cpd_auto(K, max_seg_num - 1, vmax=1)
        seg_num = len(cps) + 1

        seg_points = [x * self.beta for x in cps]
        seg_points = np.insert(seg_points, 0, 0)
        seg_points = np.append(seg_points, video_length)

        seg_windows = [
            [int(seg_points[i]), int(seg_points[i + 1])] for i in range(seg_num)
        ]
        logger.info(f"Num of Scenes: {len(seg_windows)}")
        return seg_windows

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--alpha",
            default=10,
            type=int,
            help="Determine the maximum segment number for KTS algorithm, the larger the value, the fewer segments.",
        )
        parser.add_argument(
            "--beta",
            default=1,
            type=int,
            help="The smallest time gap between successive clips, in seconds.",
        )
        parser.add_argument(
            "--feature_extractor",
            default="openai/clip-vit-base-patch32",
            help="Select the feature extractor model for video segmentation",
        )
        parser.add_argument(
            "--feature_extractor_device",
            default="cuda:1",
            help="Select the device: cuda or cpu",
        )
        parser.add_argument("--dedup_threshold", default=0.9)
        parser.add_argument("--max_dedup_iter", default=5)
