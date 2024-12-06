import os
import cv2
import pdb
import sys
import time
import numpy as np
from transformers import logging

from video_segmenter import VideoSegmenter
from asr import ASR
from slide_analyzer import SlideAnalyzer
from llm import LLMReasoner
from post_editor import ASRPostEditor

logger = logging.get_logger(__name__)


class SciVASR:
    def __init__(self, args):
        self.args = args
        self.llm_reasoner = LLMReasoner(args)
        self.video_segmenter = VideoSegmenter(self.args)
        self.asr = ASR(self.args)
        self.slide_analyzer = SlideAnalyzer(self.args, llm_reasoner=self.llm_reasoner)
        self.asr_post_editor = ASRPostEditor(self.args, llm_reasoner=self.llm_reasoner)
        self.stop_at = args.stop_at

    def run(
        self,
        video_path,
        video_id,
        output_dir,
    ):
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Segmenting video...")
        scene_timestamps, scene_image_files = self.video_segmenter(
            video_path, video_id, output_dir
        )

        if self.check_stop("segmenter"):
            return
        logger.info("Transcribing audio...")
        scene_transcripts = self.asr(video_path, video_id, scene_timestamps, output_dir)
        if self.check_stop("asr"):
            return
        logger.info("Analyzing scene...")
        scene_summaries, overall_summary = self.slide_analyzer(
            scene_image_files, scene_timestamps, scene_transcripts, video_id, output_dir
        )
        if self.check_stop("scene_analyzer"):
            return
        logger.info("Post-editing...")
        post_edited_scene_transcripts = self.asr_post_editor(
            scene_transcripts, scene_summaries, overall_summary, video_id, output_dir
        )
        if self.check_stop("asr_post_editor"):
            return
        logger.info("Translating...")
        logger.info(f"{video_path} :: {video_id} is Done!")
        open(os.path.join(output_dir, video_id + ".record"), "w").write(
            f"{video_id}:{video_path}"
        )

    def check_stop(self, stage):
        if self.stop_at == stage:
            logger.info(f"Stopped at {stage}")
            return True
        return False

    @staticmethod
    def add_args(parser):
        parser.add_argument("--stop_at", type=str, default=None, help="Stop at a specific stage")
        LLMReasoner.add_args(parser)
        ASR.add_args(parser)
        VideoSegmenter.add_args(parser)
        SlideAnalyzer.add_args(parser)
        ASRPostEditor.add_args(parser)
