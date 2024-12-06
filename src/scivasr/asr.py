import os
import whisper
import json
import copy
from utils import *


def has_intersection(t1, t2):
    if t1[1] < t2[0] or t2[1] < t1[0]:
        return False
    else:
        return True


class ASR:
    def __init__(self, args):
        self.device = args.asr_model_device
        self.asr_lang = args.asr_lang
        self.model = whisper.load_model(args.asr_model).to(self.device)

    def __call__(self, video_path, video_id, scene_timestamps, output_dir):
        output_file = os.path.join(output_dir, video_id)

        if check_existence(output_file + ".scene_transcript.json"):
            scene_transcripts = read_json(output_file + ".scene_transcript.json")
            return scene_transcripts
        transcripts = self.model.transcribe(video_path, language=self.asr_lang)
        write_json(transcripts, output_file + ".transcript.json")
        scene_transcripts = self.get_scene_transcript(
            transcripts,
            scene_timestamps,
            save_transcript=True,
            video_id=video_id,
            output_dir=output_dir,
        )
        return scene_transcripts

    def get_scene_transcript(
        self,
        all_trans,
        scene_timestamp,
        save_transcript=False,
        video_id=0,
        output_dir=None,
    ):
        trans_seg = copy.deepcopy(all_trans["segments"])
        scene_transcripts = []

        for i, (start, end) in enumerate(scene_timestamp):
            transcript_scene_i = []
            all_gathered = False
            while len(trans_seg) > 0 and not all_gathered:
                if i == 0 and start > trans_seg[0]["start"]:
                    #                 might not be possible
                    seg = trans_seg.pop(0)
                    transcript_scene_i.append((seg["text"], seg["start"], seg["end"]))
                else:
                    if trans_seg[0]["start"] < end and trans_seg[0]["end"] - end < 3:
                        seg = trans_seg.pop(0)
                        transcript_scene_i.append(
                            (seg["text"], seg["start"], seg["end"])
                        )
                    elif trans_seg[0]["end"] > start and trans_seg[0]["end"] < end:
                        seg = trans_seg.pop(0)
                        transcript_scene_i.append(
                            (seg["text"], seg["start"], seg["end"])
                        )
                    else:
                        all_gathered = True
            scene_transcripts.append(transcript_scene_i)

        if len(trans_seg) > 0:
            while len(trans_seg) > 0:
                seg = trans_seg.pop(0)
                scene_transcripts[-1].append((seg["text"], seg["start"], seg["end"]))

        scene_transcripts = [
            " ".join([u[0].strip() for u in x]) for x in scene_transcripts
        ]
        if save_transcript:
            output_file = os.path.join(output_dir, video_id)
            write_json(
                scene_transcripts,
                output_file + ".scene_transcript.json",
            )
            write_lines(
                scene_transcripts,
                output_file + ".scene_transcript.txt",
            )

        assert len(trans_seg) == 0

        return scene_transcripts

    @staticmethod
    def add_args(parser):
        parser.add_argument("--asr_model", default="base")
        parser.add_argument("--asr_model_device", default="cuda:1")
        parser.add_argument("--asr_lang", default="English")
