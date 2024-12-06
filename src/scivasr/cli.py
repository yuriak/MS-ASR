import argparse
from scivasr.core import SciVASR
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=False)
    parser.add_argument("--video_id", required=False)
    parser.add_argument("--video_batch_job", required=False)
    parser.add_argument(
        "--output_dir",
        default="./results",
        type=str,
        help="Directory for saving videos and logs.",
    )
    SciVASR.add_args(parser)
    args = parser.parse_args()
    sci_vasr = SciVASR(args)

    if args.video_batch_job:
        batch_job = json.load(open(args.video_batch_job, "r"))
        videos = batch_job["videos"]
        language_pairs = batch_job["language_pairs"]
        for video_id, video_path in videos.items():
            logger.info(
                f"Transcribing {video_id}:{video_path}"
            )
            sci_vasr.run(
                video_path,
                video_id,
                output_dir=os.path.join(args.output_dir, video_id),
            )

    elif args.video_path and args.video_id:
        sci_vasr.run(args.video_path, args.video_id, output_dir=args.output_dir)
    else:
        raise ValueError(
            "Please provide either video_path and video_id or video_series_json"
        )
