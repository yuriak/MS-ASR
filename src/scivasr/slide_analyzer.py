from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from transformers.generation import GenerationConfig
import torch
import cv2
import os
import tqdm
import json
import logging
import vllm
from llm import LLMReasoner
from PIL import Image
import zlib
import base64
import requests
import re
from utils import *


logger = logging.getLogger(__name__)


def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


system_qa_based_summary_prompt = """
Your task is to craft a thorough, precise, and encompassing summary derived from the given questions and answers pertaining to a slide. 
It's crucial to incorporate all vital text from the slides into your summary. 
Please scrutinize the Q&A pairs closely, as some of the information presented may be incorrect, and ensure that your summary reflects only the accurate details.
"""

user_qa_based_summary_prompt = """
### Question and Answers:
{qa_pairs}

### Summary:"""


overall_summary_system_prompt = """
Using the individual summaries of each slide provided, your task is to synthesize an overall detailed summary that captures and condenses the key points, themes, and insights from the entire presentation. 
The final summary should weave together the content from the page-level summaries into a coherent narrative, highlighting the main ideas, significant findings, and any conclusions drawn across the slides. 
Ensure that the summary is comprehensive, accurately reflecting the breadth and depth of the material covered, and maintains the logical flow of information as presented in the sequence of slides. 
Your goal is to provide a clear and concise overview that offers readers a complete understanding of the presentation's content without needing to review each slide individually.
"""

overall_summary_user_prompt = """
## Slides Summaries:
{slide_summaries}

## Brief Introduction:
""".strip()

qa_pair_cleaning_sys_prompt = """
You are a professional QA pair content checker.
Your task is to deduplicate and clean the QA pairs given by the user.
Please make sure that only QA pairs with informative answers are retained.
Please return the cleaned QA pairs in a JSON list with the same format as what is given by the user.
Do not return any other content except from the JSON list.
"""


class SlideAnalyzer:
    def __init__(self, args, llm_reasoner: LLMReasoner):
        self.args = args
        self.analyzer_device = args.analyzer_device
        self.analyzer_vlm_retry = args.analyzer_vlm_retry
        self.analyzer_model = args.analyzer_model
        self.analyzer_compression_threshold = args.analyzer_compression_threshold
        self.analyzer_dtype = (
            torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )

        self.analyzer_type = args.analyzer_type
        self.vlm_decode_method = self.standard_vlm_decode
        self.vlm_batch_decode_method = self.standard_vlm_batch_decode
        if self.analyzer_type == "local":
            if (
                "cogvlm" in self.args.analyzer_model
                or "cogagent" in self.args.analyzer_model
            ):
                self.init_cogvlm()
                self.vlm_decode_method = self.cogvlm_decode
                self.vlm_batch_decode_method = self.cogvlm_batch_decode
            else:
                self.init_standard_local_vlm()
        else:
            self.init_api_vlm()
            self.analyzer_api_key = args.analyzer_api_key
            self.vlm_decode_method = self.api_vlm_decode
            self.vlm_batch_decode_method = self.api_vlm_batch_decode

        self.reasoner = llm_reasoner

        self.page_level_analyze_function = {
            "general_qa_summarize": self.extract_with_general_qa_summarize,
            "general_qa": self.extract_with_general_qa,
            "transcript_summarize": self.extract_with_transcript_summarize,
            "cot_ocr": self.extract_with_cot_ocr,
            "simple_ocr": self.extract_with_ocr,
        }.get(self.args.page_level_analyze_mode, self.extract_with_cot_ocr)

    # ===============================================================================
    # Visual Language Model Functions

    def init_standard_local_vlm(self):
        self.vl_tokenizer = AutoTokenizer.from_pretrained(
            self.args.analyzer_model, trust_remote_code=True
        )
        self.vl_model = AutoModelForCausalLM.from_pretrained(
            self.args.analyzer_model,
            device_map=self.analyzer_device,
            trust_remote_code=True,
            bf16=True if torch.cuda.is_bf16_supported() else False,
            fp16=False if torch.cuda.is_bf16_supported() else True,
        ).eval()

        vl_generation_config = GenerationConfig.from_pretrained(
            self.args.analyzer_model, trust_remote_code=True
        )
        self.vl_model.generation_config = vl_generation_config

    def init_cogvlm(self):
        self.vl_tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        self.vl_model = AutoModelForCausalLM.from_pretrained(
            self.args.analyzer_model,
            torch_dtype=self.analyzer_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=self.analyzer_device,
        ).eval()

    def init_api_vlm(self):
        pass

    def standard_vlm_decode(self, image_path, q, history=None):
        query = self.vl_tokenizer.from_list_format(
            [
                {"image": image_path},
                {"text": q},
            ]
        )
        response, history = self.vl_model.chat(self.vl_tokenizer, query, history=None)
        return response, history

    def standard_vlm_batch_decode(self, image_paths, queries, histories=None):
        responses = []
        for image_path, q in zip(image_paths, queries):
            response, _ = self.standard_vlm_decode(image_path, q, history=None)
            responses.append(response)
        return responses

    def cogvlm_decode(self, image_path, q, history=None):
        image = Image.open(image_path).convert("RGB")
        input_by_model = self.vl_model.build_conversation_input_ids(
            self.vl_tokenizer, query=q, history=[], images=[image]
        )  # chat mode
        inputs = {
            "input_ids": input_by_model["input_ids"]
            .unsqueeze(0)
            .to(self.analyzer_device),
            "token_type_ids": input_by_model["token_type_ids"]
            .unsqueeze(0)
            .to(self.analyzer_device),
            "attention_mask": input_by_model["attention_mask"]
            .unsqueeze(0)
            .to(self.analyzer_device),
            "images": [
                [
                    input_by_model["images"][0]
                    .to(self.analyzer_device)
                    .to(self.analyzer_dtype)
                ]
            ],
        }
        if "cross_images" in input_by_model and input_by_model["cross_images"]:
            inputs["cross_images"] = [
                [
                    input_by_model["cross_images"][0]
                    .to(self.analyzer_device)
                    .to(self.analyzer_dtype)
                ]
            ]
        gen_kwargs = {"max_new_tokens": 2048, "do_sample": True, "temperature": 0.9}
        with torch.no_grad():
            outputs = self.vl_model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = self.vl_tokenizer.decode(outputs[0]).replace("</s>", "").strip()
        return response, None

    def cogvlm_batch_decode(self, image_paths, queries, histories=None):

        def collate_fn(features, tokenizer) -> dict:
            images = [feature.pop("images") for feature in features]
            tokenizer.padding_side = "left"
            padded_features = tokenizer.pad(features)
            inputs = {**padded_features, "images": images}
            return inputs

        def recur_move_to(item, tgt, criterion_func):
            if criterion_func(item):
                device_copy = item.to(tgt)
                return device_copy
            elif isinstance(item, list):
                return [recur_move_to(v, tgt, criterion_func) for v in item]
            elif isinstance(item, tuple):
                return tuple([recur_move_to(v, tgt, criterion_func) for v in item])
            elif isinstance(item, dict):
                return {
                    k: recur_move_to(v, tgt, criterion_func) for k, v in item.items()
                }
            else:
                return item

        assert len(image_paths) == len(queries)
        image_list = [
            Image.open(image_path).convert("RGB") for image_path in image_paths
        ]

        input_samples = [
            self.vl_model.build_conversation_input_ids(
                self.vl_tokenizer, query=query, history=[], images=[image]
            )
            for image, query in zip(image_list, queries)
        ]

        input_batch = collate_fn(input_samples, self.vl_tokenizer)
        input_batch = recur_move_to(
            input_batch, self.analyzer_device, lambda x: isinstance(x, torch.Tensor)
        )
        input_batch = recur_move_to(
            input_batch,
            self.analyzer_dtype,
            lambda x: isinstance(x, torch.Tensor) and torch.is_floating_point(x),
        )
        gen_kwargs = {"max_new_tokens": 2048, "temperature": 0.9, "do_sample": False}
        with torch.no_grad():
            outputs = self.vl_model.generate(**input_batch, **gen_kwargs)
            outputs = outputs[:, input_batch["input_ids"].shape[1] :]

        responses = [
            x.replace("</s>", "").replace("<unk>", "")
            for x in self.vl_tokenizer.batch_decode(outputs)
        ]

        return responses

    def api_vlm_decode(self, image_path, q, history=None):
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        base64_image = encode_image(image_path)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.analyzer_api_key}",
        }

        payload = {
            "model": self.analyzer_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": q},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 4096,
        }

        url = "https://api.openai.com/v1/chat/completions"
        for i in range(3):
            try:
                response = requests.post(url, headers=headers, json=payload)
                break
            except Exception:
                print(f"{image_path} occured error")

        results = response.json()
        response_str = results["choices"][0]["message"]["content"]
        # parsed_result = parse_result(response_str)
        return response_str, None

    def api_vlm_batch_decode(self, image_paths, queries, histories=None):
        responses = []
        for image_path, q in zip(image_paths, queries):
            response, _ = self.api_vlm_decode(image_path, q, history=None)
            responses.append(response)
        return responses

    def vlm_decode(self, image_path, q, history=None):
        for i in range(self.analyzer_vlm_retry):
            response, history = self.vlm_decode_method(image_path, q, history=history)
            if compression_ratio(response) < self.analyzer_compression_threshold:
                return response, history
            else:
                logger.warning(
                    f"Encountered repetitive pattern in vlm response {response}"
                )
        return response, history

    def vlm_batch_decode(self, image_paths, queries, histories=None):
        return self.vlm_batch_decode_method(image_paths, queries, histories=histories)

    # ===================================================================================
    # Page Level Analysis Functions

    def extract_with_general_qa_summarize(
        self, scene_image_files=None, video_id=None, output_dir=None, **kwargs
    ):

        def page_level_summary_promptify(qa_pairs):

            qa_pairs = "\n".join(
                [f"* Question: {x[0]}\n* Answer: {x[1]}" for x in qa_pairs]
            )

            return [
                {"role": "system", "content": system_qa_based_summary_prompt},
                {
                    "role": "user",
                    "content": user_qa_based_summary_prompt.format(
                        qa_pairs=qa_pairs.strip()
                    ),
                },
            ]

        slides_qa_pairs = self._extract_with_qa(
            scene_image_files=scene_image_files,
            video_id=video_id,
            output_dir=output_dir,
        )
        summarizer_inputs = [
            page_level_summary_promptify(slide_qa_pairs)
            for slide_qa_pairs in slides_qa_pairs
        ]

        summaries = self.reasoner.batch_inference(
            summarizer_inputs, decoding_method="sampling", post_process=lambda x: x
        )

        return summaries

    def _extract_with_qa(
        self, scene_image_files=None, video_id=None, output_dir=None, **kwargs
    ):
        questions_for_vLLM = [
            "Please describe the layout of the slide."
            "What is the main topic or headline of the slide?",
            "Can you list the key points or bullet points presented on the slide?",
            "Are there any important dates, statistics, or quantitative data mentioned? Please summarize.",
            "Could you identify and list all the affiliations mentioned in the slide?",
            "Are there any visual aids (e.g., charts, graphs, images, diagrams) on the slide? What information do they convey?",
            "Does the slide include any quotes, citations, or references to other works? Please detail.",
            "Please provide a holistic summary that encapsulates the entire content and context of the slide.",
        ]

        output_file = os.path.join(output_dir, video_id)
        if check_existence(output_file + ".slide_qa_pairs.json"):
            slides_qa_pairs = read_json(output_file + ".slide_qa_pairs.json")
        else:

            slides_qa_pairs = []
            history = None
            for image_path in tqdm.tqdm(scene_image_files):
                slide_qa_pairs = []
                for question in questions_for_vLLM:
                    response, history = self.vlm_decode(
                        image_path, question, history=history
                    )
                    slide_qa_pairs.append((question, response))
                slides_qa_pairs.append(slide_qa_pairs)
            write_json(slides_qa_pairs, output_file + ".slide_qa_pairs.json")
        return slides_qa_pairs

    def extract_with_general_qa(
        self, scene_image_files=None, video_id=None, output_dir=None, **kwargs
    ):
        slides_qa_pairs = self._extract_with_qa(
            scene_image_files=scene_image_files,
            video_id=video_id,
            output_dir=output_dir,
        )

        def page_level_qa_cleaning_promptify(qa_pairs):
            return [
                {"role": "system", "content": qa_pair_cleaning_sys_prompt},
                {
                    "role": "user",
                    "content": str(qa_pairs),
                },
            ]

        summarizer_inputs = [
            page_level_qa_cleaning_promptify(slide_qa_pairs)
            for slide_qa_pairs in slides_qa_pairs
        ]

        cleaned_qa_pairs = self.reasoner.batch_inference(
            summarizer_inputs,
            decoding_method="greedy",
            post_process=lambda x: x
        )

        def parse_result(result):
            try:
                for x in ["```json", "```python"]:
                    result = result.replace(x, "```")
                if "```" in result:
                    result = result.split("```")[1].strip()
                    result = re.sub(r"\n\s*", "", result)

                result = json.loads(result)
                if type(result[0]) == dict:
                    list_result = []
                    for qa_object in result:
                        list_result.append(list(list(qa_object.items())[0]))
                    result = "\n".join(
                        [f"* {x[1]}" for x in list_result]
                    )
                    return result
                elif type(result[0]) == list:
                    result = "\n".join(
                        [f"* {x[1]}" for x in result]
                    )
                    return result
            except:
                return f"* {result}"

        final_summaries = []
        for cleaned_qa_pair in cleaned_qa_pairs:
            summary = parse_result(cleaned_qa_pair)
            final_summaries.append(summary)

        return final_summaries

    def extract_with_transcript_summarize(
        self, scene_image_files=None, video_id=None, transcripts=None, **kwargs
    ):
        transcript_based_summary_prompt = """### Speaker's Transcript:\n"{transcript}"\n\nSummarize the content of this image."""
        slides_level_summaries = []
        history = None
        for image_path, transcript in tqdm.tqdm(zip(scene_image_files, transcripts)):
            prompt = transcript_based_summary_prompt.format(transcript=transcript)
            response, history = self.vlm_decode(image_path, prompt, history=history)
            slides_level_summaries.append(response)
        return slides_level_summaries

    def extract_with_ocr(self, scene_image_files=None, **kwargs):
        slides_level_summaries = []
        history = None
        for image_path in tqdm.tqdm(scene_image_files):
            prompt = "OCR this image, make sure to extract all important text."
            response, history = self.vlm_decode(image_path, prompt, history=history)
            slides_level_summaries.append(response)
        return slides_level_summaries

    def extract_with_cot_ocr(self, scene_image_files=None, **kwargs):
        detection_prompts = {
            "Overall": 'Does the slide contain lots of text? Return "Yes" or "No" only.',
            "Titles": 'Does the slide contain any Titles or Headlines? Return "Yes" or "No" only.',
            "Figures": 'Does the slide contain any Figures? Return "Yes" or "No" only.',
            "Tables": 'Does the slide contain any Tables? Return "Yes" or "No" only.',
            "References": 'Does the slide contain any References? Return "Yes" or "No" only.',
        }
        slides_level_summaries = []
        ocr_prompts = {
            "Titles": "OCR this image, focus on major bullet points and headlines, try to retain the original structure and order of bullet points.",
            "Figures": "OCR this image, focus on figures and diagrams.",
            "Tables": "OCR this image, focus on tables.",
            "References": "OCR this image, focus on references and citations.",
            "Summary": "Write an overall introduction on the given image.",
        }
        format_results = lambda x: "\n\n".join(
            [f"#### {k.strip()}\n{v.strip()}" for k, v in x.items()]
        )

        def finegrained_ocr_extract(image_path):
            responses = [
                self.vlm_decode(image_path, prompt)[0]
                for prompt in detection_prompts.values()
            ]
            # responses = self.vlm_batch_decode(
            #     [image_path] * len(detection_prompts), list(detection_prompts.values())
            # )
            detection_result = dict(
                zip(detection_prompts.keys(), ["Yes" in x for x in responses])
            )
            results = {}
            results["Summary"] = self.vlm_decode(image_path, ocr_prompts["Summary"])[0]

            if not detection_result["Overall"]:
                return format_results(results)

            for key, prompt in ocr_prompts.items():
                if key in results:
                    continue
                if key in detection_result and detection_result[key]:
                    response, _ = self.vlm_decode(image_path, prompt)
                    results[key] = response
            return format_results(results)

        for image_path in tqdm.tqdm(scene_image_files):
            slide_level_summary = finegrained_ocr_extract(image_path)
            slides_level_summaries.append(slide_level_summary)
        return slides_level_summaries

    def create_overall_summary(self, scene_level_summaries):
        overview_summarizer_input = [
            {
                "role": "system",
                "content": overall_summary_system_prompt,
            },
            {
                "role": "user",
                "content": overall_summary_user_prompt.format(
                    slide_summaries="\n\n".join(
                        [
                            f"### Page: {i+1}: {s.strip()}"
                            for i, s in enumerate(scene_level_summaries)
                        ]
                    )
                ),
            },
        ]
        overview_summary = self.reasoner.batch_inference(
            [overview_summarizer_input],
            decoding_method="sampling",
            post_process=lambda x: x.strip(),
        )[0]

        return overview_summary

    def __call__(
        self, scene_image_files, scene_timestamps, transcripts, video_id, output_dir
    ):
        output_file = os.path.join(output_dir, video_id)
        if check_existence(output_file + ".slide_summary.json"):
            scene_level_summaries = read_json(output_file + ".slide_summary.json")
            logger.info(f"Num of Slides Content: {len(scene_level_summaries)}")
        else:
            for x in scene_image_files:
                assert os.path.exists(x), f"{x} not found"

            scene_level_summaries = self.page_level_analyze_function(
                scene_image_files=scene_image_files,
                video_id=video_id,
                scene_timestamps=scene_timestamps,
                transcripts=transcripts,
                output_dir=output_dir,
            )

            write_json(scene_level_summaries, output_file + ".slide_summary.json")

            write_lines(
                [u.replace("\n", " ") for u in scene_level_summaries],
                output_file + ".slide_summary.txt",
            )

        if check_existence(output_file + ".overall_summary.txt"):
            overall_summary = read_file(output_file + ".overall_summary.txt")
            return scene_level_summaries, overall_summary

        overall_summary = self.create_overall_summary(scene_level_summaries)
        write_file(overall_summary, output_file + ".overall_summary.txt")
        logger.info(f"Num of Slides Content: {len(scene_level_summaries)}")
        return scene_level_summaries, overall_summary

    @staticmethod
    def add_args(parser):
        parser.add_argument("--analyzer_device", default="cuda:0")
        parser.add_argument("--analyzer_model", default="THUDM/cogagent-vqa-hf")
        parser.add_argument("--analyzer_api_key", default=None)
        parser.add_argument("--analyzer_type", default="local")
        parser.add_argument("--page_level_analyze_mode", default="general_qa")
        parser.add_argument("--analyzer_vlm_retry", default=3)
        parser.add_argument("--analyzer_compression_threshold", default=3.0)
