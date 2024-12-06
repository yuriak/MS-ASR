from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import os
import tqdm
import json
import logging
import vllm
import re
import time

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMReasoner:
    def __init__(self, args):
        self.args = args
        self.llm_reasoner_model = args.llm_reasoner_model
        default_post_process_mode = "default"
        self.support_system_prompts = False
        self.force_post_process = lambda x: x.strip()
        self.llm_type = args.llm_type

        if self.llm_type == "local":
            # "mistralai/Mistral-7B-Instruct-v0.2"
            self.llm = vllm.LLM(
                model=self.llm_reasoner_model,
                tokenizer=self.llm_reasoner_model,
                tensor_parallel_size=1,
                trust_remote_code=True,
                dtype=(
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                ),
                gpu_memory_utilization=0.6,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(args.llm_reasoner_model)
            self.sampling_decode_params = vllm.SamplingParams(
                max_tokens=2048,
                top_p=0.9,
                temperature=0.9,
            )

            self.greedy_decode_params = vllm.SamplingParams(
                max_tokens=2048,
                temperature=0.0,
            )

            self.beam_decode_params = vllm.SamplingParams(
                max_tokens=2048,
                n=5,
                temperature=0.0,
                use_beam_search=True,
                min_tokens=5,
            )

            self.decoding_params = {
                "sampling": self.sampling_decode_params,
                "beam": self.beam_decode_params,
                "greedy": self.greedy_decode_params,
            }

            if "Llama-3" in self.llm_reasoner_model:
                default_post_process_mode = "llama3"
                self.support_system_prompts = True
                self.force_post_process = lambda x: x.replace(
                    "<|start_header_id|>assistant<|end_header_id|>", ""
                )
        else:
            self.client = OpenAI(
                api_key=args.api_key,
                base_url=None if not args.api_address else args.api_address,
            )
            default_post_process_mode = "close"
            self.support_system_prompts = True

        self.default_post_process = {
            "close": lambda x: x,
            "default": lambda x: re.sub("\n+", "\n", x).split("\n")[0],
            "llama3": lambda x: x.split("\n")[-1].strip(),
        }.get(default_post_process_mode, lambda x: x)

    def batch_inference(
        self,
        prompts,
        decoding_method="sampling",
        post_process=None,
        **kwargs,
    ):
        post_process_func = (
            self.default_post_process if post_process is None else post_process
        )

        messages = []
        for i, prompt in enumerate(prompts):
            message = prompt
            if not self.support_system_prompts and prompt[0]["role"] == "system":
                message = [
                    {
                        "role": "user",
                        "content": "\n\n".join(
                            [prompt[0]["content"], prompt[1]["content"]]
                        ),
                    }
                ] + prompt[2:]
            messages.append(message)

        if self.llm_type == "local":
            input_prompts = [
                self.tokenizer.apply_chat_template(message, tokenize=False)
                for message in messages
            ]
            response = self.llm.generate(
                input_prompts,
                sampling_params=self.decoding_params.get(decoding_method),
                **kwargs,
            )
            response = list(
                map(
                    lambda x: post_process_func(
                        self.force_post_process(x.outputs[0].text.strip())
                    ),
                    response,
                )
            )
            return response
        else:

            responses = []
            for prompt in tqdm.tqdm(messages):

                response = self.client.chat.completions.create(
                    model=self.llm_reasoner_model, messages=prompt, **kwargs
                )
                response = response.choices[0].message.content.strip()
                responses.append(post_process_func(self.force_post_process(response)))
            return responses

    @staticmethod
    def add_args(parser):
        parser.add_argument("--api_key", default="xxx", type=str, help="OpenAI API key")
        parser.add_argument(
            "--llm_type", default="local", type=str, help="local or api"
        )
        parser.add_argument(
            "--llm_reasoner_model", default="meta-llama/Meta-Llama-3-8B-Instruct"
        )
        parser.add_argument("--api_address", default=None, type=str, help="API address")
