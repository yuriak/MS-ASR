from typing import Any
from llm import LLMReasoner
import os
import json
import re
from utils import *


system_multiround_cot_visual_pe = """
You are a professional ASR transcript post-editor.
You are tasked with post-editing the ASR transcript of a presentation of an ACL paper.
We will finish the post-editing in multiple rounds of dialogue.
Make sure you follow the instructions.
"""

user_multiround_cot_visual_pe = """
Please carefully review the following paragraph from an ASR transcript. 
Identify and list any errors that may have occurred during the speech-to-text conversion process. 
This includes misrecognitions with homophones, omissions, and inappropriate insertions of words. 
Additionally, highlight any areas where the context or sentence structure seems unclear due to potential ASR errors.
Don't try to fix the errors, just list all of them!

### Transcript:
{transcript}
"""

user_multiround_cot_visual_pe_answer_question = """
There are some textual information including an overall summary and slide content of the presentaion:

### Overall Summary
{overall_summary}

### Slide Content
{slide_summary}

Please answer your questions about ASR errors using cues from them.
"""

system_cot_visual_pe = """
You are a professional ASR transcript post-editor. 
You are tasked with post-editing the ASR transcript of a presentation of an ACL paper.
Focus on rectifying inaccuracies introduced by ASR (Automatic Speech Recognition) errors within the transcript in Transcript section.
Utilize the comprehensive Presentation Summary and the individual Slide Content as references for your corrections.
Aim for minimal alterations; restrict your changes solely to essential corrections (e.g. terminologies, names, entities) without rewording the original sentences in the transcript.
You should first identify the possible errors, list them in a section named ### Error Identification.
After identifing errors, you should propose the correct form for each error in a section named ### Proposed Corrections.
Finally return the post-edited transcript based on your correction in the last section titled with ### Transcript PE.
If the transcript is error-free, simply submit it in the Transcript PE section.
"""

user_cot_visual_pe = """
### Presentation Summary
{overall_summary}

### Slide Content:
{slide_summary}

### Transcript:
{transcript}
"""

system_visual_pe = """
You are tasked with post-editing the ASR transcript of a presentation of an ACL paper.
Focus on rectifying inaccuracies introduced by ASR (Automatic Speech Recognition) errors within the current page slide's transcript.
Utilize the comprehensive presentation summary and the individual slide summaries as references for your corrections.
If the transcript is error-free, simply submit it as is.
Your edits should strictly address corrections without adding extraneous content such as explanations or notes.
Aim for minimal alterations; restrict your changes solely to essential corrections (e.g. terminologies, names, entities) without rewording the original sentences in the transcript.
Just return the fixed ASR transcript without any other explanations.
DO NOT return any other irrelevant content!
""".strip()

user_visual_pe = """
## Overall Presentation Summary
{overall_summary}

## Page-level Slides Summary (Current Page):
{slide_summary}

## Transcripts (Current page):
{transcript}

## Post-edited Transcripts (Current page):
""".strip()


system_visual_pe_no_overall = """
You are tasked with post-editing the ASR transcript of a presentation of an ACL paper.
Focus on rectifying inaccuracies introduced by ASR (Automatic Speech Recognition) errors within the current page slide's transcript.
Utilize the individual slide summaries as references for your corrections.
If the transcript is error-free, simply submit it as is.
Your edits should strictly address corrections without adding extraneous content such as explanations or notes.
Aim for minimal alterations; restrict your changes solely to essential corrections (e.g. terminologies, names, entities) without rewording the original sentences in the transcript.
Just return the fixed ASR transcript without any other explanations.
DO NOT return any other irrelevant content!
""".strip()

user_visual_pe_no_overall = """
## Page-level Slides Summary (Current Page):
{slide_summary}

## Transcripts (Current page):
{transcript}

## Post-edited Transcripts (Current page):
""".strip()


# system_visual_pe_simple = """
# Your task is to fix mis-spelled terminologies and names in the user provided ASR transcript.
# A slide summary will be provided as the context for your post-editing.
# Focus on incorrect terminologies and names, don't make unnecessary changes to the content such as rephrasing, or reordering.
# If there is no error in the transcript, just copy it.
# Just return the fixed ASR transcript without any other explanations.
# DO NOT return any other irrelevant content!
# """


system_text_pe = """
You are tasked with post-editing the ASR transcript of a presentation of an ACL paper.
Focus on rectifying inaccuracies introduced by ASR (Automatic Speech Recognition) errors within the current page slide's transcript.
If the transcript is error-free, simply submit it as is.
Your edits should strictly address corrections without adding extraneous content such as explanations or notes.
Aim for minimal alterations; restrict your changes solely to essential corrections (e.g. terminologies, names, entities) without rewording the original sentences in the transcript.
Just return the fixed ASR transcript without any other explanations.
DO NOT return any other irrelevant content!
""".strip()

user_text_pe = """
## Transcripts (Current page):
{transcript}

## Post-edited Transcripts (Current page):
""".strip()


class ASRPostEditor:
    def __init__(self, args, llm_reasoner: LLMReasoner):
        self.args = args
        self.reasoner = llm_reasoner
        self.subtitle_form = False
        self.post_edit_mode = args.post_edit_mode

        self.post_editing_function = {
            "visual": self.visual_based_post_edit,
            "visual_no_overall": self.visual_based_post_edit,
            "cot_visual": self.cot_visual_based_post_edit,
            "multi_round_cot_visual": self.multi_round_cot_visual_based_post_edit,
            "text_only": self.text_only_post_edit,
            "do_nothing": self.do_nothing_post_edit,
        }.get(self.post_edit_mode, self.do_nothing_post_edit)

    def __call__(
        self,
        scene_transcripts,
        page_level_slide_summaries,
        overall_summary,
        video_id,
        output_dir,
    ) -> Any:
        output_file = os.path.join(output_dir, video_id)
        if check_existence(output_file + ".transcript_pe.json"):
            transcripts_pe = read_json(output_file + ".transcript_pe.json")
            return transcripts_pe

        model_inputs, model_outputs = self.post_editing_function(
            scene_transcripts, page_level_slide_summaries, overall_summary, output_file
        )
        transcripts_pe = model_outputs

        write_json(transcripts_pe, output_file + ".transcript_pe.json")

        write_json(
            [{"input": x, "output": y} for x, y in zip(model_inputs, model_outputs)],
            output_file + ".transcript_pe.record.json",
        )

        write_lines(transcripts_pe, output_file + ".transcript_pe.txt")

        return transcripts_pe

    def do_nothing_post_edit(
        self,
        scene_transcripts,
        page_level_slide_summaries,
        overall_summary,
        output_file,
    ):
        return scene_transcripts, scene_transcripts

    def text_only_post_edit(
        self,
        scene_transcripts,
        page_level_slide_summaries,
        overall_summary,
        output_file,
    ):
        model_inputs = [
            [
                {
                    "role": "system",
                    "content": system_text_pe,
                },
                {
                    "role": "user",
                    "content": user_text_pe.format(
                        transcript=transcript,
                    ),
                },
            ]
            for i, transcript in enumerate(scene_transcripts)
        ]

        model_outputs = self.reasoner.batch_inference(model_inputs, "sampling")

        return model_inputs, model_outputs

    def visual_based_post_edit(
        self,
        scene_transcripts,
        page_level_slide_summaries,
        overall_summary,
        output_file,
    ):
        if self.post_edit_mode == "visual_no_overall":
            system_message = system_visual_pe_no_overall
            user_message = user_visual_pe_no_overall
        else:
            system_message = system_visual_pe
            user_message = user_visual_pe


        model_inputs = [
            [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_message.format(
                        overall_summary=overall_summary,
                        slide_summary=slide_summary,
                        transcript=transcript,
                    ),
                },
            ]
            for i, (slide_summary, transcript) in enumerate(
                zip(page_level_slide_summaries, scene_transcripts)
            )
        ]
        def parse_result(model_output:str):
            if '\n' in model_output:
                return model_output.split('\n')[-1]
            else:
                return model_output


        model_outputs = self.reasoner.batch_inference(model_inputs, "sampling", post_process=parse_result)

        return model_inputs, model_outputs

    def cot_visual_based_post_edit(
        self,
        scene_transcripts,
        page_level_slide_summaries,
        overall_summary,
        output_file,
    ):
        # Identify the ### Transcript PE section and get the last row of the section
        def extract_post_edited_transcripts(model_response):
            pe_title = re.search("### Transcript PE", model_response)
            if not pe_title:
                return model_response.split("\n")[-1]
            transcript_pe = model_response[pe_title.start() :].split("\n")[-1]
            return transcript_pe

        model_inputs = [
            [
                {
                    "role": "system",
                    "content": system_cot_visual_pe.strip(),
                },
                {
                    "role": "user",
                    "content": user_cot_visual_pe.format(
                        overall_summary=overall_summary,
                        slide_summary=slide_summary,
                        transcript=transcript,
                    ).strip(),
                },
            ]
            for i, (slide_summary, transcript) in enumerate(
                zip(page_level_slide_summaries, scene_transcripts)
            )
        ]

        model_outputs = self.reasoner.batch_inference(
            model_inputs, "sampling", post_process=extract_post_edited_transcripts
        )

        return model_inputs, model_outputs

    def multi_round_cot_visual_based_post_edit(
        self,
        scene_transcripts,
        page_level_slide_summaries,
        overall_summary,
        output_file,
    ):

        def create_initial_message(transcript):
            initial_message = [
                {"role": "system", "content": system_multiround_cot_visual_pe.strip()},
                {
                    "role": "user",
                    "content": user_multiround_cot_visual_pe.format(
                        transcript=transcript,
                    ).strip(),
                },
            ]
            return initial_message

        def create_new_message(previous_response, user_instruction):
            new_message = [
                {"role": "assistant", "content": previous_response},
                {
                    "role": "user",
                    "content": user_instruction,
                },
            ]
            return new_message

        def append_history(histories, new_messages):
            return [
                history + new_message
                for history, new_message in zip(histories, new_messages)
            ]

        messages = [
            create_initial_message(transcript=transctipt)
            for transctipt in scene_transcripts
        ]
        identified_errors = self.reasoner.batch_inference(
            messages, "sampling", post_process=lambda x: x
        )
        question_messages = [
            create_new_message(
                identified_error,
                user_instruction="Please write a question to ask a vision large language model to retrieve"
                + " corresponding textual context from the slide for errors you are not sure. Return your questions"
                + " in a Python list, don't return any other content, as I will parse your response.".strip(),
            )
            for i, identified_error in enumerate(identified_errors)
        ]
        messages = append_history(messages, question_messages)
        questions = self.reasoner.batch_inference(
            messages, "sampling", post_process=lambda x: x
        )
        answer_messages = [
            create_new_message(
                questions[i],
                user_instruction=user_multiround_cot_visual_pe_answer_question.format(
                    overall_summary=overall_summary,
                    slide_summary=page_level_slide_summaries[i],
                ).strip(),
            )
            for i in range(len(questions))
        ]
        messages = append_history(messages, answer_messages)
        answers = self.reasoner.batch_inference(
            messages, "sampling", post_process=lambda x: x
        )
        pe_messages = [
            create_new_message(
                answers[i],
                user_instruction="Use your answer to correct the ASR transcript. Only return the fixed transcript without any other explanation.",
            )
            for i in range(len(answers))
        ]
        messages = append_history(messages, pe_messages)
        post_edited_transcipts = self.reasoner.batch_inference(
            messages, "sampling", post_process=lambda x: x.split("\n")[-1]
        )

        return messages, post_edited_transcipts

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--post_edit_mode",
            default="cot_visual",
            type=str,
            help="How to post edit ASR transcript",
        )
