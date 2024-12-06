import pandas as pd
import re
import jiwer
import numpy as np
import argparse
import os
import tqdm
import openai
from openai import OpenAI
import json
import itertools


video_to_slit = {
    "268": "dev",
    "367": "dev",
    "590": "dev",
    "110": "dev",
    "117": "dev",
    "410": "eval",
    "468": "eval",
    "567": "eval",
    "597": "eval",
    "111": "eval",
}


def collect_doc(data):
    doc_level_content = {}
    current_doc_id = None
    for x in data:
        if x.startswith("<doc"):
            doc_id = re.findall(
                '<doc docid="2022.acl-long.(\d+)" genre="presentations">', x
            )[0]
            doc_level_content[doc_id] = []
            current_doc_id = doc_id
        else:
            content = re.sub('<seg id="\d+">|</seg>', "", x)
            doc_level_content[current_doc_id].append(content)
    return doc_level_content


def reannotate_sentences(A, B):
    tokens_B = [word for line in B for word in line.split()]

    annotated_A = []
    for a_sentence in A:
        tokens_A = a_sentence.split()
        result = []
        for token in tokens_A:
            if tokens_B:
                b_token = tokens_B.pop(0)
                if b_token.startswith("[") or b_token.endswith("]"):
                    result.append(b_token)
                else:
                    result.append(token)
            else:
                result.append(token)

        annotated_A.append(" ".join(result))
    return annotated_A


def extract_terms(text):
    """Extract terms enclosed in brackets from the ground truth."""
    return re.findall(r"\[(.*?)\]", text)


def match_terms(asr_text, terms):
    """Find each term from the ground truth in the ASR text."""
    found_terms = []
    for term in terms:
        if re.search(re.escape(term), asr_text, re.IGNORECASE):
            found_terms.append(term)
    return found_terms


def sent_level_term_recall(hypothesis, reference):
    all_deteced_terms = []
    all_groundtruth_terms = []
    true_positives = 0
    false_negatives = 0
    for x, y in zip(hypothesis, reference):
        ground_truth_terms = extract_terms(y)
        all_groundtruth_terms.extend(ground_truth_terms)
        asr_detected_terms = match_terms(x, ground_truth_terms)
        all_deteced_terms.extend(asr_detected_terms)
        true_positive = len(set(ground_truth_terms) & set(asr_detected_terms))
        false_negative = len(set(ground_truth_terms) - set(asr_detected_terms))
        true_positives += true_positive
        false_negatives += false_negative
    if true_positives + false_negatives == 0:
        return 0
    sentence_level_recall = true_positives / (true_positives + false_negatives)
    return sentence_level_recall


def make_ground_truth(
    video_id,
    scene_transcripts,
    dataset_path="./dataset/conference/2/acl_6060",
    split="eval",
    tmp_dir="./tmp",
):
    data = [
        x.strip()
        for x in open(
            f"{dataset_path}/{split}/text/xml/ACL.6060.{split}.en-xx.en.xml"
        ).readlines()
    ]
    terms = [
        x.strip()
        for x in open(
            f"{dataset_path}/{split}/text/tagged_terminology/ACL.6060.{split}.tagged.en-xx.en.txt"
        ).readlines()
    ]
    os.makedirs(tmp_dir, exist_ok=True)
    ground_truth_transcript = collect_doc(
        list(
            map(
                lambda x: x.strip(),
                filter(lambda u: u.startswith("<seg") or u.startswith("<doc"), data),
            )
        )
    )[video_id]

    open(f"./{tmp_dir}/ground_truth.tmp", "w").write("\n".join(ground_truth_transcript))

    aligned_terms = []
    for i, x in enumerate(
        list(
            map(
                lambda x: x.strip(),
                filter(lambda u: u.startswith("<seg") or u.startswith("<doc"), data),
            )
        )
    ):
        if x.startswith("<doc"):
            aligned_terms.append(x)
            continue
        aligned_terms.append('<seg id="0">' + terms.pop(0) + "</seg>")

    ground_truth_transcript_terms = collect_doc(
        list(
            map(
                lambda x: x.strip(),
                filter(
                    lambda u: u.startswith("<seg") or u.startswith("<doc"),
                    aligned_terms,
                ),
            )
        )
    )[video_id]

    open(f"{tmp_dir}/ground_truth.tmp", "w").write("\n".join(ground_truth_transcript))
    open(f"{tmp_dir}/transcript.tmp", "w").write("\n".join(scene_transcripts))

    os.system(
        f"mwerSegmenter -mref {tmp_dir}/transcript.tmp -hypfile {tmp_dir}/ground_truth.tmp -usecase 1"
    )
    scene_groundtruth = open("./__segments").readlines()
    scene_groundtruth = [x.strip() for x in scene_groundtruth]
    scene_groundtruth_terms = reannotate_sentences(
        scene_groundtruth, ground_truth_transcript_terms
    )
    return scene_groundtruth, scene_groundtruth_terms


transformation = jiwer.Compose(
    [
        jiwer.RemovePunctuation(),
        jiwer.ToLowerCase(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
    ]
)


def evaluate(groundtruth, transcripts, transcripts_pe):
    return (
        jiwer.wer(
            groundtruth,
            transcripts,
            truth_transform=transformation,
            hypothesis_transform=transformation,
        ),
        jiwer.wer(
            groundtruth,
            transcripts_pe,
            truth_transform=transformation,
            hypothesis_transform=transformation,
        ),
    )


def highlight_errors(A, B):
    import difflib

    # Function to clean and normalize text
    def normalize(text):
        # Remove punctuation, convert to lower case, and remove hyphens
        return "".join(c.lower() for c in text if c.isalnum() or c.isspace()).split()

    # Clean and normalize both A and B
    A_clean = normalize(A)
    B_clean = normalize(B)

    # Use difflib to get differences between the cleaned sentences
    s = difflib.SequenceMatcher(None, A_clean, B_clean)
    a_result = []  # Results for A
    b_result = []  # Results for B
    a_index = 0  # Track word position in A
    b_index = 0  # Track word position in B

    # Iterate over the matching blocks, looking for differences
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "equal":
            # Append correct words directly to both results
            a_result.extend(A.split()[a_index : a_index + (i2 - i1)])
            b_result.extend(B.split()[b_index : b_index + (j2 - j1)])
            a_index += i2 - i1
            b_index += j2 - j1
        elif tag == "replace":
            # Handle replacements
            a_words = A.split()[a_index : a_index + (i2 - i1)]
            b_words = B.split()[b_index : b_index + (j2 - j1)]
            a_result.append(f"[{' '.join(a_words)}]")
            b_result.append(f"[{' '.join(b_words)}]")
            a_index += i2 - i1
            b_index += j2 - j1
        elif tag == "delete":
            # Handle deletions
            a_words = A.split()[a_index : a_index + (i2 - i1)]
            a_result.append(f"{{{' '.join(a_words)}}}")
            b_result.append("{}")
            a_index += i2 - i1
        elif tag == "insert":
            # Handle insertions
            b_words = B.split()[b_index : b_index + (j2 - j1)]
            a_result.append("<>")
            b_result.append(f"<{' '.join(b_words)}>")
            b_index += j2 - j1

    # Return the marked sentences for A and B
    return " ".join(a_result), " ".join(b_result)


error_annotation_system = """
## Role Introduction
You are a professional ASR transcript error annotator. Your task is to identify and categorize mismatches between the ground-truth text and the ASR-generated transcript.

## Input Data Format
The data you will review is presented in two sections:
- **### Ground-Truth**: This is the original, correct version of the text.
- **### ASR-Transcript**: This is the ASR-generated text that may contain mismatches.

## Mismatch Symbols
In both transcripts and ground-truth, mismatches are highlighted using specific symbols:
- **"<>": Insertion** - Text that the ASR system added erroneously.
- **"[]": Substitution** - Text that the ASR system altered incorrectly.
- **"{}": Omission** - Text that the ASR system failed to include.

## Mismatch Categories
You will categorize each mismatch into one of the following types, based on its content and the context:
- **Terminology**: Specialized terms used in specific domains. These terms have specific meanings and are often used in a particular context. Example: "Neural networks" in a tech discussion, "Quantum entanglement" in physics, "BERT" or "GPT" in an introduction of NLP models.
- **Numerical Data**: Numbers, quantities, or statistical data. Example: "3.14", "thirty percent" or "3 out of 4" when discussing experimental results.
- **Named Entities**: Names of people, places and organizations. These are unique identifiers that do not change across different contexts. Example: "Google" in a business case study, "Albert Einstein" as a person's name.
- **Grammatical Words**: Words that primarily serve a grammatical function in a sentence, including articles, prepositions, conjunctions, and auxiliary verbs. Example: "and" in "cats and dogs", "the" in "the cat".
- **Disfluencies and Fillers**: Words or phrases that reflect speech disfluency, including fillers, repetitions, or corrections. Example: "uh", "um", "actually", or repeated words due to stuttering.
- **General Words**: Common words that are not specialized terms, named entities, grammatical words, disfluencies, or fillers. Example: "increase" in "the company saw a significant increase in profits", "very" in "a very big dog".

Note: The distinction between these categories should always consider the context of the content, not just the meaning of the text within the brackets. Additionally, classification should be based on the corresponding words in the Ground-Truth text, not the erroneous words in the ASR-Transcript.

## Severity Annotation
For each mismatch, assess the severity of the impact on the meaning of the text, categorized into one of three levels:
- **OK**: No significant impact on the meaning. These are typically minor grammatical errors or slight pronunciation issues that do not alter the understanding of the sentence. For example, minor grammatical errors, tense, voice, singular/plural errors, conversions between numerical and text formats (e.g., "0" vs "zero"), missing colloquial words, missing or inserted punctuation, case conversion, and minor rephrasing that does not change the original meaning.
  - **Example**: "present" instead of "presents" due to unclear pronunciation (Ground-Truth: "The speaker [presents] the results."; ASR-Transcript: "The speaker [present] the results.")
- **MINOR**: Slight impact on the meaning, but the main point is still understandable. These errors may cause slight confusion or require minor inferencing to understand the original intent. For example, non-critical homophone errors and spelling mistakes of general words.
  - **Example**: "their" instead of "there" (Ground-Truth: "[There] were several experiments conducted."; ASR-Transcript: "[Their] were several experiments conducted.")
- **CRITICAL**: Significant or complete change or distortion of the original meaning. These errors alter the key message or cause major confusion, potentially leading to misunderstanding of the content. For example, critical homophone errors and serious misspellings or misrecognition of key terms and names, misrecognition of numerical values (e.g. 90 vs 19), and large omissions or rephrasings that deviate from the original meaning.
  - **Example**: "social lightwork" instead of "neural network" (Ground-Truth: "The [neural network] showed high accuracy."; ASR-Transcript: "The [social lightwork] showed high accuracy.")

Note: The determination of severity requires careful consideration of the context.

## Return Format
Return the annotation results as a JSON object with a key named "mismatches", the value of it should be a JSON list. Each object in the list should correspond to a mismatch in the text and include the following information:

```json
{
  "mismatch_type": "insertion | substitution | omission",
  "mismatch": "For insertions: inserted string | For substitutions: incorrect text | For omissions: empty string",
  "correct_form": "For insertions: empty string | For substitutions: correct text | For omissions: omitted text",
  "mismatch_content_type": "One of the defined category names",
  "severity": "OK | MINOR | CRITICAL",
}
```

## Example for Clarity

### Ground-Truth
The conference starts at [9]. Dr. [Smith] will <> give [the] keynote {speech} about [neural networks] such as [BERT] and [Roberta] as well as [their] applications for [forty five] minutes.


### ASR-Transcript
The conference starts at [night]. Dr. [Smath] will <ah> give [a] keynote {} about [social lightworks] such as [birds] and [robots] as well as [these] applications for [45] minutes.


Your returned annotation should look like this:

```json
{
    "mismatches": [
        {"mismatch_type": "substitution", "mismatch": "night", "correct_form": "9", "mismatch_content_type": "Numerical Data", "severity": "CRITICAL"},
        {"mismatch_type": "substitution", "mismatch": "Smath", "correct_form": "Smith", "mismatch_content_type": "Named Entities", "severity": "CRITICAL"},
        {"mismatch_type": "insertion", "mismatch": "ah", "correct_form": "", "mismatch_content_type": "Disfluencies and Fillers", "severity": "OK"},
        {"mismatch_type": "substitution", "mismatch": "a", "correct_form": "the", "mismatch_content_type": "Grammatical Words", "severity": "OK"},
        {"mismatch_type": "omission", "mismatch": "", "correct_form": "speech", "mismatch_content_type": "General Words", "severity": "MINOR"}, 
        {"mismatch_type": "substitution", "mismatch": "social lightworks", "correct_form": "neural networks", "mismatch_content_type": "Terminology", "severity": "CRITICAL"},
        {"mismatch_type": "substitution", "mismatch": "birds", "correct_form": "BERT", "mismatch_content_type": "Terminology", "severity": "CRITICAL"},
        {"mismatch_type": "substitution", "mismatch": "robots", "correct_form": "Roberta", "mismatch_content_type": "Terminology", "severity": "CRITICAL"},
        {"mismatch_type": "substitution", "mismatch": "these", "correct_form": "their", "mismatch_content_type": "General Words", "severity": "MINOR"},
        {"mismatch_type": "substitution", "mismatch": "45", "correct_form": "forty five", "mismatch_content_type": "Numerical Data", "severity": "OK"},
    ]
}
```


Note that only the errors highlighted with Mismatch Symbols should be extracted, make sure the number of objects in the returned JSON list are matched to mismatch symbols in the ASR-Transcript. Please return only the JSON result with the specified details and no additional content.
""".strip()

error_annotation_user = """
### Ground-Truth
{ground_truth}

### ASR-Transcript
{asr_transcript}
""".strip()

highlighting = lambda x, y: dict(
    zip(["ground_truth", "asr_transcript"], highlight_errors(x, y))
)


def parse_json_response(response):
    # Remove triple backticks and "json" if present
    clean_response = re.sub(r"```json|```", "", response.strip())
    try:
        return json.loads(clean_response)
    except json.JSONDecodeError:
        return clean_response


def annotate_results(openai_client, refs, hyps, model="gpt-4o"):
    annotated_result = []
    for ref, hyp in tqdm.tqdm(zip(refs, hyps), total=len(refs)):
        highlighted_input = highlighting(ref, hyp)
        user_input = error_annotation_user.format(**highlighted_input)
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": error_annotation_system},
                {"role": "user", "content": user_input},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        response = response.choices[0].message.content
        maybe_json_object = parse_json_response(response)
        annotated_result.append({**highlighted_input, **maybe_json_object})
    return annotated_result


def annotate_and_compute_swer(
    client, scene_groundtruth, transcripts, output_dir, video_id, annotator="gpt-4o"
):
    
    weight = {
        "CRITICAL": 1,
        "MINOR": 0.6,
        "OK": 0.2
    }
    annotation_record_name = f"{output_dir}/annotation_record_{video_id}.json"
    annotated_result = annotate_results(
        client, scene_groundtruth, transcripts, model=annotator
    )
    with open(annotation_record_name, "w") as f:
        json.dump(annotated_result, f)
    
    df = pd.DataFrame(annotated_result)
    errors = list(itertools.chain.from_iterable(df['mismatches'].tolist()))
    df = pd.DataFrame(errors)
    swer = df['severity'].apply(lambda x:weight[x]).sum()
    type_statistics = df['mismatch_content_type'].value_counts().to_dict()
    return swer, type_statistics


def compute_wer(groundtruth, transcripts, transcripts_pe):
    return (
        jiwer.wer(
            groundtruth,
            transcripts,
            truth_transform=transformation,
            hypothesis_transform=transformation,
        ),
        jiwer.wer(
            groundtruth,
            transcripts_pe,
            truth_transform=transformation,
            hypothesis_transform=transformation,
        ),
    )


def evaluate_video(args, video_id):
    result_dir = args.result_dir
    tmp_dir = args.tmp_dir
    dataset_path = args.dataset_path
    compute_swer = args.compute_swer
    output_dir = args.output_dir
    split = video_to_slit[video_id]
    transcripts_path = f"{result_dir}/{video_id}/{video_id}.scene_transcript.txt"
    transcripts_pe_path = f"{result_dir}/{video_id}/{video_id}.transcript_pe.txt"
    transcripts = [x.strip() for x in open(transcripts_path).readlines()]
    transcripts_pe = [x.strip() for x in open(transcripts_pe_path).readlines()]
    scene_groundtruth, scene_groundtruth_terms = make_ground_truth(
        video_id, transcripts, split, dataset_path=dataset_path, tmp_dir=tmp_dir
    )
    wer_transcript = jiwer.wer(scene_groundtruth, transcripts)
    transcripts_pe = list(filter(lambda x: len(x) > 0, transcripts_pe))
    wer_pe = jiwer.wer(scene_groundtruth, transcripts_pe)
    normalized_wers = compute_wer(scene_groundtruth, transcripts, transcripts_pe)

    evaluate_result = {
        "wer_tarnscript": wer_transcript,
        "wer_pe": wer_pe,
        "normalized_wer_transcript": normalized_wers[0],
        "normalized_wer_pe": normalized_wers[1],
        "term_recall_transcript": sent_level_term_recall(
            transcripts, scene_groundtruth_terms
        ),
        "term_recall_pe": sent_level_term_recall(
            transcripts_pe, scene_groundtruth_terms
        ),
    }

    if compute_swer:
        annotator = args.annotator
        client = OpenAI(args.openai_api_key)
        swer_transcripts, statistics = annotate_and_compute_swer(
            client, scene_groundtruth, transcripts, output_dir, video_id, annotator
        )
        swer_transcripts_pe, statistics_pe = annotate_and_compute_swer(
            client, scene_groundtruth, transcripts_pe, output_dir, video_id, annotator
        )
        evaluate_result.update(
            {
                "swer_transcripts": swer_transcripts,
                "swer_transcripts_pe": swer_transcripts_pe,
            }
        )

        for k,v in statistics.items():
            evaluate_result[f"{k}_transcripts"] = v
        for k,v in statistics_pe.items():
            evaluate_result[f"{k}_transcripts_pe"] = v

    return evaluate_result

current_path = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--tmp_dir", type=str, default="./tmp")
parser.add_argument("--mwerSegmenter_path", type=str, default=f"{current_path}/mwerSegmenter")
parser.add_argument(
    "--dataset_path", type=str, default="../dataset/conference/2/acl_6060"
)
parser.add_argument("--split", type=str, choices=["dev", "eval", "all"], default="eval")
parser.add_argument("--result_dir", required=True, type=str)
parser.add_argument("--output_dir", required=True, type=str)
parser.add_argument("--compute_swer", action="store_true")
parser.add_argument("--annotator", type=str, default="gpt-4o")
parser.add_argument("--openai_api_key", type=str)

args = parser.parse_args()


def evaluate(args):

    result_dir = args.result_dir
    tmp_dir = args.tmp_dir
    split = args.split
    output_dir = args.output_dir
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    if split == "all":
        all_video_ids = [k for k, v in video_to_slit.items()]
    else:
        all_video_ids = [k for k, v in video_to_slit.items() if v == split]

    results = {}
    for video_id in all_video_ids:
        print(result_dir, video_id)
        try:
            result = evaluate_video(args, video_id)
            results[video_id] = result
        except Exception as e:
            print(e)
    results = pd.DataFrame(results).T
    output_name = f"{output_dir}/results.csv"
    results.to_csv(output_name)


if __name__ == "__main__":
    evaluate(args)
