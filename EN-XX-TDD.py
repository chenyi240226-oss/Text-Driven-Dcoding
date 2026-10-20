#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
full_main_contrast_translation_qwen3vl32b_data_en2xx.py

目标：
1. 使用 Qwen3-VL-32B-Instruct 模型。
2. 支持 VG / Multi30K 两套数据。
3. 由命令行输入：
   --data vg 或 --data multi30k
   --lang
   --gpu
   --alpha

4. 翻译方向：
   English -> 指定语言

5. full-main contrastive decoding:
   new_logits = logits_full + CONTRAST_ALPHA * (logits_full - logits_image)

6. 当：
   CONTRAST_ALPHA == 0
   HEAD_BETA == 0

   程序不走手动 logits 解码，而是直接调用：
   model.generate(..., do_sample=False, use_cache=True)

7. 输出路径：
   /home/chenyi/offline_blaser_qe/out/Qwen3_VL_32B/{DATA}/{lang}/en_{lang}_{alpha_tag}_full_main_contrast_translation.jsonl

VG 输入：
   /home/chenyi/offline_blaser_qe/dataSet/VG/{lang}/{lang}_vg_test.jsonl

VG 字段：
   src_en -> 现在作为待翻译源句，参与生成
   tgt    -> 现在作为参考译文，只写入 reference_translation，不参与生成
   image_id -> 用于在 /home/chenyi/offline_blaser_qe/dataSet/VG/images 下找图片

Multi30K 输入：
   /mnt/raid/cycy/data/txt_multi30k/en_{lang}_test.jsonl

Multi30K 字段：
   source -> 现在作为待翻译源句，参与生成
   target -> 现在作为参考译文，只写入 reference_translation，不参与生成
   image / image_path / image_id -> 图片路径解析

运行示例：
   python full_main_contrast_translation_qwen3vl32b_data_en2xx.py --data vg --lang bn --gpu 0 --alpha 0
   python full_main_contrast_translation_qwen3vl32b_data_en2xx.py --data vg --lang bn --gpu 0 --alpha 0.05

   python full_main_contrast_translation_qwen3vl32b_data_en2xx.py --data multi30k --lang hi --gpu 0 --alpha 0
   python full_main_contrast_translation_qwen3vl32b_data_en2xx.py --data multi30k --lang lv --gpu 0 --alpha 0.05
"""

import os
import argparse

# =====================================================
# 0. 命令行参数：必须在 import torch 前解析并设置 GPU
# =====================================================

VG_LANGUAGES = {"hi", "ha", "bn", "ma"}
MULTI30K_LANGUAGES = {"hi", "lv", "tr"}
ALL_LANGUAGES = VG_LANGUAGES | MULTI30K_LANGUAGES

TARGET_LANGUAGE_NAME_MAP = {
    "hi": "Hindi",
    "ha": "Hausa",
    "bn": "Bengali",
    "ma": "Malayalam",
    "lv": "Latvian",
    "tr": "Turkish",
}

DATA_DIR_NAME = {
    "vg": "VG",
    "multi30k": "Multi30k",
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=["vg", "multi30k"],
        help="Dataset name: vg or multi30k.",
    )

    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=sorted(ALL_LANGUAGES),
        help="Language code. VG: bn, ha, hi, ma. Multi30K: hi, lv, tr.",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        required=True,
        help="Physical GPU id, for example: 0, 1, 2, 5, 6, 7.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="CONTRAST_ALPHA. Example: 0, 0.05, 0.1.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum generated target-language tokens.",
    )

    return parser.parse_args()


ARGS = parse_args()

DATASET = ARGS.data.strip().lower()
LANGUAGE = ARGS.lang.strip().lower()
GPU = str(ARGS.gpu).strip()
CONTRAST_ALPHA = float(ARGS.alpha)
MAX_NEW_TOKENS = int(ARGS.max_new_tokens)

if DATASET == "vg" and LANGUAGE not in VG_LANGUAGES:
    raise ValueError(
        f"--data vg only supports languages: {sorted(VG_LANGUAGES)}, "
        f"but got --lang {LANGUAGE}"
    )

if DATASET == "multi30k" and LANGUAGE not in MULTI30K_LANGUAGES:
    raise ValueError(
        f"--data multi30k only supports languages: {sorted(MULTI30K_LANGUAGES)}, "
        f"but got --lang {LANGUAGE}"
    )

DATA_OUTPUT_NAME = DATA_DIR_NAME[DATASET]
TARGET_LANG_NAME = TARGET_LANGUAGE_NAME_MAP[LANGUAGE]
SOURCE_LANG_NAME = "English"

# ====== 环境变量必须在 import torch 前设置 ======
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["HF_HOME"] = "/home/chenyi/model_cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/home/chenyi/model_cache/huggingface/hub"

SYSLIBS_DIR = "/home/chenyi/offline_blaser_qe/syslibs"
os.environ["LD_LIBRARY_PATH"] = SYSLIBS_DIR + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import re
import json
import time
import warnings
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# =====================================================
# Torch compatibility patch
# =====================================================

if not hasattr(torch, "compiler"):
    class _DummyCompiler:
        @staticmethod
        def is_compiling():
            return False

    torch.compiler = _DummyCompiler()
else:
    if not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False


warnings.filterwarnings("ignore", category=UserWarning)
torch.set_grad_enabled(False)


# =====================================================
# 1. 固定配置
# =====================================================

MODEL_PATH = "/mnt/raid/wjx/vision-rag-mt/Qwen3-VL-32B-Instruct"

VG_INPUT_ROOT = "/home/chenyi/offline_blaser_qe/dataSet/VG"
VG_IMAGE_DIR = "/home/chenyi/offline_blaser_qe/dataSet/VG/images"

MULTI30K_INPUT_ROOT = "/mnt/raid/cycy/data/txt_multi30k"
MULTI30K_IMAGE_DIR = "/mnt/raid/wjx/vision-rag-mt/images_multi30k"

OUTPUT_ROOT_BASE = "/home/chenyi/offline_blaser_qe/out/Qwen3_VL_32B"

DTYPE_NAME = "bfloat16"
ATTN_IMPLEMENTATION = "sdpa"

# None 表示跑全量；100 表示只跑前100条
MAX_SAMPLES = None

# =====================================================
# 字段配置
# =====================================================

IMAGE_ID_FIELD = "image_id"

# VG 原始字段：现在调换方向
VG_SOURCE_FIELD = "src_en"
VG_REFERENCE_FIELD = "tgt"

# Multi30K 字段：现在调换方向
MULTI30K_SOURCE_FIELDS = ["source", "reference_translation"]
MULTI30K_REFERENCE_FIELDS = ["target", "source_text"]

# 输出给评分程序使用的统一字段
STANDARD_SOURCE_FIELD = "source_text"
STANDARD_REFERENCE_FIELD = "reference_translation"

# 为了兼容后续评分程序，生成字段名保持不变
OUTPUT_FIELD = "full_main_contrast_translation_en"
COMPAT_OUTPUT_FIELD = "text_main_contrast_translation_en"

# =====================================================
# full-main contrastive decoding 参数
# =====================================================

HEAD_BETA = 0
MIN_NEW_TOKENS = 0
SEED = 42


# =====================================================
# 2. Prompt
# =====================================================

SYSTEM_PROMPT_TEMPLATE = (
    "/no_think\n"
    "You are a professional translator.\n"
    "Your job is to translate the given {source_lang_name} source sentence into {target_lang_name}.\n"
    "STRICT RULES:\n"
    "1) Faithful: Do NOT add, remove, or change objects, attributes, actions, or relations.\n"
    "2) Use the image only when it helps resolve ambiguity.\n"
    "3) Do not turn the task into image captioning.\n"
    "4) Output ONLY the translation in {target_lang_name}. No explanations.\n"
    "5) NEVER output any <think>...</think> content.\n"
)


def build_initial_style_system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        source_lang_name=SOURCE_LANG_NAME,
        target_lang_name=TARGET_LANG_NAME,
    )


def build_initial_style_user_text(src: str) -> str:
    return (
        f"{SOURCE_LANG_NAME} SOURCE:\n"
        f"{src}\n\n"
        f"Translate into {TARGET_LANG_NAME}."
    )


def build_messages_initial_style_full(image_path: str, src: str):
    system_prompt = build_initial_style_system_prompt()
    user_text = build_initial_style_user_text(src)

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text}
            ],
        }
    ]

    return messages


def build_messages_initial_style_text_only(src: str):
    system_prompt = build_initial_style_system_prompt()
    user_text = build_initial_style_user_text(src)

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text}
            ],
        }
    ]

    return messages


def build_messages_initial_style_image_only(image_path: str):
    system_prompt = build_initial_style_system_prompt()
    user_text = build_initial_style_user_text("[SOURCE SENTENCE HIDDEN]")

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text}
            ],
        }
    ]

    return messages


# =====================================================
# 3. 路径与基础工具
# =====================================================

def alpha_to_filename_tag(alpha: float) -> str:
    return f"{float(alpha):g}"


def contrast_formula_str() -> str:
    return f"new_logits = logits_full + {CONTRAST_ALPHA} * (logits_full - logits_image)"


def use_generate_baseline() -> bool:
    return abs(float(CONTRAST_ALPHA)) < 1e-12 and abs(float(HEAD_BETA)) < 1e-12


def get_image_dir() -> str:
    if DATASET == "vg":
        return VG_IMAGE_DIR
    if DATASET == "multi30k":
        return MULTI30K_IMAGE_DIR
    raise ValueError(f"Unsupported DATASET: {DATASET}")


def get_input_root() -> str:
    if DATASET == "vg":
        return VG_INPUT_ROOT
    if DATASET == "multi30k":
        return MULTI30K_INPUT_ROOT
    raise ValueError(f"Unsupported DATASET: {DATASET}")


def get_paths(data: str, lang: str) -> Tuple[str, str]:
    if data == "vg":
        input_jsonl = os.path.join(
            VG_INPUT_ROOT,
            lang,
            f"{lang}_vg_test.jsonl",
        )
    elif data == "multi30k":
        input_jsonl = os.path.join(
            MULTI30K_INPUT_ROOT,
            f"en_{lang}_test.jsonl",
        )
    else:
        raise ValueError(f"Unsupported data: {data}")

    output_dir = os.path.join(
        OUTPUT_ROOT_BASE,
        DATA_OUTPUT_NAME,
        lang,
    )

    alpha_tag = alpha_to_filename_tag(CONTRAST_ALPHA)

    output_jsonl = os.path.join(
        output_dir,
        f"en_{lang}_{alpha_tag}_full_main_contrast_translation.jsonl",
    )

    return input_jsonl, output_jsonl


def get_torch_dtype(name: str):
    if name == "auto":
        return "auto"

    if not torch.cuda.is_available():
        return torch.float32

    if name == "float16":
        return torch.float16

    if name == "bfloat16":
        return torch.bfloat16

    return torch.float32


def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def read_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] JSON parse failed at line {line_idx}: {e}", flush=True)
                continue

            if max_samples is not None and len(rows) >= max_samples:
                break

    return rows


def init_jsonl(path: str):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8"):
        pass


def append_jsonl(path: str, row: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def first_non_empty_with_field(row: Dict[str, Any], fields: List[str]) -> Tuple[str, str]:
    for field in fields:
        value = normalize_text(row.get(field, ""))
        if value:
            return value, field
    return "", ""


def get_row_id(row: Dict[str, Any], fallback_idx: int) -> str:
    if DATASET == "vg":
        if IMAGE_ID_FIELD in row and row[IMAGE_ID_FIELD] is not None:
            return str(row[IMAGE_ID_FIELD])

        for k in ["id", "qid", "question_id", "sample_index"]:
            if k in row and row[k] is not None:
                return str(row[k])

        return str(fallback_idx)

    if DATASET == "multi30k":
        for k in ["id", "image_id", "qid", "question_id"]:
            if k in row and row[k] is not None:
                return str(row[k])

        image_path = normalize_text(row.get("image", row.get("image_path", "")))
        if image_path:
            base = os.path.basename(image_path)
            stem = os.path.splitext(base)[0]
            if stem:
                return stem

        return str(fallback_idx)

    return str(fallback_idx)


def format_seconds(sec: float) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def cleanup_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


# =====================================================
# 4. 输出清洗
# =====================================================

def is_effectively_empty_output(text: str) -> bool:
    if text is None:
        return True

    text = str(text).strip()
    if not text:
        return True

    if re.fullmatch(r"[\s\.\,\!\?\:\;\"\'\-\(\)\[\]\{\}]+", text):
        return True

    lower = text.lower()
    bad_patterns = [
        "i can't",
        "i cannot",
        "cannot determine",
        "cannot translate",
        "unable to",
        "no translation",
        "no output",
        "sorry",
    ]

    if any(p in lower for p in bad_patterns):
        return True

    return False


def clean_generation_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^\s*(translation\s*[:：-]\s*)", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_think_blocks(text: str) -> str:
    if not text:
        return text

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)

    return text.strip()


def clean_translation_output(text: str) -> str:
    text = strip_think_blocks(text)
    text = clean_generation_text(text)

    bad_prefixes = [
        "Translation:",
        "The translation is:",
        "Output:",
        "Answer:",
        "Final translation:",
        "Final answer:",
    ]

    for p in bad_prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines:
        text = lines[0]

    text = text.strip().strip('"').strip("'").strip()
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================================
# 5. 数据字段与图片路径解析
# =====================================================

def get_source_and_reference(row: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    返回：
      source_text, reference_translation, source_field_name, reference_field_name

    当前方向：
      English -> target language
    """

    if DATASET == "vg":
        source_text = normalize_text(row.get(VG_SOURCE_FIELD, ""))
        reference_translation = normalize_text(row.get(VG_REFERENCE_FIELD, ""))
        return source_text, reference_translation, VG_SOURCE_FIELD, VG_REFERENCE_FIELD

    if DATASET == "multi30k":
        source_text, source_field = first_non_empty_with_field(
            row,
            MULTI30K_SOURCE_FIELDS,
        )
        reference_translation, reference_field = first_non_empty_with_field(
            row,
            MULTI30K_REFERENCE_FIELDS,
        )
        return source_text, reference_translation, source_field, reference_field

    raise ValueError(f"Unsupported DATASET: {DATASET}")


def resolve_image_path(row: Dict[str, Any]) -> Optional[str]:
    if DATASET == "vg":
        image_id = row.get(IMAGE_ID_FIELD, None)

        if image_id is None:
            return None

        image_id = str(image_id).strip()

        if not image_id:
            return None

        for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"]:
            p = os.path.join(VG_IMAGE_DIR, image_id + ext)
            if os.path.isfile(p):
                return p

        return None

    if DATASET == "multi30k":
        image_path = normalize_text(row.get("image", ""))

        if image_path and os.path.isfile(image_path):
            return image_path

        for key in ["image_path", "img_path"]:
            p = normalize_text(row.get(key, ""))
            if p and os.path.isfile(p):
                return p

        image_id = row.get("image_id", row.get("id", None))

        if image_id is None:
            return None

        image_id = str(image_id).strip()

        if not image_id:
            return None

        for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"]:
            p = os.path.join(MULTI30K_IMAGE_DIR, image_id + ext)
            if os.path.isfile(p):
                return p

        return None

    raise ValueError(f"Unsupported DATASET: {DATASET}")


# =====================================================
# 6. Qwen3-VL-32B 输入
# =====================================================

def get_tokenizer(processor):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("processor.tokenizer is None")
    return tokenizer


def get_input_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_inputs_to_device(
    inputs: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    moved = {}

    for k, v in inputs.items():
        if hasattr(v, "to"):
            moved[k] = v.to(device)
        else:
            moved[k] = v

    return moved


def prepare_inputs_from_messages(
    processor,
    messages,
) -> Dict[str, Any]:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    return move_inputs_to_device(
        inputs=inputs,
        device=get_input_device(),
    )


def prepare_three_condition_inputs(
    processor,
    image_path: str,
    source_text: str,
) -> Dict[str, Dict[str, Any]]:
    messages_text = build_messages_initial_style_text_only(source_text)
    messages_full = build_messages_initial_style_full(image_path, source_text)
    messages_image = build_messages_initial_style_image_only(image_path)

    return {
        "text": prepare_inputs_from_messages(
            processor=processor,
            messages=messages_text,
        ),
        "full": prepare_inputs_from_messages(
            processor=processor,
            messages=messages_full,
        ),
        "image": prepare_inputs_from_messages(
            processor=processor,
            messages=messages_image,
        ),
    }


# =====================================================
# 7. EOS / PAD
# =====================================================

def get_pad_eos_ids(processor) -> Tuple[Optional[int], List[int]]:
    tokenizer = get_tokenizer(processor)

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    if pad_token_id is None:
        pad_token_id = eos_token_id

    eos_ids = []
    if eos_token_id is not None:
        eos_ids.append(int(eos_token_id))

    eos_ids = sorted(set(eos_ids))

    return pad_token_id, eos_ids


# =====================================================
# 8. generate baseline
# =====================================================

def generate_full_direct_with_cache(
    model,
    processor,
    image_path: str,
    source_text: str,
    max_new_tokens: int,
    image_id: Any = "",
) -> Tuple[str, Dict[str, Any]]:

    messages = build_messages_initial_style_full(image_path, source_text)

    prepared_inputs = prepare_inputs_from_messages(
        processor=processor,
        messages=messages,
    )

    tokenizer = get_tokenizer(processor)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    if pad_token_id is None:
        pad_token_id = eos_token_id

    with torch.no_grad():
        out = model.generate(
            **prepared_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    input_ids = prepared_inputs["input_ids"]
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(input_ids, out)
    ]

    result = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    result = strip_think_blocks(result)
    result = clean_generation_text(result)

    info = {
        "decoding": "generate_full_direct_baseline_use_cache",
        "use_cache": True,
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_token_id,
        "eos_token_id": eos_token_id,
        "contrast_alpha": CONTRAST_ALPHA,
        "head_beta": HEAD_BETA,
        "reason": "CONTRAST_ALPHA == 0 and HEAD_BETA == 0",
        "baseline_aligned_to_initial_translation_program": True,
    }

    return result, info


# =====================================================
# 9. 手动 contrastive decoding
# =====================================================

def build_step_inputs(
    base_inputs: Dict[str, Any],
    generated_ids: List[int],
) -> Dict[str, Any]:
    base_input_ids = base_inputs["input_ids"]
    device = base_input_ids.device

    merged = {}

    for k, v in base_inputs.items():
        if k in ["input_ids", "attention_mask"]:
            continue
        merged[k] = v

    if generated_ids:
        gen_tensor = torch.tensor(
            [generated_ids],
            dtype=base_input_ids.dtype,
            device=device,
        )
        input_ids = torch.cat([base_input_ids, gen_tensor], dim=1)
    else:
        input_ids = base_input_ids

    if "attention_mask" in base_inputs:
        base_mask = base_inputs["attention_mask"]

        if generated_ids:
            gen_mask = torch.ones(
                (base_mask.shape[0], len(generated_ids)),
                dtype=base_mask.dtype,
                device=base_mask.device,
            )
            attention_mask = torch.cat([base_mask, gen_mask], dim=1)
        else:
            attention_mask = base_mask
    else:
        attention_mask = torch.ones_like(input_ids)

    merged["input_ids"] = input_ids
    merged["attention_mask"] = attention_mask

    return merged


def get_next_logits_no_cache(
    model,
    base_inputs: Dict[str, Any],
    generated_ids: List[int],
) -> torch.Tensor:
    step_inputs = build_step_inputs(
        base_inputs=base_inputs,
        generated_ids=generated_ids,
    )

    with torch.inference_mode():
        outputs = model(
            **step_inputs,
            use_cache=False,
        )

    logits = outputs.logits[:, -1, :].float()
    return logits


def sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    logits = torch.nan_to_num(
        logits,
        nan=-1e4,
        posinf=1e4,
        neginf=-1e4,
    )
    return logits


def apply_text_head_constraint(
    new_logits: torch.Tensor,
    logits_text: torch.Tensor,
    head_beta: float,
    eos_ids: List[int],
    step_idx: int,
) -> torch.Tensor:
    if head_beta is None or head_beta <= 0:
        return new_logits

    probs_text = F.softmax(logits_text.float(), dim=-1)
    max_prob = torch.max(probs_text, dim=-1, keepdim=True).values

    head_mask = probs_text >= (float(head_beta) * max_prob)

    if step_idx >= MIN_NEW_TOKENS:
        for eid in eos_ids:
            if 0 <= eid < head_mask.shape[-1]:
                head_mask[:, eid] = True

    if not bool(head_mask.any().item()):
        top_id = torch.argmax(logits_text, dim=-1, keepdim=True)
        head_mask.scatter_(dim=-1, index=top_id, value=True)

    filtered_logits = new_logits.masked_fill(~head_mask, -1e9)

    return filtered_logits


def contrastive_next_logits(
    model,
    condition_inputs: Dict[str, Dict[str, Any]],
    generated_ids: List[int],
    eos_ids: List[int],
    step_idx: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    logits_text = get_next_logits_no_cache(
        model=model,
        base_inputs=condition_inputs["text"],
        generated_ids=generated_ids,
    )

    logits_full = get_next_logits_no_cache(
        model=model,
        base_inputs=condition_inputs["full"],
        generated_ids=generated_ids,
    )

    logits_image = get_next_logits_no_cache(
        model=model,
        base_inputs=condition_inputs["image"],
        generated_ids=generated_ids,
    )

    logits_text = sanitize_logits(logits_text)
    logits_full = sanitize_logits(logits_full)
    logits_image = sanitize_logits(logits_image)

    new_logits = logits_full + CONTRAST_ALPHA * (logits_full - logits_image)
    new_logits = sanitize_logits(new_logits)

    new_logits = apply_text_head_constraint(
        new_logits=new_logits,
        logits_text=logits_text,
        head_beta=HEAD_BETA,
        eos_ids=eos_ids,
        step_idx=step_idx,
    )

    diag = {
        "contrast_alpha": CONTRAST_ALPHA,
        "head_beta": HEAD_BETA,
    }

    return new_logits, diag


def decode_generated_ids(
    processor,
    generated_ids: List[int],
) -> str:
    tokenizer = get_tokenizer(processor)

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return clean_translation_output(text)


def generate_full_main_contrast_translation_manual(
    model,
    processor,
    image_path: str,
    source_text: str,
    max_new_tokens: int,
) -> Tuple[str, Dict[str, Any]]:

    _, eos_ids = get_pad_eos_ids(processor)

    condition_inputs = prepare_three_condition_inputs(
        processor=processor,
        image_path=image_path,
        source_text=source_text,
    )

    generated_ids: List[int] = []
    step_diag: List[Dict[str, Any]] = []

    for step_idx in range(max_new_tokens):
        new_logits, diag = contrastive_next_logits(
            model=model,
            condition_inputs=condition_inputs,
            generated_ids=generated_ids,
            eos_ids=eos_ids,
            step_idx=step_idx,
        )

        next_id = int(torch.argmax(new_logits, dim=-1).item())

        diag["step"] = step_idx
        diag["next_id"] = next_id
        diag["is_eos"] = next_id in eos_ids
        step_diag.append(diag)

        if next_id in eos_ids and step_idx >= MIN_NEW_TOKENS:
            break

        generated_ids.append(next_id)

    text = decode_generated_ids(
        processor=processor,
        generated_ids=generated_ids,
    )

    info = {
        "decoding": "manual_greedy_full_main_contrastive_full_minus_image",
        "generated_token_count": len(generated_ids),
        "stopped_by_eos": bool(step_diag[-1]["is_eos"]) if step_diag else False,
        "max_new_tokens": max_new_tokens,
        "contrast_formula": contrast_formula_str(),
        "contrast_alpha": CONTRAST_ALPHA,
        "head_beta": HEAD_BETA,
        "head_constraint": "disabled" if HEAD_BETA <= 0 else "p_text(token) >= HEAD_BETA * max_token p_text(token)",
        "use_cache": False,
    }

    return text, info


def generate_full_main_contrast_translation(
    model,
    processor,
    image_path: str,
    source_text: str,
    max_new_tokens: int,
    image_id: Any = "",
) -> Tuple[str, Dict[str, Any]]:

    if use_generate_baseline():
        return generate_full_direct_with_cache(
            model=model,
            processor=processor,
            image_path=image_path,
            source_text=source_text,
            max_new_tokens=max_new_tokens,
            image_id=image_id,
        )

    return generate_full_main_contrast_translation_manual(
        model=model,
        processor=processor,
        image_path=image_path,
        source_text=source_text,
        max_new_tokens=max_new_tokens,
    )


# =====================================================
# 10. 单样本处理
# =====================================================

def process_one_row(
    model,
    processor,
    row: Dict[str, Any],
    idx: int,
    max_new_tokens: int,
) -> Dict[str, Any]:

    out = dict(row)

    row_id = get_row_id(row, fallback_idx=idx)

    source_text, reference_translation, source_field, reference_field = get_source_and_reference(row)

    if not source_text:
        raise ValueError(f"missing source field for data={DATASET}")

    image_path = resolve_image_path(row)

    if not image_path:
        raise FileNotFoundError(
            f"cannot resolve image path for data={DATASET}, id={row_id}"
        )

    translation, generation_info = generate_full_main_contrast_translation(
        model=model,
        processor=processor,
        image_path=image_path,
        source_text=source_text,
        max_new_tokens=max_new_tokens,
        image_id=row_id,
    )

    translation = clean_translation_output(translation)

    if is_effectively_empty_output(translation):
        raise RuntimeError("empty generation output")

    # 当前方向是 English -> 指定语言，输出可能是非英文脚本，因此不做 mostly-English 检查。

    out["id"] = row_id
    out[STANDARD_SOURCE_FIELD] = source_text
    out[STANDARD_REFERENCE_FIELD] = reference_translation

    out[OUTPUT_FIELD] = translation
    out[COMPAT_OUTPUT_FIELD] = translation

    out["image_path_used"] = image_path

    out["generation_config"] = {
        "dataset": DATASET,
        "dataset_output_name": DATA_OUTPUT_NAME,
        "translation_direction": f"English_to_{LANGUAGE}",
        "source_language_name": SOURCE_LANG_NAME,
        "target_language_name": TARGET_LANG_NAME,

        "model_path": MODEL_PATH,
        "dtype": DTYPE_NAME,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "formula": contrast_formula_str(),
        "contrast_alpha": CONTRAST_ALPHA,
        "head_beta": HEAD_BETA,
        "min_new_tokens": MIN_NEW_TOKENS,
        "max_new_tokens": max_new_tokens,
        "max_samples": MAX_SAMPLES,

        "standard_source_field": STANDARD_SOURCE_FIELD,
        "standard_reference_field": STANDARD_REFERENCE_FIELD,

        "source_text_value_from": source_field,
        "reference_translation_value_from": reference_field,
        "reference_used_in_generation": False,

        "image_dir": get_image_dir(),
        "image_path_used": image_path,

        "full_condition": f"{DATA_OUTPUT_NAME}: English source visible, image visible",
        "text_condition": f"{DATA_OUTPUT_NAME}: English source visible, no image",
        "image_condition": f"{DATA_OUTPUT_NAME}: source hidden, image visible",
        "system_prompt_template": SYSTEM_PROMPT_TEMPLATE,
        "full_user_prompt_template": f"{SOURCE_LANG_NAME} SOURCE:\\n{{source_text}}\\n\\nTranslate into {TARGET_LANG_NAME}.",
        "generate_baseline_when_alpha0_beta0": True,
        "generation_info": generation_info,
    }

    if DATASET == "vg":
        out["generation_config"].update({
            "vg_image_id_field": IMAGE_ID_FIELD,
            "vg_source_field_used_for_generation": VG_SOURCE_FIELD,
            "vg_reference_field_preserved_only": VG_REFERENCE_FIELD,
            "src_en_participates_in_generation": True,
            "tgt_participates_in_generation": False,
            "src_en_saved_as_source_text": True,
            "tgt_saved_as_reference_translation": True,
        })

    if DATASET == "multi30k":
        out["generation_config"].update({
            "multi30k_source_fields": MULTI30K_SOURCE_FIELDS,
            "multi30k_reference_fields": MULTI30K_REFERENCE_FIELDS,
            "source_participates_in_generation": source_field == "source",
            "target_participates_in_generation": False,
            "source_saved_as_source_text": source_field == "source",
            "target_saved_as_reference_translation": reference_field == "target",
        })

    return out


# =====================================================
# 11. 模型加载：Qwen3-VL-32B
# =====================================================

def load_model_and_processor(dtype: torch.dtype):
    print("[Info] Loading Qwen3-VL-32B model ...")
    print(f"[Info] MODEL_PATH: {MODEL_PATH}")
    print(f"[Info] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[Info] torch.cuda.is_available(): {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"[Info] Visible GPU count: {torch.cuda.device_count()}")
        print(f"[Info] Current device: {torch.cuda.current_device()}")

    model_kwargs = {
        "pretrained_model_name_or_path": MODEL_PATH,
        "device_map": "auto",
        "dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": True,
    }

    if ATTN_IMPLEMENTATION:
        model_kwargs["attn_implementation"] = ATTN_IMPLEMENTATION

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        **model_kwargs
    )

    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    model.eval()

    print("[Info] Model loaded.")
    return model, processor


# =====================================================
# 12. 主程序
# =====================================================

def main():
    torch.manual_seed(SEED)

    input_jsonl, output_jsonl = get_paths(DATASET, LANGUAGE)
    ensure_dir(output_jsonl)

    dtype = get_torch_dtype(DTYPE_NAME)

    print("========== Runtime Config ==========")
    print(f"[INFO] DATASET: {DATASET}")
    print(f"[INFO] DATA_OUTPUT_NAME: {DATA_OUTPUT_NAME}")
    print(f"[INFO] LANGUAGE: {LANGUAGE}")
    print(f"[INFO] SOURCE_LANG_NAME: {SOURCE_LANG_NAME}")
    print(f"[INFO] TARGET_LANG_NAME: {TARGET_LANG_NAME}")
    print(f"[INFO] TRANSLATION_DIRECTION: English -> {TARGET_LANG_NAME}")
    print(f"[INFO] GPU: {GPU}")
    print(f"[INFO] CONTRAST_ALPHA: {CONTRAST_ALPHA}")
    print(f"[INFO] ALPHA_FILENAME_TAG: {alpha_to_filename_tag(CONTRAST_ALPHA)}")
    print(f"[INFO] MAX_SAMPLES: {MAX_SAMPLES}")
    print(f"[INFO] MAX_NEW_TOKENS: {MAX_NEW_TOKENS}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"[INFO] torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"[INFO] torch.cuda.device_count: {torch.cuda.device_count()}")
    print(f"[INFO] dtype: {DTYPE_NAME}")
    print(f"[INFO] attn_implementation: {ATTN_IMPLEMENTATION}")
    print(f"[INFO] MODEL_PATH: {MODEL_PATH}")
    print(f"[INFO] INPUT_JSONL: {input_jsonl}")
    print(f"[INFO] OUTPUT_JSONL: {output_jsonl}")
    print(f"[INFO] IMAGE_DIR: {get_image_dir()}")

    if DATASET == "vg":
        print(f"[INFO] VG_INPUT_ROOT: {VG_INPUT_ROOT}")
        print(f"[INFO] IMAGE_ID_FIELD: {IMAGE_ID_FIELD}")
        print(f"[INFO] VG_SOURCE_FIELD: {VG_SOURCE_FIELD}")
        print(f"[INFO] VG_REFERENCE_FIELD: {VG_REFERENCE_FIELD}")
        print(f"[INFO] src_en participates in generation: True")
        print(f"[INFO] tgt participates in generation: False")
        print(f"[INFO] tgt only saved as reference_translation: True")

    if DATASET == "multi30k":
        print(f"[INFO] MULTI30K_INPUT_ROOT: {MULTI30K_INPUT_ROOT}")
        print(f"[INFO] MULTI30K_SOURCE_FIELDS: {MULTI30K_SOURCE_FIELDS}")
        print(f"[INFO] MULTI30K_REFERENCE_FIELDS: {MULTI30K_REFERENCE_FIELDS}")
        print(f"[INFO] source/reference_translation participates in generation: True")
        print(f"[INFO] target/source_text participates in generation: False")
        print(f"[INFO] target saved as reference_translation: True")

    print(f"[INFO] STANDARD_SOURCE_FIELD: {STANDARD_SOURCE_FIELD}")
    print(f"[INFO] STANDARD_REFERENCE_FIELD: {STANDARD_REFERENCE_FIELD}")
    print(f"[INFO] OUTPUT_FIELD: {OUTPUT_FIELD}")
    print(f"[INFO] COMPAT_OUTPUT_FIELD: {COMPAT_OUTPUT_FIELD}")

    print(f"[INFO] HEAD_BETA: {HEAD_BETA}")
    print(f"[INFO] MIN_NEW_TOKENS: {MIN_NEW_TOKENS}")
    print(f"[INFO] formula: {contrast_formula_str()}")
    print(f"[INFO] use_generate_baseline: {use_generate_baseline()}")
    print("====================================")

    if not os.path.exists(input_jsonl):
        raise FileNotFoundError(f"输入文件不存在: {input_jsonl}")

    rows = read_jsonl(
        path=input_jsonl,
        max_samples=MAX_SAMPLES,
    )

    total = len(rows)

    print(f"[INFO] read rows: {total}", flush=True)

    model, processor = load_model_and_processor(dtype=dtype)

    init_jsonl(output_jsonl)

    stats = {
        "dataset": DATASET,
        "language": LANGUAGE,
        "translation_direction": f"English_to_{LANGUAGE}",
        "total": total,
        "saved": 0,
        "error": 0,
        "empty_or_invalid": 0,
        "generate_baseline": 0,
        "manual_contrast": 0,

        "missing_source": 0,
        "missing_reference": 0,
        "missing_image": 0,

        "missing_src_en": 0,
        "missing_tgt": 0,

        "missing_source_or_reference_translation": 0,
        "missing_target_or_source_text": 0,
    }

    start_time = time.time()

    for idx, row in enumerate(rows, start=1):
        iter_start = time.time()
        row_id = get_row_id(row, fallback_idx=idx)

        try:
            out = process_one_row(
                model=model,
                processor=processor,
                row=row,
                idx=idx,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            gen_info = (
                out.get("generation_config", {})
                   .get("generation_info", {})
            )

            if gen_info.get("decoding") == "generate_full_direct_baseline_use_cache":
                stats["generate_baseline"] += 1
            else:
                stats["manual_contrast"] += 1

            if not normalize_text(out.get(STANDARD_REFERENCE_FIELD, "")):
                stats["missing_reference"] += 1

                if DATASET == "vg":
                    stats["missing_tgt"] += 1

                if DATASET == "multi30k":
                    stats["missing_target_or_source_text"] += 1

        except Exception as e:
            stats["error"] += 1

            err_text = repr(e)

            source_text, reference_translation, source_field, reference_field = get_source_and_reference(row)

            if not source_text:
                stats["missing_source"] += 1

                if DATASET == "vg":
                    stats["missing_src_en"] += 1

                if DATASET == "multi30k":
                    stats["missing_source_or_reference_translation"] += 1

            if not reference_translation:
                stats["missing_reference"] += 1

                if DATASET == "vg":
                    stats["missing_tgt"] += 1

                if DATASET == "multi30k":
                    stats["missing_target_or_source_text"] += 1

            if "cannot resolve image path" in err_text:
                stats["missing_image"] += 1

            out = dict(row)

            out["id"] = row_id
            out[STANDARD_SOURCE_FIELD] = source_text
            out[STANDARD_REFERENCE_FIELD] = reference_translation

            out[OUTPUT_FIELD] = ""
            out[COMPAT_OUTPUT_FIELD] = ""
            out["error"] = err_text
            out["generation_config"] = {
                "dataset": DATASET,
                "dataset_output_name": DATA_OUTPUT_NAME,
                "translation_direction": f"English_to_{LANGUAGE}",
                "source_language_name": SOURCE_LANG_NAME,
                "target_language_name": TARGET_LANG_NAME,
                "model_path": MODEL_PATH,
                "dtype": DTYPE_NAME,
                "attn_implementation": ATTN_IMPLEMENTATION,
                "formula": contrast_formula_str(),
                "contrast_alpha": CONTRAST_ALPHA,
                "head_beta": HEAD_BETA,
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_samples": MAX_SAMPLES,
                "source_text_value_from": source_field,
                "reference_translation_value_from": reference_field,
                "reference_used_in_generation": False,
                "failed": True,
            }

            print(f"[WARN] id={row_id} failed: {err_text}", flush=True)

        append_jsonl(output_jsonl, out)
        stats["saved"] += 1

        elapsed = time.time() - start_time
        avg = elapsed / max(1, idx)
        eta = avg * max(0, total - idx)

        finish_time = time.strftime(
            "%Y-%m-%d %H:%M:%S",
            time.localtime(time.time() + eta),
        )

        iter_time = time.time() - iter_start

        hyp = normalize_text(out.get(OUTPUT_FIELD, ""))
        if not hyp:
            stats["empty_or_invalid"] += 1

        print(
            f"[{idx}/{total}] data={DATASET} | lang={LANGUAGE} | id={row_id} | "
            f"iter={iter_time:.2f}s | ETA={format_seconds(eta)} | finish={finish_time} | "
            f"saved={stats['saved']} | error={stats['error']} | "
            f"hyp={hyp[:80]}",
            flush=True,
        )

        cleanup_cuda()

    print("[done] " + json.dumps(stats, ensure_ascii=False), flush=True)
    print(f"[INFO] output saved to: {output_jsonl}", flush=True)


if __name__ == "__main__":
    main()
