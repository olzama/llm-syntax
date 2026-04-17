import logging
import os
from typing import Dict, Optional

import torch
from config_loader import ChatCfg, DecodingCfg, ModelCfg, QuantizationCfg
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from utils import estimate_cost

load_dotenv("src/config/article_searcher.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextGenerator:
    def __init__(
        self,
        model_cfg: ModelCfg,
        decoding_cfg: DecodingCfg,
        quant_cfg: QuantizationCfg,
        chat_cfg: ChatCfg,
    ):

        self.model_cfg = model_cfg
        self.decoding_cfg = decoding_cfg
        self.quant_cfg = quant_cfg
        self.chat_cfg = chat_cfg

        logger.info(self.model_cfg)
        logger.info(self.decoding_cfg)
        logger.info(self.quant_cfg)
        logger.info(self.chat_cfg)

        if self.model_cfg.name.startswith("gpt"):
            logger.info(f"Using OpenAI API model: {self.model_cfg.name}")
            return

        mode = quant_cfg.mode.lower()
        cfg = AutoConfig.from_pretrained(self.model_cfg.name)
        prequant = getattr(
            cfg, "quantization_config", None
        )  # test for gpt model dict si trae MXFP4, None si no trae nada

        quantization_config = None

        if prequant is None:
            if mode == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=quant_cfg.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=quant_cfg.bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=getattr(
                        torch, quant_cfg.bnb_4bit_compute_dtype
                    ),
                )

            elif mode == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=quant_cfg.llm_int8_threshold,
                    llm_int8_has_fp16_weight=quant_cfg.llm_int8_has_fp16_weight,
                    llm_int8_enable_fp32_cpu_offload=quant_cfg.llm_int8_enable_fp32_cpu_offload,
                )
            else:
                assert (
                    mode == "none"
                ), f"Quantization mode '{mode}' not supported. Use '4bit', '8bit' or 'none'."

        load_kwargs = {
            "device_map": getattr(self.model_cfg, "device_map", "auto"),
            "dtype": getattr(self.model_cfg, "dtype", "auto"),
        }

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_cfg.name, **load_kwargs
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.name)
        if (
            self.tokenizer.pad_token_id is None
            and self.tokenizer.eos_token_id is not None
        ):
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_messages(self, user_prompt: str):
        is_chat = self.model_cfg.is_chat_model
        has_template = bool(getattr(self.tokenizer, "chat_template", None))
        sys = (self.chat_cfg.system_prompt or "").strip()

        if not is_chat:
            return f"{user_prompt}" if sys else user_prompt

        msgs = []
        if sys:
            msgs.append({"role": "system", "content": sys})

        msgs.append({"role": "user", "content": user_prompt})

        if has_template:
            return self.tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )

        return f"{sys}\n\n{user_prompt}" if sys else user_prompt

    def _messages_to_inputs(self, messages) -> Dict[str, torch.Tensor]:

        batch = self.tokenizer(messages, return_tensors="pt", padding=True)
        return {k: v.to(self.model.device) for k, v in batch.items()}

    def generate_text(self, prompt: str) -> str:

        if self.model_cfg.name.startswith("gpt"):

            response = client.chat.completions.create(
                model="gpt-4o",
                temperature=self.decoding_cfg.temperature,
                messages=[
                    {"role": "system", "content": self.chat_cfg.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip(), estimate_cost(
                response.usage
            )

        msgs = self._build_messages(prompt)
        inputs = self._messages_to_inputs(msgs)
        d = self.decoding_cfg

        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=d.max_new_tokens,
                do_sample=d.do_sample,
                temperature=d.temperature,
                top_p=d.top_p,
                top_k=d.top_k,
                repetition_penalty=d.repetition_penalty,
                num_beams=d.num_beams,
                num_return_sequences=d.num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        seq = out.sequences
        inp_len = inputs["input_ids"].shape[1]

        # if we want to return the prompt + generation
        # gen_only = seq
        # if we want to return only the generation
        gen_only = seq[:, inp_len:]
        decoded_list = self.tokenizer.batch_decode(
            gen_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        decoded_list = [t.strip() for t in decoded_list if t is not None]

        sep = "\n\n" + "*" * 30 + "\n\n"
        generated_text = (
            sep.join(decoded_list) if len(decoded_list) > 1 else decoded_list[0]
        )

        return generated_text, 0.0

    def generate_texts(self, prompts: list[str]) -> list[str]:
        outputs: list[str] = []

        iterator = tqdm(
            prompts,
            total=len(prompts),
            desc="Generating data",
            unit="article",
            dynamic_ncols=True,
            smoothing=0.1,
            leave=True,
        )

        total_cost = 0.0
        for prompt in iterator:
            text, cost = self.generate_text(prompt)
            total_cost += cost
            # iterator.set_postfix({"est. total cost ($)": f"{total_cost:.6f}"})
            outputs.append(text)

        return outputs
