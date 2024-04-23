import torch
import gc
gc.collect()
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

torch.cuda.empty_cache()

from unsloth import FastLanguageModel

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel


# import zipfile
# import os

# def unzip_file(zip_file, extract_to):
#     with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#         zip_ref.extractall(extract_to)

# # Example usage
# zip_file = "gemma_model_only_ai_gen.zip"
# extract_to = ''

# unzip_file(zip_file, extract_to)


model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "gemma_model_only_ai_gen",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )


FastLanguageModel.for_inference(model) # Enable native 2x faster inference


# import textwrap
# def process_response(response):
    # lines = response.replace("", "").replace("", "").split("\n")
    # wrapped_lines = [textwrap.fill(line, width=100) for line in lines]
    # for wrapped_line in wrapped_lines:
        # print(wrapped_line)

prompt_chess = """
Instruction:{}; previous moves:{}; last move:{}.
Response:{}"""

inputs = tokenizer(
[
    prompt_chess.format(
        "in a paragraph, explain the rationale behind the last move, where all previous moves are",
        "e4 c5 Nf3 Nc6 Bc4 g6 Ng5 Ne5 Bb3 h6 Nf3 Bg7 Nxe5 Bxe5 O-O Qc7 Qf3 Nf6 h3 h5 c3 O-O d3 Nh7 g4 hxg4 Qxg4 d6 Qxg6+ Kh8 Bxf7 Bxh3 Re1 Rg8 Bxg8 Rxg8 Qxg8+ Kxg8 Re3",
        "Qc8",
        ""
    )
], return_tensors = "pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
print(tokenizer.decode(outputs[0]))


