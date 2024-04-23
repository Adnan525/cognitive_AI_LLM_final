from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "gemma_model",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

FastLanguageModel.for_inference(model)

prompt_chess = """
Instruction:{}; previous moves:{}; last move:{}.
Response:{}"""

def get_explanantion(move):
	inputs = tokenizer(move, return_tensors = "pt").to("cuda")
	outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
	return tokenizer.decode(outputs[0]).split("\n")[-1][:-5]

# print(get_explanantion("In a paragraph, explain the rationale behind the last move, where all previous moves are - previous moves : e4 c5 Nf3 Nc6 Bb5 g6 O-O Bg7, last move :  Re1."))