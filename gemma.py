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
	return tokenizer.decode(outputs[0]).split("\n")[1][:-5]

# print(get_explanantion("in a paragraph, explain the rationale behind the last move, where all previous moves are; previous moves:e4 c5 Nf3 Nc6 Bc4 g6 Ng5 Ne5 Bb3 h6 Nf3 Bg7 Nxe5 Bxe5 O-O Qc7 Qf3 Nf6 h3 h5 c3O-O d3 Nh7 g4 hxg4 Qxg4 d6 Qxg6+ Kh8 Bxf7 Bxh3 Re1 Rg8 Bxg8 Rxg8 Qxg8+ Kxg8 Re3; last move:Qc8."))