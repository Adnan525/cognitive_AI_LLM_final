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
        user_text = "in a paragraph, explain the rationale behind the last move, where all previous moves are; previous "
        if user_text in move:
            inputs = tokenizer(move, return_tensors = "pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
            return tokenizer.decode(outputs[0]).split("\n")[-1][:-5]
        else:
            # prompt_custom = """
            # Instruction:{}."""
            # EOS_TOKEN = tokenizer.eos_token
            # to_send = prompt_custom.format(move) + EOS_TOKEN
            # inputs = tokenizer(to_send, return_tensors = "pt").to("cuda")
            # outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
            # return tokenizer.decode(outputs[0])
            EOS_TOKEN = tokenizer.eos_token
            to_send = move
            inputs = tokenizer(to_send, return_tensors = "pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
            return tokenizer.decode(outputs[0])

# def get_explanantion_for_user_text(move):
# 	inputs = tokenizer(move, return_tensors = "pt").to("cuda")
# 	outputs = model.generate(**inputs, max_new_tokens = 512, use_cache = True)
# 	return tokenizer.decode(outputs[0])

# print(get_explanantion("do you like AI?"))