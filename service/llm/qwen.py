from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/data/ljy/model/Qwen3-8B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

class QWEN:
    def call_model(messages):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return content
    
if __name__ == "__main__":
    messages = [{"role": "user", "content": "Give me a short introduction to large language model."}]
    print(QWEN.call_model(messages))




