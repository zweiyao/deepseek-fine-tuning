from unsloth import FastLanguageModel
max_seq_length=2048
dtype=None
load_in_4bit=False
##基于权重加载一个大语言模型和分词器
model_qwen, tokenizer_qwen = FastLanguageModel.from_pretrained(
    model_name = "./Deepseek-R1-Distill-Qwen-7B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
##将模型调整为推理模式
FastLanguageModel.for_inference(model_qwen)

prompt_style_chat = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Please answer the following medical question. 

### Question:
{}

### Response:
<think>{}"""

question_1="ConnectionError: Couldn't reach 'FreedomIntelligence/medical-o1-reasoning-SFT' on the Hub (LocalEntryNotFoundError)"
question_2="Given a patient who experiences sudden-onset chest pain radiating to the neck and left arm, with a past medical history of hypercholesterolemia and coronary artery disease, elevated troponin I levels, and tachycardia, what is the most likely coronary artery involved based on this presentation?"

##将输入的问题转化为标记索引
inputs=tokenizer_qwen([prompt_style_chat.format(question_1,"")],return_tensors="pt").to("cuda")
##带入对话
outputs=model_qwen.generate(
  input_ids=inputs.input_ids,
  max_new_tokens=1200,
  use_cache=True,
)

response=tokenizer_qwen.batch_decode(outputs)

print(response[0].split("### Response:")[1])