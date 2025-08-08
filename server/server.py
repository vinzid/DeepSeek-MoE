import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import uvicorn
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Literal, Optional


app = FastAPI()


class MessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    name: Optional[str] = None


class MessageOutput(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None


class Choice(BaseModel):
    message: MessageOutput 


class Request(BaseModel):
    messages: List[MessageInput]


class Response(BaseModel):
    model: str
    choices: List[Choice]


@app.post("/v1/chat/completions", response_model=Response)
async def create_chat_completion(request: Request):
    global model, tokenizer

    print(datetime.now())
    print("\033[91m--received_request\033[0m", request)
    input_tensor = tokenizer.apply_chat_template(request.messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=128000)

    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:])
    result = result.replace('<｜end▁of▁sentence｜>', '')
    end_of_think = result.find("</think>")
    if end_of_think != -1:
        result = result[end_of_think + 8:]
    print(datetime.now())
    print("\033[91m--generated_text\033[0m", result)

    message = MessageOutput(
        role="assistant",
        content=result,
    )
    choice = Choice(
        message=message,
    )
    response = Response(model=sys.argv[1].split('/')[-1].lower(), choices=[choice])
    return response


torch.cuda.empty_cache()

if __name__ == "__main__":
    MODEL_PATH = sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.generation_config = GenerationConfig.from_pretrained(MODEL_PATH)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
