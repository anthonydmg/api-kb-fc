from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from utils import count_num_tokens, strings_ranked_by_relatedness, join_docs

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

class Message(BaseModel):
    content: str = None
    token_budget: int = 2500

class Document(BaseModel):
    content: str = None
    relatedness: float
class Information(BaseModel):
    content: str = None
class CountTokens(BaseModel):
    num_tokens: int

@app.get("/")
def read_root():
    return {"message": "¡Hola, mundo!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
def create_item(item: Item):
    return item

@app.post("/retrieval_info/", response_model= Information)
def retrieval_info(message: Message):
    print("message:", message.content)
    documents = strings_ranked_by_relatedness(message.content)

    information = join_docs(message.content, documents, message.token_budget)

    return {"content": information}

@app.post("/count_tokens/", response_model = CountTokens)
def count_token(messages: List[Message]):
    #print("messages:", messages)
    num_tokens = sum([count_num_tokens(m.content) for m in messages])
    return { "num_tokens": num_tokens}