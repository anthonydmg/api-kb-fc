from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from utils import strings_ranked_by_relatedness

app = FastAPI()

class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = None

class Message(BaseModel):
    content: str = None

class Document(BaseModel):
    content: str = None
    relatedness: float

@app.get("/")
def read_root():
    return {"message": "¡Hola, mundo!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
def create_item(item: Item):
    return item

@app.post("/retrieval_info/", response_model= List[Document])
def retrieval_info(message: Message):
    print("message:", message.content)
    documents = strings_ranked_by_relatedness(message.content)
    return documents