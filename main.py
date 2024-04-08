from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import ml_func


class Item(BaseModel):
    images:list
    type:str

app = FastAPI()
origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/items")
async def create_item(item:Item):
    #print(item)
    print(list(ml_func.get_geo(ml_func.pil_from_base(item.images)[0])))
    return {'images':[{"img":i, 'text': 'text','pos':list(ml_func.get_geo(ml_func.pil_from_base(item.images)[0]))} for i in item.images]}