from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import preprocess
import Func


class Item(BaseModel):
    images:list
    type:str

preds = Func.Predictor()

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
    preds.test()
    #if item.type == 'all':
        #preds.get_preds(item.images)
    return {'images':[{"img":i, 'text': 'text','pos':list(preprocess.get_geo(preprocess.pil_from_base(item.images)[0]))} for i in item.images]}