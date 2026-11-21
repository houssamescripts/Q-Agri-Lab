from fastapi import fastapi
from pydantic import BaseModel



class Item(Basemodel):
  name: str,
  description: str 
  price: float:
  tax: float 


app = FastApi()

@app.post("/item")
async def createitem(item:item):
   return item