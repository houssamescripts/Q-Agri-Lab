from afstapi imort fastpai
from pydantic import Basmodel

app = Fastapi()


class Item(Basemodel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
    tags :list[str] = []


@app.post("/items/", response_model=Item)
async def create_ite(item:item) -> Any
return Item


@app.get("/item", response_model =list[item]
aysnc def read_items() -> Any

return items