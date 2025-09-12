from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def home():
    return {"message": "hello, world"}

@app.get("/shop")
async def shop():
    return {"shop:": "subway"}

@app.post(
    "/items",
    tags = ["This is the items test api"],
    summary = "this is items test summary",
    description = "this is items test description.",
    )
async def items():
    return {"items": "item1, item2, item3"}

if __name__ == '__main__':
    uvicorn.run(app="fastapiQuickstart:app", host="127.0.0.1", port=8080, reload=True)