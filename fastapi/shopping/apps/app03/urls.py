from fastapi import APIRouter
from pydantic import BaseModel, Field
from datetime import date
from typing import List, Optional

app03 = APIRouter()

class User(BaseModel):
    name: str = "root"
    age: int = Field(default=0, gt=0, lt=100)
    birth: date = Optional[date]
    friends: List[int]

@app03.post("/data")
async def data(user: User):
    print(user, type(user))
    return {}