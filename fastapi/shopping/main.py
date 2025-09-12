from typing import Union, List, Optional

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field, ValidationError, validator
from datetime import date
from apps.app01.urls import user
from apps.app02.urls import app02
from apps.app03.urls import app03
from apps.app04.urls import app04

app = FastAPI()

app.include_router(app02, prefix="/app02", tags=["app02 api"])
app.include_router(app03, prefix="/app03", tags=["app03 api"])
app.include_router(app04, prefix="/app04", tags=["app04 api"])
app.include_router(user, prefix="/user", tags=["user center api"])

@app.get("/user/{user_id}")
def get_user(user_id: int):
    print(user_id, type(user_id))
    return {"user_id": user_id}

if __name__ == '__main__':
    uvicorn.run(app="main:app", host="127.0.0.1", port=8080, reload=True)