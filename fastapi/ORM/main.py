from fastapi import FastAPI
import uvicorn

from tortoise.contrib.fastapi import register_tortoise
from setting import TORTOISE_ORM
from api.student import student_api

app = FastAPI()

app.include_router(student_api, prefix="/student", tags=["student api of canvas"])


register_tortoise(
    app=app,
    config=TORTOISE_ORM
)

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8090, reload=True, workers=1)