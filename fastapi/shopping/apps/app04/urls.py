from fastapi import APIRouter
from pydantic import BaseModel, Field
from datetime import date
from typing import List, Optional, Union
from fastapi.staticfiles import StaticFiles
from fastapi import Form, File, UploadFile, Request
from pydantic import BaseModel, EmailStr

app04 = APIRouter()

class UserIn(BaseModel):
    username: str
    password: str
    email: EmailStr
    full_name: Union[str, None] = None

class UserOut(BaseModel):
    username: str
    email: EmailStr
    full_name: Union[str, None] = None

@app04.post("/user", response_model=UserOut)
async def create_user(user:UserIn):
    return user

@app04.post("/regin")
async def data(username: str = Form(), password: str = Form()):
    print(f"username: {username}, password: {password}")
    return {
        "username": username
    }

@app04.post("/file")
async def get_file(file: bytes = File()):
    print("file", file)
    return {
        "file": "file"
    }

@app04.post("/uploadFile")
async def upload_file(file: UploadFile):
    print("file", file)
    return {
        "file": file.filename
    }

@app04.post("/items")
async def items(request: Request):
    print("URL:", request.url)
    print("IP address:", request.client.host)
    print("Header:", request.headers.get("user-agent"))
    print("cookies:", request.cookies)
    return {
        "URL": request.url,
        "IP address": request.client.host,
        "host": request.headers.get("user-agent"),
        "cookies": request.cookies
    }

