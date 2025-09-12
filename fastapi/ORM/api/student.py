from typing import List

from fastapi import APIRouter, HTTPException
from models import *
from pydantic import BaseModel

student_api = APIRouter()

@student_api.get("")
async def get_all_students():
    students = await Student.all()
    for student in students:
        print(student.name, student.sno)
    return {}

class StudentIn(BaseModel):
    name: str
    pwd: str
    sno: int
    class_id: int
    courses: List[int] = []

@student_api.post("")
async def add_student(student_in: StudentIn):
    student = Student(name=student_in.name, pwd=student_in.pwd, sno=student_in.sno, class1_id=student_in.class_id)
    await student.save()
    courses = await Course.filter(id__in=student_in.courses)
    print("courses:", courses)
    await student.courses.add(*courses)

    return student

@student_api.get("/{student_id}")
async def get_student_by_id(student_id: int):
    return {
        "Operation": "select student by id"
    }

@student_api.put("/{student_id}")
async def update_student_by_id(student_id: int):
    return {
        "Operation": "update student by id"
    }

@student_api.delete("/{student_id}")
async def delete_student_by_id(student_id: int):
    deleted_count = await Student.filter(id=student_id).delete()
    if not deleted_count:
        raise HTTPException(status_code=404, detail="student not found")
    return {}