from tortoise.models import Model
from tortoise import fields

class Student(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=32, description="teacher name")
    pwd = fields.CharField(max_length=32, description="password")
    sno = fields.IntField(description="student number")

    # one to many relationship
    class1 = fields.ForeignKeyField("models.Class1", related_name="students")

    # many to many relationship
    courses = fields.ManyToManyField("models.Course", related_name="students")

class Course(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=32, description="course name")
    teacher = fields.ForeignKeyField("models.Teacher", related_name="courses")
    addr = fields.CharField(max_length=32, description="course address")

class Class1(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=32, description="class name")

class Teacher(Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=32, description="teacher name")
    pwd = fields.CharField(max_length=32, description="password")
    tno = fields.IntField(description="teacher number")
