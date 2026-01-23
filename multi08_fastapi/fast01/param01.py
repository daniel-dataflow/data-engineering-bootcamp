from fastapi import FastAPI

app = FastAPI()

@app.get("/param01/{param}")
def param01(param):
    return {"parameter01":param}

@app.get("/param02/{param}")
def param02(param: int):
    return {"parameter02":param}

# 터미널에서  uvicorn param01:app --reload
# http://127.0.0.1:8000/docs/ 을 열어서 각종 파라미터를 넣어보자.

from enum import Enum

class Role(str, Enum):
    admin = "admin"
    manager = "manager"
    user = "user"

@app.get("/param03/{role}")
def param03(role: Role):
    if role is Role.admin:
        return {"manage": "Hello, admin"}
    if role is Role.manager:
        return {"manage": "Hello, manager"}
    if role is Role.user:
        return {"manage": "Hello, user"}

    return {"message": "Hello, visitor"}




from fastapi import Path

@app.get("/param04/{name}/{age}")
# 0보다 큰 숫자가 들어가야 한다.
def param04(name: str, age: int=Path(title="test", gt=0)):
    return {"message": f"Hello {name}", "age": age}
