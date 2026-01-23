from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Union

from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


class User(BaseModel):
    name: str
    age: int
    addr: str
    id: Union[int, None] = None

 # db 가져왔다 치고
user_list = [
     {"name": "John", "age": 18, "addr": "suwon", "id":1},
     {"name": "hong-gd", "age": 50, "addr": "seoul", "id":2},
     {"name": "kim-sd", "age": 70, "addr": "sinal", "id":3},
     {"name": "Jeck", "age": 30, "addr": "busan", "id":4},
     {"name": "daniel", "age": 100, "addr": "seoul", "id":5},
]

@app.get("/user/all", response_class=HTMLResponse, response_model=List[User])
def user_select_all(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"list": user_list}
    )

