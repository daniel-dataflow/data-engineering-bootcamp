from fastapi import FastAPI


app = FastAPI()

@app.get("/param01/")
def param01(name: str, age: int):
    return {"message": f"Hello {name}", "age": age}

# 터미널에서  uvicorn param02:app --reload
# http://localhost:8000/param01/?name=daesung&age=100 으로 요청해보자.

@app.get("/param02")
def param02(name: str="hang-gd", age: int=0):
    return {"message": f"Hello {name}", "age": age}


from typing import Union

@app.get("/param03")
def param03(name: str="hang-gd", age: Union[int, None]= None):
    if age:
        return {"message": f"Hello {name}", "age": age}

    return {"message": f"Hello {name}"}

# http://127.0.0.1:8000/docs/ 요청한다.
# age 값이 들어가지 않아도 나온다.



from fastapi import Query

@app.get("/param04")
def param04(name: str, addr: Union[str, None]= Query(default=None, max_lenght=100)):
    return {"message": f"Hello {name}"}
    if addr:
        results.update({"addr": addr})

    return results

#  Query을 통해 제한을 걸 수 있다.



from typing import List

@app.get("/param05")
def param05(name: str, hobby: Union[List[str], None] = Query(default=None)):
    results = {"message": f"Hello {name}"}

    if hobby:
        hobby_list = {"hobby": hobby}
        results.update(hobby_list)

    return results
# http://localhost:8000/param05?name=daesung&hobby=mysic&hobby=book



from pydantic import BaseModel, Field
from typing import Annotated

class User(BaseModel):
    name: str
    age: int = Field(default=0, gte=0)
    addr: str = Field(default=None)

@app.get("/user")
def get_user(user: Annotated[User, Query()]):
    return user

# post 인 경우는 괜찮지만 get 방식인 경우에 Field 을 사용한 경우 Annotated(주석)을 사용해야 정상적으로 받아준다.
# http://localhost:8000/user?name=daniel&age=100&addr=seoul
# http://localhost:8000/user?name=daniel


