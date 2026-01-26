'''
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Union

from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import session
from sqlalchemy.sql.functions import user

from database import create_db_and_tables, get_session, User
from fastapi import Depends, HTTPException
from typing import Annotated
from sqlmodel import Session, select



app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


SessionDep = Annotated[Session, Depends(get_session)]

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/user/all", response_class=HTMLResponse, response_model=List[User])
def user_select_all(request: Request, session: SessionDep):

    user_list = session.exec(select(User)).all()

    return templates.TemplateResponse(
        request=request, name="index.html", context={"list": user_list}
    )

@app.post("/user/")
def insert_user(user:User, session: SessionDep):
    session.add(user)
    session.commit()

    # session.refresh(user)
    return user
'''
##################################################

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List
from fastapi.staticfiles import StaticFiles

from database import create_db_and_tables, get_session, User
from fastapi import Depends, HTTPException
from typing import Annotated
from sqlmodel import Session, select

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

SessionDep = Annotated[Session, Depends(get_session)]


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.get("/user/all", response_class=HTMLResponse, response_model=List[User])
def get_user_list(request: Request, session: SessionDep):
    user_list = session.exec(select(User)).all()

    return templates.TemplateResponse(
        request=request, name="index.html", context={"list": user_list}
    )


@app.get("/users/{user_id}", response_model=User)
def get_one_user(user_id: int, session: SessionDep):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/user/")
def insert_user(user: User, session: SessionDep):
    session.add(user)
    session.commit()

    return user


@app.put("/users/{user_id}", response_model=User)
def update_user(user_id: int, user: User, session: SessionDep):
    db_user = session.get(User, user_id)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user.model_dump(exclude_unset=True)
    for key, value in user_data.items():
        setattr(db_user, key, value)

    session.add(db_user)
    session.commit()
    session.refresh(db_user)
    return db_user


@app.delete("/users/{user_id}")
def delete_user(user_id: int, session: SessionDep):
    user = session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    session.delete(user)
    session.commit()
    return {"ok": True}



