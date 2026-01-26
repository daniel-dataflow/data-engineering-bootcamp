from typing import Optional
from sqlmodel import SQLModel, Field, create_engine, Session

SQLALCHEMY_DATABASE_URL = "mysql+mysqlconnector://root:1234@localhost:3306/mysql"

engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

class User(SQLModel, table=True):
    __tablename__ = "users"
    name: str = Field(index=True)
    age: int
    addr: str
    id: Optional[int] = Field(default=None, primary_key=True)





