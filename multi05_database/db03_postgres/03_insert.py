from psycopg import connect, sql
from datetime import date


def get_connection():
    return connect("dbname=test user=postgres password=1234", autocommit=True)


def create():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                           create table member(
                           id serial primary key,
                           name varchar(100),`
                           age integer,
                           birthday date
                       )
                """)

def select():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("select * from member")
            for row in cursor:
                print(row)


def insert01():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            name = "daesung"
            age = 100
            birthday = date(2025, 9, 15)
            sql = """
                    insert into member(name, age, birthday)
                    values (%s, %s, %s)
                  """

            cursor.execute(sql, (name, age, birthday))
            print(cursor.rowcount)

            select()


def insert02():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            name = "hong-gd"
            age = 800
            birthday = date(2025, 12, 25)
            sql = """
                insert into member(name, age, birthday)
                values (%(name)s, %(age)s, %(birthdat)s)
            """

            cursor.execute(sql, {"name": name, "age": age, "birthdat": birthday})
            print(cursor.rowcount)

            select()



def insert03():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            name = "kim-ss"
            age = 60
            birthday = date(2025, 12, 10)
            query = sql.SQL("""
                insert into member(name, age, birthday)
                values({name}, {age}, {birthday})
            """).format(
                name=sql.Literal(name),
                age=sql.Literal(age),
                birthday=sql.Literal(birthday)
            )

            cursor.execute(query)
            print(cursor.rowcount)

            select()


if __name__ == "__main__":
    # create()
    # insert01()
    # insert02()
    insert03()
