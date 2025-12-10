import mysql.connector
from datetime import datetime, date

def insert01(cursor):
    query = """
                insert into student(id, name, phone, birthday)
                values (%s, %s, %s, %s) 
            """
    insert_data = (1, "dongheon", "010-1234-5678", date(2025,9,15))
    cursor.execute(query, insert_data)

    print(f"inserted : {cursor.rowcount}")


def insert02(conn):
    cursor = conn.cursor()
    query = """
                insert into student(id, name, phone, birthday)
                values (%s, %s, %s, %s) 
            """

    insert_list = [
        (2, 'hong-gd', '010-1111-1111', date(2025, 9, 15)),
        (3, 'kim-sd', '010-2222-2222', datetime(2025, 9, 15, 0, 0)),
        (4, 'lee-ss', '010-3333-3333', datetime(2026, 3, 13, 18, 0 ))
    ]
    cursor.executemany(query, insert_list)
    print(f"inserted : {cursor.rowcount}")
    cursor.close()
    conn.commit()


def select(cursor):
    cursor.execute("select * from student")

    for row in cursor:
        print(row)



def connection():
    config = {
        "user": "root",
        "password": "1234",
        "host": "127.0.0.1",
        "database": "pymysql"
    }

    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()

    insert01(cursor)
    insert02(conn)
    select(cursor)

    cursor.close()
    conn.close()


if __name__ == '__main__':
    connection()