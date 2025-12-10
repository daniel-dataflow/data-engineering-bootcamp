import mysql.connector


def select01(cursor):
    query = "select * from student"
    cursor.execute("select * from student")

    for row in cursor:
        print(row)


def select02(cursor):
    query = "select * from student"
    cursor.execute(query)
    row = cursor.fetchone()
    rows = cursor.fetchall()
    print(row)
    print(rows)
    for row in rows:
        print(row)

    # print(cursor.fetchone())


def select03(cursor):
    query = "select * from student"
    cursor.execute(query)

    # print(cursor.description())
    columns = [col[0] for col in cursor.description]
    print(columns)
    for row in cursor:
        print(row)



def select04(cursor):
    name = "hong-gd"
    query = "select * from student where name = %s"

    # (name,) -> tuply 로 전달해줘야 함!!!
    cursor.execute(query, (name,))
    for row in cursor:
        print(row)




if __name__ == '__main__':
    conn = mysql.connector.connect(user="root",
                   password="1234",

                                   host="127.0.0.1",
                                   database="pymysql")
    cursor = conn.cursor()
    # select01(cursor)
    # select02(cursor)
    # select03(cursor)
    select04(cursor)


    cursor.close()
    conn.close()


