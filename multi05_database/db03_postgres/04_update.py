import psycopg


def select():
    with psycopg.connect("dbname=test user=postgres password=1234") as conn:
        with conn.cursor() as cursor:
            cursor.execute('select * from member')
            for row in cursor:
                print(row)


def update():
    with psycopg.connect("dbname=test user=postgres password=1234") as conn:
        with conn.cursor() as cursor:
            query = "update member set name = %s where name like (%s)"
            cursor.execute(query, ("김순신", "kim-ss"))
            print(cursor.rowcount)

        conn.commit()

    select()



if __name__ == '__main__':
    select()
    update()