import psycopg


def select():
    with psycopg.connect("dbname=test user=postgres password=1234") as conn:
        with conn.cursor() as cursor:
            cursor.execute('select * from member')
            for row in cursor:
                print(row)

def delete():
    with psycopg.connect("dbname=test user=postgres password=1234") as conn:
        with conn.cursor() as cursor:
            query= "delete from member where name =%s"
            cursor.execute(query, ("김순신",))
            print(cursor.rowcount)

        conn.commit()
    select()

if __name__ == '__main__':
    delete()