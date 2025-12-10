import psycopg


def connect():
    with psycopg.connect("dbname=test", user="postgres", password="1234") as conn:
        with conn.cursor() as cursor:
            cursor.execute("select * from card")
            for row in cursor:
                print(row)



if __name__ == '__main__':
    connect()