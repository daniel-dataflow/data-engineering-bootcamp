import psycopg


def get_connection():
    return psycopg.connect("dbname=test user=postgres password=1234", autocommit=True)

def select01():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            sql = "select * from card"
            cursor.execute(sql)
            column_names = [col.name for col in cursor.description]
            print(column_names)

def select02():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            main_cate = "생활"
            sql = f"select * from card where main_cate = %s"

            cursor.execute(sql, (main_cate,))
            for row in cursor:
                print(row)


def select03():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            sql = "select * from card"
            cursor.execute(sql)

            print(cursor.fetchone())
            print(cursor.fetchone())
            print("---------")
            for row in cursor.fetchall():
                print(row)






if __name__ == '__main__':
    # select01()
    # select02()
    select03()
