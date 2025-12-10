import pymysql

conn = pymysql.connect(host='localhost',
                       user='root',
                       password='1234',
                       database='pymysql',
                        # cursors.DictCursor : 결과를 딕셔너리로 변경해준다.
                       cursorclass=pymysql.cursors.DictCursor)


with conn:
    with conn.cursor() as cursor:
        sql = "delete from student where id= %s"
        cursor.execute(sql, (1,))


    with conn.cursor() as cursor:
        sql = "select * from student"
        cursor.execute(sql)

        for row in cursor:
            print(row)

    conn.commit()


