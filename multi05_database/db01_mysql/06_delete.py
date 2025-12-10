import mysql.connector


# 1. pymysql에 접속한다.
conn = mysql.connector.connect(user="root",
                               password="1234",
                               host ="127.0.0.1",
                               database="pymysql")

# 2. cursor 객체를 만든다.
cursor =conn.cursor()

# 3. student 테이블에서 id 가 1인 data를 삭제한다.
queue = "delete from student where id = %s"
cursor.execute(queue, (1,))
print(cursor.rowcount)
conn.commit()


# 4. student 테이블 전체 출력 한다.
cursor.execute("select * from student")
for row in cursor:
    print(row)


# 5. cursor 객체를 닫는다.
cursor.close()

# 6. connection 객체를 닫는다.
conn.close()