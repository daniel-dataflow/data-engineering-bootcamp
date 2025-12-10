from mysql.connector import connection


conn = connection.MySQLConnection(user='root',
                                  password='1234',
                                  host='127.0.0.1')

cursor = conn.cursor()
# 데이터베이스 만들기
cursor.execute("create database pymysql default character set 'utf8'")

# 데이터베이스 사용
cursor.execute("use pymysql")

# 테이블 만들기
test_table = """
    create table student(
        id int,
        name varchar(100),
        phone char(13),
        birthday date
    )
"""

cursor.execute(test_table)

# 커서 닫기
cursor.close()
# 커넥션 닫기
conn.close()










