def select(conn):
    # connection 객체 (conn) 에서 cursor 객체 생성
    # cursor : query 실행, 결과 리턴
    cursor = conn.cursor()
    query = "select * from employees limit 100"

    cursor.execute(query)
    # 쿼리를 출력해준다.
    print(cursor)

    for row in cursor:
        #결과값을 출력해준다.
        print(row)



def connection01():
    import mysql.connector

    conn = mysql.connector.connect(user='root',
                                   password='1234',
                                   host='127.0.0.1',
                                   database='employees')

    select(conn)

    conn.close()

def connection02():
    from mysql.connector import connection

    conn = connection.MySQLConnection(user='root',
                                      password='1234',
                                      host='127.0.0.1',
                                      database='employees')
    select(conn)
    conn.close()


def connection03():
    import mysql.connector

    config = {
        "user": "root",
        "password":"1234",
        "host": "127.0.0.1",
        "database": "employees",
        "raise_on_warnings": True
    }

    conn = mysql.connector.connect(**config)
    select(conn)
    conn.close()
    """
    *arg :  몇개든 상관없이 값이 들어간다.
    **kwaras : 몇개든 상관없이 키와 밸류가 들어간다.
    """


if __name__ == '__main__':
    # connection01()
    # connection02()
    connection03()