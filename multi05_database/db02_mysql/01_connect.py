# import MySQLdb
import pymysql

# db = MySQLdb.connect(host="localhost",
#                      user = 'root',
#                      passwd = '1234',
#                      database = 'pymysql')


db = pymysql.connect(host = 'localhost',
                     user = 'root',
                     passwd = '1234',
                     database = 'pymysql')


cursor = db.cursor()
cursor.execute("select * from student")
for row in cursor:
    print(row)

# print(cursor.fetchall())
