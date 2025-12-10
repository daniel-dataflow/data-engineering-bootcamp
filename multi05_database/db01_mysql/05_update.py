import mysql.connector


conn = mysql.connector.connect(user="root",
                               passwd="1234",
                               host="127.0.0.1",
                               database="pymysql")

cursor = conn.cursor()

query = "update student set phone = %(phone)s, name = %(name)s where id = %(id)s"
cursor.execute(query,{"phone":"010-9999-9999", "name" : "동헌", "id" : 1})
print(cursor.rowcount)
conn.commit()

cursor.execute("select * from student")
for row in cursor:
    print(row)

cursor.close()
conn.close()



