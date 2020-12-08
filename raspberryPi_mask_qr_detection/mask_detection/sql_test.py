import mysql.connector

config = {
    "user": "root",
    "password": "1234",
    "host": "127.0.0.1", #local
    "database": "mysql", #Database name
    "port": "3306" #port는 최초 설치 시 입력한 값(기본값은 3306)
}

conn = mysql.connector.connect(**config)
print(conn)
    # db select, insert, update, delete 작업 객체
cursor = conn.cursor()
    # 실행할 select 문 구성
sql = "SELECT * FROM users"
    # cursor 객체를 이용해서 수행한다.
cursor.execute(sql)
    # select 된 결과 셋 얻어오기
resultList = cursor.fetchall()  # tuple 이 들어있는 list
print(resultList)