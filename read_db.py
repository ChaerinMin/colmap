import sqlite3

db_path = '/tmp/brics_calib/db.db'

# find tables 
connection = sqlite3.connect(db_path)
cursor = connection.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in the database:", tables)
connection.close()


table_name = tables[0]

# read a table
connection = sqlite3.connect(db_path)
cursor = connection.cursor()
cursor.execute(f"SELECT * FROM images;")
rows = cursor.fetchall()
for row in rows:
    print(row)
connection.close()
