import sqlite3
c = sqlite3.connect("copilot.db")
for name, sql in c.execute("select name, sql from sqlite_master where type='table'"):
    print(name)
    print(sql)
    print()