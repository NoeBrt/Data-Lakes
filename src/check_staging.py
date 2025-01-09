import mysql.connector

# Connexion `a la base MySQL
conn = mysql.connector.connect(
host="localhost",
user="root",
password="root",
database="staging"
)

cursor = conn.cursor()

 # V´erification des donn´ees
cursor.execute("SELECT COUNT(*) FROM texts WHERE text IS NOT NULL;")
count = cursor.fetchone()
print(f"Nombre de lignes valides : {count[0]}")

cursor.close()
conn.close()
