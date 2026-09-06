# ==========================================
# RetailAI - MySQL Database Connection
# ==========================================

import mysql.connector


# ==========================================
# DATABASE SETTINGS
# ==========================================

DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "RetailAI@2026"
DB_NAME = "retailai"


# ==========================================
# CREATE DATABASE CONNECTION
# ==========================================

def get_db_connection():

    connection = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

    return connection


# ==========================================
# TEST CONNECTION
# ==========================================

if __name__ == "__main__":

    print("")
    print("======================================")
    print("       RetailAI Database Test")
    print("======================================")
    print("")

    try:

        connection = get_db_connection()

        print("MySQL connection successful!")
        print("Database: retailai")

        connection.close()

        print("")
        print("Database connection test completed.")
        print("")

    except mysql.connector.Error as error:

        print("")
        print("Database connection failed.")
        print("Error:", error)
        print("")