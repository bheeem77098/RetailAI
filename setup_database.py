import mysql.connector
from getpass import getpass

print("")
print("======================================")
print("       RetailAI Database Setup")
print("======================================")
print("")

password = getpass("Enter MySQL root password: ")

try:
    # Connect to MySQL Server
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password=password
    )

    cursor = connection.cursor()

    print("MySQL connection successful!")

    # Create database
    cursor.execute(
        "CREATE DATABASE IF NOT EXISTS retailai"
    )

    print("Database 'retailai' is ready.")

    # Select database
    cursor.execute("USE retailai")

    # Customer analytics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_analytics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            customer_count INT NOT NULL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Zone analytics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS zone_analytics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            zone_name VARCHAR(100) NOT NULL,
            current_count INT DEFAULT 0,
            visits INT DEFAULT 0,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Queue analytics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queue_analytics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            people INT NOT NULL,
            status VARCHAR(30),
            estimated_wait_minutes INT,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Hourly analytics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hourly_analytics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            hour_label VARCHAR(20) NOT NULL,
            visitor_count INT DEFAULT 0,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()

    print("")
    print("======================================")
    print("Database setup completed!")
    print("======================================")
    print("")
    print("Database: retailai")
    print("")
    print("Tables created:")
    print(" - customer_analytics")
    print(" - zone_analytics")
    print(" - queue_analytics")
    print(" - hourly_analytics")
    print("")

    cursor.close()
    connection.close()

except mysql.connector.Error as error:

    print("")
    print("Database connection failed.")
    print("")
    print("Error:", error)
    print("")