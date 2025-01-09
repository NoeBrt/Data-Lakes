import os
import pandas as pd
import boto3
import io
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import re
import sqlite3
from dotenv import load_dotenv
import mysql.connector
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
# Découper le texte en paragraphes


def init_mysql():
    print ("Initialisation du client MySQL")
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL", "root"),
        database=os.getenv("MYSQL_DATABASE", "staging"),
        password=os.getenv("MYSQL_ROOT_PASSWORD", "root"))
    return conn
        

def preprocess_data(client,bucket_raw,input_file,mysql):
    """
    Unpacks and combines multiple CSV files from a directory into a single CSV file.

    Parameters:
    input_dir (str): Path to the directory containing the CSV files.
    output_file (str): Path to the output combined CSV file.
    """
        # Load the data
    print('Loading Data...')
    response = client.get_object(Bucket=bucket_raw, Key=input_file)
    text=response['Body'].read().decode("utf-8")
    # Afficher les paragraphes
    tokens= sent_tokenize(text)
    print(f"tokens list size {len(tokens)}")
    # Create a DataFrame from the list
    df = pd.DataFrame(tokens, columns=['texts'])

    # Remove rows that are empty, just quotes, or double quotes
    df = df[~df['texts'].isin(["", "\"", "\"\""])]
    df=df[1:] 
    df['texts'] = df['texts'].str.strip('"')
    # Re}move duplicates
    df = df.drop_duplicates(subset=['texts'])

    # Reset the index after dropping rows
    df.reset_index(drop=True, inplace=True)

    # Check the cleaned data
    print(df.head())
    try:

        cursor = mysql.cursor()

            # Create the table if it doesn't exist
        cursor.execute('''
                CREATE TABLE IF NOT EXISTS texts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    text TEXT NOT NULL
                )
            ''')

            # Insert data into the table
        for text in df['texts']:
            cursor.execute('INSERT INTO texts (text) VALUES (%s)', (text,))
            
            # Commit the changes
        mysql.commit()
        print("Data successfully inserted into MySQL database.")
    finally:
        if mysql.is_connected():
            cursor.close()
            mysql.close()
            print("MySQL connection closed.")
    # Handle missing values


def init_client():
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566"),
        aws_access_key_id=os.getenv("AWS_API_KEY"),  # Default for LocalStack
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),  # Default for LocalStack
    )
    return s3




if __name__ == "__main__":
    import argparse
    load_dotenv()
    mysql=init_mysql()
    parser = argparse.ArgumentParser(description="Preprocess protein data")
    parser.add_argument("--bucket_raw", type=str, required=True, help="Path to raw CSV file")
    parser.add_argument("--input_file", type=str, required=True, help="Path to raw CSV file")
    args = parser.parse_args()
    s3=init_client()
    preprocess_data(s3,args.bucket_raw,args.input_file,mysql)