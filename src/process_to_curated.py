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
import argparse
import pymongo
from transformers import AutoTokenizer
from datetime import datetime
import pymysql
import pymongo
from datetime import datetime, timezone


def init_mysql():
    print ("Initialisation du client MySQL")
    conn = pymysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL", "root"),
        database=os.getenv("MYSQL_DATABASE", "staging"),
        password=os.getenv("MYSQL_ROOT_PASSWORD", "root"))
    return conn

def init_mongodb():
    print ("Initialisation du client MongoDB")
    client = pymongo.MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017/"))
    return client

def process_data(mysql,mongodb,model_name="distilbert-base-uncased",mongo_collection="wikitext",curated_db="curated"):
    mongo_db = mongodb[curated_db]
    mongo_collection = mongo_db[mongo_collection]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    #extract data from mysql
    cursor = mysql.cursor(pymysql.cursors.DictCursor)
    cursor.execute("SELECT * FROM texts")
    rows = cursor.fetchall()
    
    
    for row in rows:
        tokens = tokenizer(row["text"], return_tensors="pt", padding=True, truncation=True, max_length=128)["input_ids"]
        documents={
            "id":row["id"],
            "text":row["text"],
            "tokens":tokens.tolist(),
            "metadata":{
                "source":"mysql",
                "processed_at":datetime.now(timezone.utc).isoformat()
            }
        }
        print(documents)
        mongo_collection.insert_one(documents)
        print(f"Text {row} inserted successfully to {mongo_collection} at {datetime.now()}")
    cursor.close()
    mysql.close()
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and upload WikiText V2 dataset')
    parser.add_argument('--mysql_name', type=str, required=True, help='The S3 bucket to upload to')
    parser.add_argument('--mongo_db_name', type=str, required=True, help='The S3 bucket to upload to')
    parser.add_argument('--mongo_collection', type=str, default="", help='model used for tokenizer')
    parser.add_argument('--model_name', type=str, default="distilbert-base-uncased", help='model used for tokenizer')
    args = parser.parse_args()
    load_dotenv()
    mongodb=init_mongodb()
    mysql=init_mysql()
    print(f"Processing data from {args.mysql_name} to {args.mongo_db_name}")
    process_data(mysql,mongodb,args.model_name,args.mongo_collection)
    
    #example of command line : python process_to_curated.py --mysql_name staging --mongo_db_name curated --mongo_collection wikitext --model_name distilbert-base-uncased