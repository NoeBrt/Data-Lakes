# Previous TP Code:

import os
import pandas as pd
import boto3
import io
from datetime import datetime

def unpack_data(input_dir, output_file,client,bucket_name):
    """
    Unpacks and combines multiple CSV files from a directory into a single CSV file.

    Parameters:
    input_dir (str): Path to the directory containing the CSV files.
    output_file (str): Path to the output combined CSV file.
    """
    data_frames = []
    for folder in os.listdir(input_dir):
        if folder not in ["dev", "test", "train"]:
            continue
        for file_name in os.listdir(os.path.join(input_dir,folder)):
            print(file_name)
            if file_name.endswith('.csv') or 'data-' in file_name:
                    file_path = os.path.join(input_dir,folder, file_name)
                    data = pd.read_csv(
                        file_path,
                        names=['family_id','sequence_name','family_accession','aligned_sequence','sequence'],header=0
                    )
                    data_frames.append(data)
    
    combined_data = pd.concat(data_frames, ignore_index=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_data.to_csv(output_file, index=False)
    try : 
        send_file(output_file,client,bucket_name)
        print(f"File {output_file} sent successfully to {bucket_name} at {datetime.now()}")
    except Exception as e :
        print(f"{output_file} Sending failed : {e} ")
        
        


def init_client():
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566"),
        aws_access_key_id=os.getenv("AWS_API_KEY"),  # Default for LocalStack
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),  # Default for LocalStack
    )
    return s3


def send_file(input_file,client,bucket_name):
    client.upload_file(input_file, bucket_name, os.path.basename(input_file))

     

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unpack and combine protein data")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory")
    parser.add_argument("--output_file_name", type=str, required=True, help="Path to output combined CSV file")
    parser.add_argument("--bucket_name", type=str, required=True, help="s3 bucket")

    args = parser.parse_args()
    s3=init_client()
    unpack_data(args.input_dir, args.output_file_name,s3,args.bucket_name)

#######################################################
# Task: Update this code to handle the following:
# 1. As you are aware, I made a mistake in my code in the last TP and forgot to add a for loop to handle the subfolders in data/raw. 
#    Instead of processing files from a single directory, iterate through the subfolders `train`, `test`, and `dev` under the `input_dir` if you want to re-use this code.
# 2. Combine all files into a single CSV file and save it locally first in a /tmp/ folder
# 3. Use the `boto3` library to handle the upload of the .csv file to the 'raw' S3 bucket you created on LocalStack. The bucket name should be passed as an argument.
# 4. Update the command-line arguments to:
#    - Include `--bucket_name` for the S3 bucket.
#    - Rename `--output_file` to `--output_file_name` to reflect its new purpose.

# Hints:
# - Initialize a boto3 client with the endpoint URL set to LocalStack (`http://localhost:4566`).
# - Replace saving the combined CSV locally as the final step with a temporary save (e.g., `/tmp/`) before uploading it to the S3 bucket.
# - Modify the iteration logic to handle subfolders (`train`, `test`, `dev`) under the `input_dir`.
# - Test your script by running:
#   python src/unpack_data.py --input_dir ./data/raw --bucket_name raw --output_file_name combined_raw.csv
