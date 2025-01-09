from datasets import load_dataset
import boto3
import os
from dotenv import load_dotenv
from datetime import datetime
import argparse
def download_dataset(name, version, cache_dir,split=True):
    dataset = load_dataset(name, version, cache_dir=cache_dir)
    if(split):
        dataset['train'].to_csv(f"{cache_dir}/train/train.csv", index=False)
        dataset['test'].to_csv(f"{cache_dir}/test/test.csv", index=False)
        dataset['validation'].to_csv(f"{cache_dir}/dev/dev.csv", index=False)
    return dataset


def upload_dataset(client,bucket_name, input_dir):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    for folder in os.listdir(input_dir):
        if folder not in ["dev", "test", "train"]:
            continue
        for file_name in os.listdir(os.path.join(input_dir,folder)):
            input_file=os.path.join(input_dir,folder,file_name)
            try:
                response=client.upload_file(input_file, bucket_name, os.path.basename(input_file))
                print(f"File {input_file} sent successfully to {bucket_name} at {datetime.now()}")
            except Exception as e :
                print(f"{input_file} Sending failed : {e} ")
        

    # Upload the file
    return response
        
def init_client():
    print ("Initialisation du client S3")
    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566"),
        aws_access_key_id=os.getenv("AWS_API_KEY"),  # Default for LocalStack
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),  # Default for LocalStack
    )
    return s3






if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Download and upload WikiText V2 dataset')
    parser.add_argument('--bucket', type=str, required=True, help='The S3 bucket to upload to')
    parser.add_argument('--dataset_dir', type=str, default="./data/raw", help='Directory to cache dataset')
    args = parser.parse_args()
    load_dotenv()
    client =init_client()
    dataset=download_dataset("wikitext", "wikitext-2-raw-v1", args.dataset_dir)
    upload_dataset(client,args.bucket,args.dataset_dir)
    print("Les données WikiText V2 ont été téléchargées et organisées")
    
