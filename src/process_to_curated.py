import io
import pandas as pd
import boto3
from transformers import AutoTokenizer
import os

def process_to_curated(bucket_staging, bucket_curated, input_file, output_file, model_name):
    """
    Processes data from the staging bucket, tokenizes sequences, and uploads the processed file to the curated bucket.

    Steps:
    1. Connect to LocalStack S3 using `boto3` and fetch the input file from the staging bucket.
    2. Ensure the input file contains a `sequence` column.
    3. Use a pre-trained tokenizer to tokenize the sequences in the `sequence` column. 
       The right model's name is already passed as a default argument so you shouldn't worry about that.
       In case you are curious, the tokenizer we are using is associted to META's ESM2 8M model, that was state of the art in protein sequence classification some time ago.
       In case you are even more curious, you can try using tokenizers from other models such as ProtBert, but you will likely need to adapt the preprocessing to those tokenizers.
    4. Drop the original sequence field, add a tokenized sequence field to the data
    5. Save the processed data to a temporary file locally.
    6. Upload the processed file to the curated bucket.

    Parameters:
    - bucket_staging (str): Name of the staging S3 bucket.
    - bucket_curated (str): Name of the curated S3 bucket.
    - input_file (str): Name of the file to process in the staging bucket.
    - output_file (str): Name of the output file to store in the curated bucket.
    - model_name (str): Name of the Hugging Face model for tokenization.
    """
    # Step 1: Initialize S3 client
    # HINT: Use boto3.client and specify the endpoint URL to connect to LocalStack.
    s3=boto3.client("s3",
        endpoint_url=os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566"),
        aws_access_key_id=os.getenv("AWS_API_KEY"),  # Default for LocalStack
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))  # Default for LocalStack
    # Step 2: Download the input file from the staging bucket
    # HINT: Use s3.get_object to download the file and load it into a Pandas DataFrame.
    # Ensure the input file exists and contains a 'sequence' column.
    response = s3.get_object(Bucket=bucket_staging, Key=input_file)
    try: 
        data = pd.read_csv(io.BytesIO(response['Body'].read())) # Use t
    except Exception as e:
        print(f"Could not load {input_file}")
    if "sequence" not in data:
        print("sequence collumn not found")
        exit(1)
    # Step 3: Initialize the tokenizer
    # HINT: Use AutoTokenizer.from_pretrained(model_name) to load a tokenizer for the specified model.
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    # Step 4: Tokenize the sequences
    # HINT: Iterate over the 'sequence' column and use the tokenizer to process each sequence.
    # Use truncation, padding, and max_length=1024 to prepare uniform tokenized sequences. 
    tokenized_out = data["sequence"].map(
        lambda seq: tokenizer(seq, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    )
     # Step 5: Create a DataFrame for tokenized sequences
    # HINT: Convert the tokenized outputs into a DataFrame. Name the columns as `token_0`, `token_1`, etc.
    token_ids = tokenized_out.map(lambda x: x["input_ids"].squeeze().tolist())

    # Step 3: Create a DataFrame from the tokenized sequences
    df_tokenized = pd.DataFrame(token_ids.tolist(), columns=[f"token_{i}" for i in range(512)])

    # Display the resulting DataFrame
    print(data.head())
   
    # Step 6: Merge the tokenized data with the metadata
    # HINT: Exclude the 'sequence' column from the metadata and concatenate it with the tokenized data.
   # Step 6: Merge the tokenized data with the metadata
    data = data.drop(columns=["sequence"])
    final_data = pd.concat([data, df_tokenized], axis=1)

    # Step 7: Save the processed data locally
    local_temp_path = "/tmp/" + output_file
    final_data.to_csv(local_temp_path, index=False)

    # Step 8: Upload the processed file to the curated bucket
    try:
        with open(local_temp_path, "rb") as f:
            s3.upload_fileobj(f, Bucket=bucket_curated, Key=output_file)
        print(f"Processed file '{output_file}' uploaded successfully to bucket '{bucket_curated}'.")
    except Exception as e:
        raise RuntimeError(f"Failed to upload the file to bucket '{bucket_curated}': {e}")
    finally:
        # Cleanup: Remove local temporary file
        if os.path.exists(local_temp_path):
            os.remove(local_temp_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process data from staging to curated bucket")
    parser.add_argument("--bucket_staging", type=str, required=True, help="Name of the staging S3 bucket")
    parser.add_argument("--bucket_curated", type=str, required=True, help="Name of the curated S3 bucket")
    parser.add_argument("--input_file", type=str, required=True, help="Name of the input file in the staging bucket")
    parser.add_argument("--output_file", type=str, required=True, help="Name of the output file in the curated bucket")
    parser.add_argument("--model_name", type=str, default="facebook/esm2_t6_8M_UR50D", help="Tokenizer model name")
    args = parser.parse_args()

    process_to_curated(args.bucket_staging, args.bucket_curated, args.input_file, args.output_file, args.model_name)
