import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv()

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "ap-south-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

def test_titan_embedding():
    print("Testing Amazon Titan Multilingual v2 embedding in ap-south-1...")
    body = json.dumps({
        "inputText": "What is the fee for Senior Secondary theory examination?",
        "dimensions": 1024,
        "normalize": True
    })
    
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        contentType="application/json",
        accept="application/json",
        body=body
    )
    
    response_body = json.loads(response["body"].read())
    embedding = response_body["embedding"]
    print(f"Titan Embedding Success! Vector Dimension: {len(embedding)}")

if __name__ == "__main__":
    try:
        test_titan_embedding()
    except Exception as e:
        print("\nBedrock Invocation Error:")
        print(e)