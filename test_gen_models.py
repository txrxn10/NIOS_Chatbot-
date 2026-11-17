import os
import boto3
from dotenv import load_dotenv

load_dotenv()

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "ap-south-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

# Models commonly active in ap-south-1
candidates = [
    # Cross-region / Regional inference profiles for Claude
    "apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    # Amazon native models (always authorized if Bedrock is open)
    "amazon.titan-text-express-v1",
    "amazon.nova-micro-v1:0",
    "amazon.nova-lite-v1:0"
]

print("Scanning for authorized generation models...\n")
for model_id in candidates:
    try:
        resp = bedrock.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": "Hello, respond with 'OK'."}]}],
            inferenceConfig={"maxTokens": 10}
        )
        print(f"✅ AUTHORIZED: {model_id}")
    except Exception as e:
        print(f"❌ DENIED ({model_id}): {str(e).split(':')[-1].strip()}")