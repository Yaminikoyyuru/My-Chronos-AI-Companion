import os

import boto3
from dotenv import load_dotenv

load_dotenv()
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
BUCKET = os.getenv("S3_BUCKET", "")
MODEL = os.getenv("BEDROCK_MODEL_ID", "")


def test_s3():
    print("\n[S3]")
    if not BUCKET:
        print("  S3_BUCKET is not set in .env")
        return
    s3 = boto3.client("s3", region_name=REGION)
    key = "tests/hello.txt"
    try:
        s3.put_object(Bucket=BUCKET, Key=key, Body=b"hello from chronos",
                      ServerSideEncryption="AES256")
        print(f"  upload OK -> s3://{BUCKET}/{key}")
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        print("  download OK:", body.decode())
        s3.delete_object(Bucket=BUCKET, Key=key)
        print("  delete OK  -> S3 is ready")
    except Exception as e:
        print("  S3 problem:", str(e)[:300])


def test_bedrock():
    print("\n[Bedrock]")
    if not MODEL:
        print("  BEDROCK_MODEL_ID is not set in .env")
        return
    try:
        br = boto3.client("bedrock-runtime", region_name=os.getenv("BEDROCK_REGION", REGION))
        resp = br.converse(
            modelId=MODEL,
            messages=[{"role": "user", "content": [{"text": "Reply with the single word: READY"}]}],
            inferenceConfig={"maxTokens": 20, "temperature": 0},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        print("  model replied:", text.strip(), "-> Bedrock is ready")
    except Exception as e:
        print("  Bedrock problem:", str(e)[:400])
        print("  Tip: check the exact Model ID / inference profile ID in the Bedrock console,")
        print("       and that model access is enabled for it.")


if __name__ == "__main__":
    test_s3()
    test_bedrock()