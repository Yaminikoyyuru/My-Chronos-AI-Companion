import os

import boto3
from dotenv import load_dotenv

load_dotenv()
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def classify(err):
    msg = str(err)
    if "service control policy" in msg:
        return "BLOCKED BY SCP (AWS/organization rule, IAM cannot fix)"
    if "no identity-based policy allows" in msg or "not authorized" in msg or "AccessDenied" in msg:
        return "IAM permission missing (fixable by adding policy)"
    return "ERROR: " + msg[:140]


def check(name, fn):
    try:
        fn()
        print(f"  {name:18s} OK")
    except Exception as e:
        print(f"  {name:18s} {classify(e)}")


def main():
    print("Region:", REGION)
    try:
        who = boto3.client("sts", region_name=REGION).get_caller_identity()
        print("Account:", who["Account"], "| User:", who["Arn"].split("/")[-1])
    except Exception as e:
        print("Credentials problem:", e)
        return

    print("\nService checks:")
    check("S3", lambda: boto3.client("s3", region_name=REGION).list_buckets())
    check("DynamoDB", lambda: boto3.client("dynamodb", region_name=REGION).list_tables())
    check("SES", lambda: boto3.client("ses", region_name=REGION).list_identities())
    check("Lambda", lambda: boto3.client("lambda", region_name=REGION).list_functions())
    check("API Gateway", lambda: boto3.client("apigatewayv2", region_name=REGION).get_apis())
    check("Bedrock", lambda: boto3.client("bedrock", region_name=REGION).list_foundation_models())
    check("SNS", lambda: boto3.client("sns", region_name=REGION).list_topics())

    sender = os.getenv("SENDER_EMAIL")
    if sender:
        def send_test():
            boto3.client("ses", region_name=REGION).send_email(
                Source=sender, Destination={"ToAddresses": [sender]},
                Message={"Subject": {"Data": "Chronos AWS test"},
                         "Body": {"Text": {"Data": "SES test from aws_check.py"}}})
        check("SES send email", send_test)
    else:
        print("  (set SENDER_EMAIL in .env to also test sending an email)")

    print("\nMeaning: 'BLOCKED BY SCP' = AWS rule on the account. 'IAM permission missing' = just add a policy.")


if __name__ == "__main__":
    main()