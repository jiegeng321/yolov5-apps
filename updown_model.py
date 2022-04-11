import boto3
import os
import argparse
from botocore.exceptions import ClientError

aws_access_key_id = os.environ.get("S3_ACCESS_KEY")
aws_secret_access_key = os.environ.get("S3_SECRET_KEY")
endpoint_url = os.environ.get("S3_ENDPOINTS")

s3 = boto3.client(
    service_name="s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    endpoint_url=endpoint_url,
)

def upload(local, remote, bucket="ai_vision"):
    s3 = boto3.client(
        service_name="s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url)

    try:
        response = s3.head_object(
            Bucket=bucket,
            Key=remote)
    except ClientError as e:
        print("uploading the file....")
        try:
            s3.upload_file(local, bucket, remote)
            print("the file has be uploaded.")
        except ClientError as e:
            print(e)
            return False
    return True


def download(local, remote, bucket="ai_vision"):
    local_dir = os.path.dirname(local)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    s3.download_file(bucket, remote, local)


def download_dir(local_dir, remote_dir, bucket='ai_vision'):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    remote_dir = os.path.dirname(remote_dir)
    objects = s3.list_objects(Bucket=bucket, Prefix=remote_dir)["Contents"]
    for object in objects:
        file_name = object.get("Key")
        if file_name[-1] != "/":
            _local_dir = local_dir + file_name.split("/")[-1] if local_dir[-1] == "/" else local_dir + "/" + file_name.split("/")[-1]
            s3.download_file(bucket, file_name, _local_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="download/upload file to object storage service"
    )
    parser.add_argument(
        "-l",
        "--local",
        type=str,
        dest="local_dir",
        required=True,
        help="local file dir",
    )
    parser.add_argument(
        "-b", "--bucket", type=str, dest="bucket", required=True, help="bucket name"
    )
    parser.add_argument(
        "-r",
        "--remote",
        type=str,
        dest="remote_dir",
        required=True,
        help="remote file dir",
    )
    args = parser.parse_args()

    print(args.bucket, args.local_dir, args.remote_dir)
    # upload(args.local_dir, args.remote_dir)
    download_dir(args.local_dir, args.remote_dir, args.bucket)

