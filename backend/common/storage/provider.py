import os
import io
import shutil
import json
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from typing import BinaryIO, Tuple

import boto3
from botocore.exceptions import ClientError
from open_webui.config import (
    S3_ACCESS_KEY_ID,
    S3_BUCKET_NAME,
    S3_ENDPOINT_URL,
    S3_REGION_NAME,
    S3_SECRET_ACCESS_KEY,
    GCS_BUCKET_NAME,
    GOOGLE_APPLICATION_CREDENTIALS_JSON,
    STORAGE_PROVIDER,
    UPLOAD_DIR,
)
from open_webui.constants import ERROR_MESSAGES


class StorageProvider(ABC):
    @abstractmethod
    def get_file(self, file_path: str) -> str:
        pass

    @abstractmethod
    def upload_file(self, file: BinaryIO, filename: str) -> Tuple[bytes, str]:
        pass

    @abstractmethod
    def delete_all_files(self) -> None:
        pass

    @abstractmethod
    def delete_file(self, file_path: str) -> None:
        pass


class LocalStorageProvider(StorageProvider):
    @staticmethod
    def upload_file(file: BinaryIO, filename: str) -> Tuple[bytes, str]:
        contents = file.read()
        if not contents:
            raise ValueError(ERROR_MESSAGES.EMPTY_CONTENT)
        file_path = f"{UPLOAD_DIR}/{filename}"
        with open(file_path, "wb") as f:
            f.write(contents)
        return contents, file_path

    @staticmethod
    def get_file(file_path: str) -> str:
        """Handles downloading of the file from local storage."""
        return file_path

    @staticmethod
    def delete_file(file_path: str) -> None:
        """Handles deletion of the file from local storage."""
        filename = file_path.split("/")[-1]
        file_path = f"{UPLOAD_DIR}/{filename}"
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            print(f"File {file_path} not found in local storage.")

    @staticmethod
    def delete_all_files() -> None:
        """Handles deletion of all files from local storage."""
        if os.path.exists(UPLOAD_DIR):
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Remove the file or link
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Remove the directory
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            print(f"Directory {UPLOAD_DIR} not found in local storage.")


class S3StorageProvider(StorageProvider):
    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            region_name=S3_REGION_NAME,
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY_ID,
            aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        )
        self.bucket_name = S3_BUCKET_NAME
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME must be set in configuration")

    def upload_file(self, file: BinaryIO, filename: str) -> Tuple[bytes, str]:
        """Uploads a file to S3-compatible storage."""
        contents = file.read()
        if not contents:
            raise ValueError(ERROR_MESSAGES.EMPTY_CONTENT)
            
        local_file_path = f"{UPLOAD_DIR}/{filename}"
        with open(local_file_path, "wb") as f:
            f.write(contents)
        
        try:
            self.s3_client.upload_file(local_file_path, self.bucket_name, filename)
            return contents, f"{S3_ENDPOINT_URL}/{self.bucket_name}/{filename}"
        except ClientError as e:
            raise RuntimeError(f"Error uploading file to S3: {e}")

    def get_file(self, file_path: str) -> str:
        """Downloads a file from S3-compatible storage."""
        try:
            # Handle both URL and non-URL paths
            if file_path.startswith(S3_ENDPOINT_URL):
                bucket_name, key = self._parse_minio_url(file_path)
            else:
                bucket_name = self.bucket_name
                key = file_path

            local_file_path = f"{UPLOAD_DIR}/{key}"
            self.s3_client.download_file(bucket_name, key, local_file_path)
            return local_file_path
        except ClientError as e:
            raise RuntimeError(f"Error downloading file from S3: {e}")

    def delete_file(self, file_path: str) -> None:
        """Deletes a file from S3-compatible storage."""
        _, key = self._parse_s3_path(file_path)
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
        except ClientError as e:
            self.logger.error(f"Error deleting file from S3: {e}")
            raise RuntimeError("Error deleting file from S3.") from e

        # Always delete from local storage
        LocalStorageProvider.delete_file(file_path)

    def delete_all_files(self) -> None:
        """Deletes all files from S3-compatible storage."""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            if "Contents" in response:
                for content in response["Contents"]:
                    self.s3_client.delete_object(Bucket=self.bucket_name, Key=content["Key"])
        except ClientError as e:
            self.logger.error(f"Error deleting all files from S3: {e}")
            raise RuntimeError("Error deleting all files from S3.") from e

        # Always delete from local storage
        LocalStorageProvider.delete_all_files()

    @staticmethod
    def _parse_s3_path(s3_path: str) -> Tuple[str, str]:
        """Parses the S3 path and returns the bucket name and key."""
        try:
            bucket_name, key = s3_path.split("//")[1].split("/", 1)
            return bucket_name, key
        except ValueError as e:
            raise ValueError(f"Invalid S3 path: {s3_path}") from e
        
    @staticmethod
    def _parse_minio_url(file_url: str) -> tuple:
        """Parses a MinIO file URL and returns the bucket name and key."""
        if not file_url.startswith(S3_ENDPOINT_URL):
            raise ValueError(f"Invalid file URL: {file_url}")
            
        # Remove the endpoint URL prefix
        path = file_url[len(S3_ENDPOINT_URL):].lstrip('/')
        parts = path.split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid file URL format: {file_url}")
            
        bucket_name, key = parts
        return bucket_name, key

def get_storage_provider(storage_provider: str):
    if storage_provider == "local":
        Storage = LocalStorageProvider()
    elif storage_provider == "s3":
        Storage = S3StorageProvider()
    else:
        raise RuntimeError(f"Unsupported storage provider: {storage_provider}")
    return Storage


Storage = get_storage_provider(STORAGE_PROVIDER)