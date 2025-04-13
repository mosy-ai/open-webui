import io
import os
import sys
import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from open_webui.storage import provider


def mock_upload_dir(monkeypatch, tmp_path):
    """Fixture to monkey-patch the UPLOAD_DIR and create a temporary directory."""
    directory = tmp_path / "uploads"
    directory.mkdir()
    monkeypatch.setattr(provider, "UPLOAD_DIR", str(directory))
    return directory


def test_imports():
    provider.StorageProvider
    provider.LocalStorageProvider
    provider.S3StorageProvider
    provider.GCSStorageProvider
    provider.MinioStorageProvider
    provider.Storage


def test_get_storage_provider():
    Storage = provider.get_storage_provider("local")
    assert isinstance(Storage, provider.LocalStorageProvider)
    Storage = provider.get_storage_provider("s3")
    assert isinstance(Storage, provider.S3StorageProvider)
    Storage = provider.get_storage_provider("gcs")
    assert isinstance(Storage, provider.GCSStorageProvider)
    Storage = provider.get_storage_provider("minio")
    assert isinstance(Storage, provider.MinioStorageProvider)
    with pytest.raises(RuntimeError):
        provider.get_storage_provider("invalid")


def test_class_instantiation():
    with pytest.raises(TypeError):
        provider.StorageProvider()
    with pytest.raises(TypeError):

        class Test(provider.StorageProvider):
            pass

        Test()
    provider.LocalStorageProvider()
    provider.S3StorageProvider()
    provider.GCSStorageProvider()


class TestLocalStorageProvider:
    Storage = provider.LocalStorageProvider()
    file_content = b"test content"
    file_bytesio = io.BytesIO(file_content)
    filename = "test.txt"
    filename_extra = "test_exyta.txt"
    file_bytesio_empty = io.BytesIO()

    def test_upload_file(self, monkeypatch, tmp_path):
        upload_dir = mock_upload_dir(monkeypatch, tmp_path)
        contents, file_path = self.Storage.upload_file(self.file_bytesio, self.filename)
        assert (upload_dir / self.filename).exists()
        assert (upload_dir / self.filename).read_bytes() == self.file_content
        assert contents == self.file_content
        assert file_path == str(upload_dir / self.filename)
        with pytest.raises(ValueError):
            self.Storage.upload_file(self.file_bytesio_empty, self.filename)

    def test_get_file(self, monkeypatch, tmp_path):
        upload_dir = mock_upload_dir(monkeypatch, tmp_path)
        file_path = str(upload_dir / self.filename)
        file_path_return = self.Storage.get_file(file_path)
        assert file_path == file_path_return

    def test_delete_file(self, monkeypatch, tmp_path):
        upload_dir = mock_upload_dir(monkeypatch, tmp_path)
        (upload_dir / self.filename).write_bytes(self.file_content)
        assert (upload_dir / self.filename).exists()
        file_path = str(upload_dir / self.filename)
        self.Storage.delete_file(file_path)
        assert not (upload_dir / self.filename).exists()

    def test_delete_all_files(self, monkeypatch, tmp_path):
        upload_dir = mock_upload_dir(monkeypatch, tmp_path)
        (upload_dir / self.filename).write_bytes(self.file_content)
        (upload_dir / self.filename_extra).write_bytes(self.file_content)
        self.Storage.delete_all_files()
        assert not (upload_dir / self.filename).exists()
        assert not (upload_dir / self.filename_extra).exists()


@mock_aws
class TestS3StorageProvider:
    def setup_method(self, method):
        self.Storage = provider.S3StorageProvider()
        self.Storage.bucket_name = "test-bucket"
        self.file_content = b"test content"
        self.filename = "test.txt"
        self.filename_extra = "test_extra.txt"
        self.file_bytesio_empty = io.BytesIO()
        
        # Create the bucket if it doesn't exist
        try:
            self.Storage.s3_client.create_bucket(Bucket=self.Storage.bucket_name)
        except self.Storage.s3_client.exceptions.BucketAlreadyOwnedByYou:
            # Bucket already exists, which is fine for our tests
            pass
        except Exception as e:
            raise RuntimeError(f"Failed to setup S3 bucket: {e}")

    def test_upload_file(self, monkeypatch, tmp_path):
        upload_dir = mock_upload_dir(monkeypatch, tmp_path)

        # Upload the file to S3
        contents, s3_file_path = self.Storage.upload_file(
            io.BytesIO(self.file_content), self.filename
        )

        # Verify the file exists in S3
        obj = self.Storage.s3_client.get_object(Bucket=self.Storage.bucket_name, Key=self.filename)
        assert self.file_content == obj["Body"].read()

        # Local checks
        assert (upload_dir / self.filename).exists()
        assert (upload_dir / self.filename).read_bytes() == self.file_content
        assert contents == self.file_content
        assert s3_file_path == f"{provider.S3_ENDPOINT_URL}/{self.Storage.bucket_name}/{self.filename}"

    def test_get_file(self):
        # First upload a file
        self.Storage.upload_file(io.BytesIO(self.file_content), self.filename)
        
        # Test getting the file using the S3 URL
        s3_url = f"{provider.S3_ENDPOINT_URL}/{self.Storage.bucket_name}/{self.filename}"
        local_path = self.Storage.get_file(s3_url)
        assert os.path.exists(local_path)
        assert open(local_path, "rb").read() == self.file_content
        
        # Test getting the file using just the filename
        local_path = self.Storage.get_file(self.filename)
        assert os.path.exists(local_path)
        assert open(local_path, "rb").read() == self.file_content


# class TestGCSStorageProvider:
#     Storage = provider.GCSStorageProvider()
#     Storage.bucket_name = "my-bucket"
#     file_content = b"test content"
#     filename = "test.txt"
#     filename_extra = "test_exyta.txt"
#     file_bytesio_empty = io.BytesIO()

#     @pytest.fixture(scope="class", autouse=True)
#     def setup(self, request):
#         try:
#             host, port = "localhost", 9023
#             server = create_server(host, port, in_memory=True)
#             server.start()
#             os.environ["STORAGE_EMULATOR_HOST"] = f"http://{host}:{port}"

#             gcs_client = storage.Client()
#             bucket = gcs_client.bucket(self.Storage.bucket_name)
#             bucket.create()
#             self.Storage.gcs_client, self.Storage.bucket = gcs_client, bucket

#             def teardown():
#                 bucket.delete(force=True)
#                 server.stop()

#             request.addfinalizer(teardown)
#         except DefaultCredentialsError:
#                pytest.skip("Skipping GCS tests due to missing credentials")

#     def test_upload_file(self, monkeypatch, tmp_path, setup):
#         upload_dir = mock_upload_dir(monkeypatch, tmp_path)
#         # catch error if bucket does not exist
#         with pytest.raises(Exception):
#             self.Storage.bucket = monkeypatch(self.Storage, "bucket", None)
#             self.Storage.upload_file(io.BytesIO(self.file_content), self.filename)
#         contents, gcs_file_path = self.Storage.upload_file(
#             io.BytesIO(self.file_content), self.filename
#         )
#         object = self.Storage.bucket.get_blob(self.filename)
#         assert self.file_content == object.download_as_bytes()
#         # local checks
#         assert (upload_dir / self.filename).exists()
#         assert (upload_dir / self.filename).read_bytes() == self.file_content
#         assert contents == self.file_content
#         assert gcs_file_path == "gs://" + self.Storage.bucket_name + "/" + self.filename
#         # test error if file is empty
#         with pytest.raises(ValueError):
#             self.Storage.upload_file(self.file_bytesio_empty, self.filename)

#     def test_get_file(self, monkeypatch, tmp_path, setup):
#         upload_dir = mock_upload_dir(monkeypatch, tmp_path)
#         contents, gcs_file_path = self.Storage.upload_file(
#             io.BytesIO(self.file_content), self.filename
#         )
#         file_path = self.Storage.get_file(gcs_file_path)
#         assert file_path == str(upload_dir / self.filename)
#         assert (upload_dir / self.filename).exists()

#     def test_delete_file(self, monkeypatch, tmp_path, setup):
#         upload_dir = mock_upload_dir(monkeypatch, tmp_path)
#         contents, gcs_file_path = self.Storage.upload_file(
#             io.BytesIO(self.file_content), self.filename
#         )
#         # ensure that local directory has the uploaded file as well
#         assert (upload_dir / self.filename).exists()
#         assert self.Storage.bucket.get_blob(self.filename).name == self.filename
#         self.Storage.delete_file(gcs_file_path)
#         # check that deleting file from gcs will delete the local file as well
#         assert not (upload_dir / self.filename).exists()
#         assert self.Storage.bucket.get_blob(self.filename) == None

#     def test_delete_all_files(self, monkeypatch, tmp_path, setup):
#         upload_dir = mock_upload_dir(monkeypatch, tmp_path)
#         # create 2 files
#         self.Storage.upload_file(io.BytesIO(self.file_content), self.filename)
#         object = self.Storage.bucket.get_blob(self.filename)
#         assert (upload_dir / self.filename).exists()
#         assert (upload_dir / self.filename).read_bytes() == self.file_content
#         assert self.Storage.bucket.get_blob(self.filename).name == self.filename
#         assert self.file_content == object.download_as_bytes()
#         self.Storage.upload_file(io.BytesIO(self.file_content), self.filename_extra)
#         object = self.Storage.bucket.get_blob(self.filename_extra)
#         assert (upload_dir / self.filename_extra).exists()
#         assert (upload_dir / self.filename_extra).read_bytes() == self.file_content
#         assert (
#             self.Storage.bucket.get_blob(self.filename_extra).name
#             == self.filename_extra
#         )
#         assert self.file_content == object.download_as_bytes()

#         self.Storage.delete_all_files()
#         assert not (upload_dir / self.filename).exists()
#         assert not (upload_dir / self.filename_extra).exists()
#         assert self.Storage.bucket.get_blob(self.filename) == None
#         assert self.Storage.bucket.get_blob(self.filename_extra) == None