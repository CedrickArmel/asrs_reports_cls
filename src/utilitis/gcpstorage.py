"""Utilities to deal with GCP buckets.
"""
import os
from typing import List

from google.cloud.exceptions import PreconditionFailed
from google.cloud.storage import Client, transfer_manager


def download_blob(bucket_name: str,
                  source_blob_name: str,
                  destination_file_name: str) -> None:
    """Downloads a blob from the bucket.

    Args:
        bucket_name (str): GCP bucket name.
        source_blob_name (str): Object to download name.
        destination_file_name (str): Local path where to put the downloaded \
            obeject.
    """
    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)
    # blob prefered here to get_blob method since we doesn't retrieve any
    # content immediately.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"""Downloaded storage object {source_blob_name} from bucket \
            {bucket_name} to local file {destination_file_name}.""")


def download_bucket_with_transfer_manager(bucket_name: str,
                                          destination_directory: str = "",
                                          workers: int = 8,
                                          max_results: int = 1000) -> None:
    """Download all of the blobs in a bucket, concurrently in a process pool.
    The filename of each blob once downloaded is derived from the blob name and
    the `destination_directory `parameter. For complete control of the filename
    of each blob, use transfer_manager.download_many() instead.

    Directories will be created automatically as needed, for instance to
    accommodate blob names that include slashes.

    Args:
        bucket_name (str): GCP bucket name.
        destination_directory (str, optional): Local path where to put the \
            downloaded objects. "" for the current dir. Defaults to "".
        workers (int, optional): The maximum number of processes to use for \
            the operation. Defaults to 8.
        max_results (int, optional): The maximum number of results to fetch \
            from bucket.list_blobs(). Defaults to 1000.
    """
    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)

    blob_names = [blob.name for blob in bucket.
                  list_blobs(max_results=max_results)]

    results = transfer_manager.\
        download_many_to_path(bucket, blob_names,
                              destination_directory=destination_directory,
                              max_workers=workers)
    for name, result in zip(blob_names, results):
        # The results list is either `None` or an exception for each blob in
        # the input list, in order.
        if isinstance(result, Exception):
            print(f"""Failed to download {name} due to exception: {result}.""")
        else:
            print(f"""Downloaded {name} to {destination_directory + name}.""")


def download_many_blobs_with_transfer_manager(bucket_name: str,
                                              blob_names: List[str],
                                              destination_directory: str = "",
                                              workers: int = 8) -> None:
    """Download blobs in a list by name, concurrently in a process pool.

    The filename of each blob once downloaded is derived from the blob name and
    the `destination_directory `parameter. For complete control of the filename
    of each blob, use transfer_manager.download_many() instead.

    Directories will be created automatically as needed to accommodate blob
    names that include slashes.

    Args:
        bucket_name (str): GCP bucket name.
        blob_names (List[str]): The list of blob names to download. \
            The names of each blobs will also be the name of each \
            destination file (use transfer_manager.download_many() \
            instead to control each destination file name). If there \
            is a "/" in the blob name, then corresponding directories \
            will be created on download.
        destination_directory (str, optional): Local path where to put the \
            downloaded objects. "" for the current dir. Defaults to "".
        workers (int, optional): The maximum number of processes to use for \
            the operation. Defaults to 8.
    """
    storage_client = Client()
    bucket = storage_client.bucket(bucket_name)
    results = transfer_manager.\
        download_many_to_path(bucket, blob_names,
                              destination_directory=destination_directory,
                              max_workers=workers)

    for name, result in zip(blob_names, results):
        if isinstance(result, Exception):
            print(f"""Failed to download {name} due to exception: {result}.""")
        else:
            print(f"""Downloaded {name} to {destination_directory + name}.""")


def copy_blob(bucket_name: str,
              blob_name: str,
              destination_bucket_name: str,
              destination_blob_name: str,
              copy_if_exists: bool = False) -> None:
    """Copies a blob from one bucket to another with a new name.

    Args:
        bucket_name (str): your-bucket-name.
        blob_name (str): your-object-name.
        destination_bucket_name (str): destination-bucket-name.
        destination_blob_name (str): destination-object-name.
        copy_if_exists (bool): Whether to proceed copy if file already exists.\
            Defaults to True.
    """
    storage_client = Client()
    source_bucket = storage_client.bucket(bucket_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)
    source_blob = source_bucket.blob(blob_name)
    if copy_if_exists:
        destination_blob = destination_bucket.blob(destination_blob_name)
        try:
            dest_generation = destination_blob.generation
            blob_copy = source_bucket.\
                copy_blob(source_blob,
                          destination_bucket,
                          destination_blob_name,
                          if_generation_match=dest_generation)
            print(f"""Blob {source_blob.name} in bucket {source_bucket.name} \
                  copied to blob {blob_copy.name} in bucket \
                    {destination_bucket.name}""")
        except TypeError:
            print(f"""Blob {source_blob.name} doesn't exist in destination.\
                   Creating new object""")
            blob_copy = source_bucket.copy_blob(source_blob,
                                                destination_bucket,
                                                destination_blob_name,
                                                if_generation_match=0)
            print(f"""Blob {source_blob.name} in bucket {source_bucket.name} \
                  copied to blob {blob_copy.name} in bucket \
                    {destination_bucket.name}""")
    else:
        try:
            blob_copy = source_bucket.copy_blob(source_blob,
                                                destination_bucket,
                                                destination_blob_name,
                                                if_generation_match=0)
            print(f"""Blob {source_blob.name} in bucket {source_bucket.name} \
                copied to blob {blob_copy.name} in bucket \
                    {destination_bucket.name}""")
        except PreconditionFailed:
            print(f"""Blob {source_blob.name} already exists in destination.\
                   Skipping copy.""")


def copy_many_blobs(bucket_name: str,
                    folder_path: str,
                    destination_bucket_name: str,
                    destination_folder_path: str,
                    copy_if_exists: bool = False) -> None:
    """Copies the blobs from one bucket folder to another.

    Args:
        bucket_name (str): your-bucket-name.
        folder_path (str): source-folder-to-copy-files-from.
        destination_bucket_name (str): destination-folder-to copy-files-to.
        copy_if_exists (bool, optional): Whether to proceed copy if file \
            already exists. Defaults to True.
    """
    storage_client = Client()
    source_bucket = storage_client.bucket(bucket_name)
    prefix = os.path.join(folder_path, "")
    for blob in source_bucket.list_blobs(prefix=prefix):
        if blob.name == prefix:
            pass
        destination_blob_name = os.path.join(destination_folder_path,
                                             blob.name[len(prefix):])
        copy_blob(bucket_name,
                  blob.name,
                  destination_bucket_name,
                  destination_blob_name,
                  copy_if_exists)
