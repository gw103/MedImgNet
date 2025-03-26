from google.cloud import storage
import io
from PIL import Image
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

def load_images_from_gcs(bucket_name, directory='datasets/nih-chest-xrays/images_001/images/', num_images=None):
    """
    Load images from Google Cloud Storage.

    Parameters:
    - bucket_name: str, the name of the Google Cloud Storage bucket
    - directory: str, the directory path within the bucket (default is 'datasets/nih-chest-xrays/images_001/images/')
    - num_images: int or None, number of images to load (if None, loads all images)

    Returns:
    - images: list of numpy arrays, the loaded images
    """
    # Initialize the Google Cloud Storage client
    client = storage.Client(project='My First Project')
    bucket = client.get_bucket(bucket_name)
    
    # List all blobs (files) in the specified directory
    blobs = bucket.list_blobs(prefix=directory)
    
    # Prepare to collect images
    images = []
    count = 0

    for blob in tqdm(blobs, desc="Loading images", unit="image"):
        if blob.name.endswith('.png'):
            image_data = blob.download_as_bytes()
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            images.append(image_array)
            count += 1
            if num_images and count >= num_images:
                break
    print(f"Total images loaded: {len(images)}")
    
    return images
