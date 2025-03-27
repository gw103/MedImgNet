from google.cloud import storage
import io
from PIL import Image
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

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

from google.cloud import storage
import io
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

class ImageLabelDataset(Dataset):
    def __init__(self, bucket_name, csv_file='datasets/nih-chest-xrays/Data_Entry_2017.csv', directories=None, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with image indices and labels (can be on GCS).
            bucket_name (string): The name of the Google Cloud Storage bucket containing the images.
            directories (list of strings): List of directories containing the images (e.g., ['images_001', 'images_002', ..., 'images_012']).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Initialize the GCS client and get the bucket
        self.client = storage.Client(project='MedImgNet')
        self.bucket = self.client.get_bucket(bucket_name)
        
        # Download the CSV file from GCS as bytes
        blob = self.bucket.blob(csv_file)  # GCS path to the CSV file
        csv_data = blob.download_as_bytes()  # Download CSV file as bytes
        self.data = pd.read_csv(io.BytesIO(csv_data))  # Read CSV file into a pandas DataFrame
        
        # Define the label encoding map
        self.label_map = {
            'Atelectasis': 1,
            'Cardiomegaly': 2,
            'Effusion': 3,
            'Infiltration': 4,
            'Mass': 5,
            'Nodule': 6,
            'Pneumonia': 7,
            'Pneumothorax': 8,
            'Consolidation': 9,
            'Edema': 10,
            'Emphysema': 11,
            'Fibrosis': 12,
            'Pleural_Thickening': 13,
            'Hernia': 14
        }
        
        # Initialize parameters
        self.bucket_name = bucket_name
        self.directories = directories if directories else ['datasets/nih-chest-xrays/images_001/images/']  # Default to images_001 if not specified
        self.transform = transform

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns the image and its corresponding encoded label."""
        # Get the image index and label from the DataFrame
        image_index = self.data.iloc[idx]['Image Index']
        label_str = self.data.iloc[idx]['Finding Labels']
        
        # Encode the label (handle multiple labels)
        labels = label_str.split('|')  # Some samples might have multiple labels
        encoded_labels = [self.label_map[label.strip()] for label in labels if label.strip() in self.label_map]
        label = encoded_labels  # If multiple labels, you could choose how to handle this (e.g., list or multi-hot encoding)
        
        # Try to construct the image file path in each directory
        image_path = None
        for directory in self.directories:
            image_path = f"{directory}{image_index}"
            blob = self.bucket.blob(image_path)
            if blob.exists():  # If the image exists in this directory, break
                break
        if image_path is None:
            raise FileNotFoundError(f"Image {image_index}.png not found in any of the specified directories.")
        
        # Download the image from GCS as bytes
        image_data = blob.download_as_bytes()
        
        # Open the image using PIL
        image = Image.open(io.BytesIO(image_data))

        # Apply any transformations, if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label
