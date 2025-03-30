from google.cloud import storage
import io
from PIL import Image
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from torchvision import transforms
from google.cloud import storage


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



class ImageLabelDataset(Dataset):
    def __init__(self, bucket_name, csv_file='datasets/nih-chest-xrays/Data_Entry_2017_cleaned.csv', directories=None, transform=None):
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
            'Atelectasis': 0,
            'Cardiomegaly': 1,
            'Effusion': 2,
            'Infiltration': 3,
            'Mass': 4,
            'Nodule': 5,
            'Pneumonia': 6,
            'Pneumothorax': 7,
            'Consolidation': 8,
            'Edema': 9,
            'Emphysema': 10,
            'Fibrosis': 11,
            'Pleural_Thickening': 12,
            'Hernia': 13
        }
        
        # Initialize parameters
        self.bucket_name = bucket_name
        self.directories = directories if directories else ['datasets/nih-chest-xrays/images_001/images/']  # Default to images_001 if not specified
        self.transform = transform

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns the image and its corresponding multi-hot encoded label."""
        # Get the image index and label from the DataFrame
        image_index = self.data.iloc[idx]['Image Index']
        label_str = self.data.iloc[idx]['Finding Labels']
        
        # Encode the label (handle multiple labels)
        labels = label_str.split('|')  # Some samples might have multiple labels
        encoded_labels = [self.label_map[label.strip()] for label in labels if label.strip() in self.label_map]
        
        # Create a multi-hot encoded label vector (14 classes in total)
        multi_hot_label = torch.zeros(len(self.label_map), dtype=torch.long)
        for label in encoded_labels:
            multi_hot_label[label] = 1
        
        # Try to construct the image file path in each directory
        image_path = None
        for directory in self.directories:
            image_path = f"{directory}{image_index}"
            blob = self.bucket.blob(image_path)
            if blob.exists(): 
                break
        if image_path is None:
            raise FileNotFoundError(f"Image {image_index}.png not found in any of the specified directories.")
        
        # Download the image from GCS as bytes
        image_data = blob.download_as_bytes()
        
        # Open the image 
        image = Image.open(io.BytesIO(image_data))

        if self.transform:
            image = self.transform(image)
        
        return image, multi_hot_label

if __name__ == "__main__":
    bucket_name = 'med-img-net'
    directories = [
    'datasets/nih-chest-xrays/images_001/images/',
    'datasets/nih-chest-xrays/images_002/images/',
    'datasets/nih-chest-xrays/images_003/images/',
    'datasets/nih-chest-xrays/images_004/images/',
    'datasets/nih-chest-xrays/images_005/images/',
    'datasets/nih-chest-xrays/images_006/images/',
    'datasets/nih-chest-xrays/images_007/images/',
    'datasets/nih-chest-xrays/images_008/images/',
    'datasets/nih-chest-xrays/images_009/images/',
    'datasets/nih-chest-xrays/images_010/images/',
    'datasets/nih-chest-xrays/images_011/images/',
    'datasets/nih-chest-xrays/images_012/images/'
]
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImageLabelDataset(bucket_name, directories=directories, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4,shuffle=True)
    for i in tqdm(range(10), desc="Epochs", position=0):
        for images, labels in tqdm(dataloader, desc="Batches", position=1, leave=False):
            continue
            
            