from google.cloud import storage
import io
from PIL import Image
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar
import torch
from torch.utils.data import Dataset, DataLoader,random_split
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
def get_train_val_test_split(transform,train_split=0.8, val_split=0.1):
        dataset = ImageLabelDataset(transform=transform)
        print("Dataset length: ", len(dataset),flush=True)
        l = len(dataset)
        train_size = int(len(dataset) * train_split)

        train_dataset, test_val_dataset = random_split(dataset, [train_size,l-train_size])

        l0 = len(test_val_dataset)
        val_size = int(l* val_split)
        test_size = l0 - val_size

        val_dataset, test_dataset = random_split(test_val_dataset, [val_size, test_size])

        print("Train dataset length: ", len(train_dataset),flush=True)
        print("Validation dataset length: ", len(val_dataset),flush=True)
        print("Test dataset length: ", len(test_dataset),flush=True)
        print("*"*10+"Loading data"+"*"*10,flush=True)
        return train_dataset, val_dataset, test_dataset
def compute_pos_weight_tensor(device,k=1):
    counts = {
        "Atelectasis": 15430,
        "Cardiomegaly": 3609,
        "Effusion": 18029,
        "Infiltration": 27765,
        "Mass": 7696,
        "Nodule": 8715,
        "Pneumonia": 1860,
        "Pneumothorax": 7370,
        "Consolidation": 6078,
        "Edema": 2998,
        "Emphysema": 3308,
        "Fibrosis": 2029,
        "Pleural_Thickening": 4630,
        "Hernia": 292,
        "No Finding": 79030
    }
    
    total_samples = 111601
    if k == 0:
        pos_weights = {label: (total_samples - count) / count for label, count in counts.items()}
    elif k == 1:
        pos_weights = {label: 1/count for label, count in counts.items()}
    for label, weight in pos_weights.items():
        print(f"{label}: {weight:.2f}")
    label_map = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
        "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia", "No Finding"
    ]
    pos_weight_tensor = torch.tensor(
        [pos_weights[label] for label in label_map],
        dtype=torch.float
    ).to(device)
    return pos_weight_tensor

def get_last_image_index():
    directories = [
            '../datasets/nih-chest-xrays/images_001/images/',
            '../datasets/nih-chest-xrays/images_002/images/',
            '../datasets/nih-chest-xrays/images_003/images/',
            '../datasets/nih-chest-xrays/images_004/images/',
            '../datasets/nih-chest-xrays/images_005/images/',
            '../datasets/nih-chest-xrays/images_006/images/',
            '../datasets/nih-chest-xrays/images_007/images/',
            '../datasets/nih-chest-xrays/images_008/images/',
            '../datasets/nih-chest-xrays/images_009/images/',
            '../datasets/nih-chest-xrays/images_010/images/',
            '../datasets/nih-chest-xrays/images_011/images/',
            '../datasets/nih-chest-xrays/images_012/images/'
            ]
    last_index = []
    for directory in directories:
        files = os.listdir(directory)
        files.sort()
        last_index.append(files[-1])
    return last_index





class ImageLabelDataset(Dataset):
    def __init__(self, csv_file='../datasets/nih-chest-xrays/Data_Entry_2017_cleaned.csv', directories=None, transform=None):
        """
        Args:
            csv_file (string): Local path to the CSV file with image indices and labels.
            directories (list of strings): List of local directories containing the images.
                Example: ['../../datasets/nih-chest-xrays/images_001/images/']
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_file)
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
            'Hernia': 13,
            "No Finding": 14
        }
        
        self.directories = [
            '../datasets/nih-chest-xrays/images_001/images/',
            '../datasets/nih-chest-xrays/images_002/images/',
            '../datasets/nih-chest-xrays/images_003/images/',
            '../datasets/nih-chest-xrays/images_004/images/',
            '../datasets/nih-chest-xrays/images_005/images/',
            '../datasets/nih-chest-xrays/images_006/images/',
            '../datasets/nih-chest-xrays/images_007/images/',
            '../datasets/nih-chest-xrays/images_008/images/',
            '../datasets/nih-chest-xrays/images_009/images/',
            '../datasets/nih-chest-xrays/images_010/images/',
            '../datasets/nih-chest-xrays/images_011/images/',
            '../datasets/nih-chest-xrays/images_012/images/'
        ]
        self.transform = transform
        self.last_index = get_last_image_index()
        

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        # Get the image index (filename) and the label string from the DataFrame
        image_index = self.data.iloc[idx]['Image Index']
        label_str = self.data.iloc[idx]['Finding Labels']
        
        # Process the label string (could be multiple labels separated by '|')
        labels = label_str.split('|')
        encoded_labels = [self.label_map[label.strip()] for label in labels if label.strip() in self.label_map]
        
        # Create a multi-hot encoded label vector (14 classes in total)
        multi_hot_label = torch.zeros(len(self.label_map), dtype=torch.long)
        for label in encoded_labels:
            multi_hot_label[label] = 1
        
        # Attempt to find the image file in one of the directories
        image_path = None
        if self.idx1_smallereqto_idx2(image_index,self.last_index[0]):
            image_path = f'{self.directories[0]}{image_index}'
        elif self.idx1_greaterthan_idx2(image_index,self.last_index[0]) and self.idx1_smallereqto_idx2(image_index,self.last_index[1]):
            image_path = f'{self.directories[1]}{image_index}'
        elif self.idx1_greaterthan_idx2(image_index,self.last_index[1]) and self.idx1_smallereqto_idx2(image_index,self.last_index[2]):
            image_path = f'{self.directories[2]}{image_index}'
        elif self.idx1_greaterthan_idx2(image_index,self.last_index[2]) and self.idx1_smallereqto_idx2(image_index,self.last_index[3]):
            image_path = f'{self.directories[3]}{image_index}'
        elif self.idx1_greaterthan_idx2(image_index,self.last_index[3]) and self.idx1_smallereqto_idx2(image_index,self.last_index[4]):
            image_path = f'{self.directories[4]}{image_index}'
        elif self.idx1_greaterthan_idx2(image_index,self.last_index[4]) and self.idx1_smallereqto_idx2(image_index,self.last_index[5]):
            image_path = f'{self.directories[5]}{image_index}'
        elif self.idx1_greaterthan_idx2(image_index,self.last_index[5]) and self.idx1_smallereqto_idx2(image_index,self.last_index[6]):
            image_path = f'{self.directories[6]}{image_index}'
        elif self.idx1_greaterthan_idx2(image_index,self.last_index[6]) and self.idx1_smallereqto_idx2(image_index,self.last_index[7]):
            image_path = f'{self.directories[7]}{image_index}'
        elif self.idx1_greaterthan_idx2(image_index,self.last_index[7]) and self.idx1_smallereqto_idx2(image_index,self.last_index[8]):
            image_path = f'{self.directories[8]}{image_index}'
        elif self.idx1_greaterthan_idx2(image_index,self.last_index[8]) and self.idx1_smallereqto_idx2(image_index,self.last_index[9]):
            image_path = f'{self.directories[9]}{image_index}'
        elif self.idx1_greaterthan_idx2(image_index,self.last_index[9]) and self.idx1_smallereqto_idx2(image_index,self.last_index[10]):
            image_path = f'{self.directories[10]}{image_index}'
        elif self.idx1_greaterthan_idx2(image_index,self.last_index[10]) and self.idx1_smallereqto_idx2(image_index,self.last_index[11]):
            image_path = f'{self.directories[11]}{image_index}'
        else:
            raise FileNotFoundError(f"Image {image_index} not found in any of the specified directories: {self.directories}")
            
        
        
        
        # Open the image file and convert to grayscale ('L' mode)
        image = Image.open(image_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, multi_hot_label
    def idx1_greaterthan_idx2(self, idx1,idx2):
        patient_id1 = int(idx1.split('_')[0])
        patient_id2 = int(idx2.split('_')[0])

        if patient_id1 > patient_id2:
            return True
        elif patient_id1 == patient_id2:
            img_id1 = int(idx1.split('_')[1].split('.')[0])
            img_id2 = int(idx2.split('_')[1].split('.')[0])
            if img_id1 > img_id2:
                return True
            else:
                return False
        else:
            return False
    def idx1_smallereqto_idx2(self, idx1,idx2):
        patient_id1 = int(idx1.split('_')[0])
        patient_id2 = int(idx2.split('_')[0])
        if patient_id1 <= patient_id2:
            if patient_id1 == patient_id2:
                img_id1 = int(idx1.split('_')[1].split('.')[0])
                img_id2 = int(idx2.split('_')[1].split('.')[0])
                if img_id1 <= img_id2:
                    return True
                else:
                    return False
            else:
                return True
        else:
            return False
        


if __name__ == "__main__":
    
    
    last_index = get_last_image_index()
    for i in last_index:
        print(i)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImageLabelDataset( transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=8,shuffle=True)
    for i in tqdm(range(10), desc="Epochs", position=0):
        for images, labels in tqdm(dataloader, desc="Batches", position=1, leave=False):
            continue
            
            