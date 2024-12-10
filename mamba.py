"""
@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
"""
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import time

from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from PIL import Image
from torchvision import transforms
from mamba_ssm import Mamba

class MHISTDataset(VisionDataset):
    def __init__(self, root, annotations_file, partition="train", transform=None):
        """
        Args:
            root (str): Root directory of the dataset (the directory containing images folder).
            annotations_file (str): Path to the CSV file with annotations.
            partition (str): Partition to load ("train" or "test").
            transform (callable, optional): Transform to apply to the images.
        """
        super(MHISTDataset, self).__init__(root, transform=transform)
        
        # Load annotations and filter by partition
        self.annotations = pd.read_csv(annotations_file)
        self.annotations = self.annotations[self.annotations['Partition'] == partition]
        
        # Map labels to numeric values
        self.label_map = {"SSA": 0, "HP": 1}
        
        # Store image directory and partition
        self.img_dir = os.path.join(root, "images", "images")  # Assuming the images are in "images/images" directory

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # Get image filename and label from the annotations file
        img_name = self.annotations.iloc[idx, 0]
        label_str = self.annotations.iloc[idx, 1]
        
        # Map the string label to an integer
        label = self.label_map[label_str]
        
        # Load the image
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        image = image.view(-1)
        
        return image, label
    

def get_mhist_dataset(data_path):
    """
    Load the MHIST dataset and return components similar to get_dataset().
    
    Args:
        data_path (str): Path to the directory containing 'images' and 'annotations.csv'.
        
    Returns:
        channel (int): Number of channels in the images (3 for RGB).
        im_size (tuple): Image dimensions (width, height).
        num_classes (int): Number of classes in the dataset.
        class_names (list): List of class names.
        mean (list): Mean for each channel, used for normalization.
        std (list): Standard deviation for each channel, used for normalization.
        dst_train (Dataset): Training dataset.
        dst_test (Dataset): Testing dataset.
        testloader (DataLoader): DataLoader for the test dataset.
    """
    # Set dataset parameters
    channel = 3
    im_size = (224, 224) 
    num_classes = 2
    class_names = ["SSA", "HP"]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Paths to annotations file and root directory
    annotations_file = os.path.join(data_path, "annotations.csv")
    
    # Initialize train and test datasets
    dst_train = MHISTDataset(root=data_path, annotations_file=annotations_file, partition="train", transform=transform)
    dst_test = MHISTDataset(root=data_path, annotations_file=annotations_file, partition="test", transform=transform)
    
    # Create a test DataLoader
    testloader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)
    trainloader = DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=0)

    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, trainloader


class MambaMHISTModel(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes):
        super(MambaMHISTModel, self).__init__()
        self.mamba = Mamba(
            d_model=input_dim,  
            d_state=64,         
            d_conv=4,     
            expand=2            
        )
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.mamba(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x


data_path = "./mhist_dataset"
channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, trainloader = get_mhist_dataset(data_path)

d_model = 128
seq_len = (3 * 224 * 224) // d_model
model = MambaMHISTModel(input_dim=d_model, seq_len=seq_len, num_classes=num_classes).to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
print("Training...")
start_time = time.time()
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for sequences, labels in trainloader:
        sequences, labels = sequences.to("cuda"), labels.to("cuda")

        batch_size = sequences.size(0)
        sequences = sequences.view(batch_size, seq_len, d_model)
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")
end_time = time.time()
print(f"Training time: {end_time - start_time}s")
model.eval()
correct, total = 0, 0
print("Testing...")
with torch.no_grad():
    for sequences, labels in testloader:
        sequences, labels = sequences.to("cuda"), labels.to("cuda")
        batch_size = sequences.size(0)
        sequences = sequences.view(batch_size, seq_len, d_model)
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

