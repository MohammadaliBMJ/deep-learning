import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ShapeDataset(Dataset):
    def __init__(self, folder_path):
        self.folder = folder_path
        self.transform = transforms.ToTensor()
        self.images = os.listdir(folder_path)

        self.classes = {
            "circle": 0,
            "square": 1,
            "triangle": 2
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.folder, image_name)

        image = Image.open(image_path).convert("L")
        image = self.transform(image)

        # Labels
        image_shape = image_name.split("_")[0]
        image_label = self.classes[image_shape]

        return image, image_label