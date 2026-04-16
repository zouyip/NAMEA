import csv
import glob
import json
import os
from os.path import join
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class LoadDataset(Dataset):
    def __init__(self, image_root: str, info_dir: str, transform = None) -> None:
        super(LoadDataset).__init__()
        self.image_root = image_root
        self.info_dir = info_dir
        self.image_name = []
        self.image_true_label = []
        self.transform = transform
        self.image_target_label = []
        with open(self.info_dir) as info_csv:
            reader = csv.reader(info_csv)
            next(reader)
            for i in reader:
                self.image_name.append(i[0])
                self.image_true_label.append(int(i[6]))
                self.image_target_label.append(int(i[7]))

    def __getitem__(self, index: int) -> any:
        image = Image.open(join(self.image_root, self.image_name[index]+'.png')).convert('RGB')
        true_label= self.image_true_label[index]
        target_label= self.image_target_label[index]

        if self.transform:
            image = self.transform(image)
        return  image, true_label-1,target_label-1
    
    def __len__(self) -> int:
        return len(self.image_name)


transforms2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CNNDataset(Dataset):
    def __init__(self, adv_path):
        self.transform = transforms2
        paths = glob.glob(os.path.join(adv_path, '*.png'))
        paths = [i.split('\\')[-1] for i in paths]
        print('Using ', len(paths))
        paths = [i.strip() for i in paths]
        self.query_paths = [i.split('.')[0] + '.JPEG' for i in paths]
        self.paths = [os.path.join(adv_path, i) for i in paths]
        with open('image_name_to_class_id_and_name.json', 'r') as ipt:
            self.json_info = json.load(ipt)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        query_path = self.query_paths[index].split('\\')[-1]
        class_id = self.json_info[query_path]['class_id']
        class_name = self.json_info[query_path]['class_name']
        # image_name = path.split('\\')[-1]
        image_name = path.split('/')[-1].split('\\')[-1]
        # deal with image
        img = Image.open(path).convert('RGB')
        img = transforms.Resize((224, 224))(img)
        img = transforms.Compose([transforms.ToTensor()])(img)
        # print(img.shape)
        return img, class_id, image_name

