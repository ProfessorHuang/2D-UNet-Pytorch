import os
import pydicom
import PIL.Image as Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2

class Chaos(Dataset):

    def __init__(self, data_dir, mode):

        base_dir = os.path.join(data_dir, 'CT')
        patient_numbers = os.listdir(base_dir)
        patient_numbers.sort(key=lambda x: eval(x))
        image_dirs = [os.path.join(base_dir, index, 'DICOM_anon') for index in patient_numbers]
        mask_dirs = [os.path.join(base_dir, index, 'Ground') for index in patient_numbers]


        # 5-fold cross validation
        fold_size = len(image_dirs) // 5
        k=1
        val_list = range(k*fold_size, (k+1)*fold_size)

        self.train_image_paths = []
        self.val_image_paths = []
        self.train_mask_paths = []
        self.val_mask_paths = []


        for i in range(len(image_dirs)):
            
            image_files = os.listdir(image_dirs[i])
            image_files.sort(key=image_name_key)
            mask_files = os.listdir(mask_dirs[i])
            mask_files.sort(key=mask_name_key)

            if i in val_list:
                for image_file in image_files:
                    self.val_image_paths.append(os.path.join(image_dirs[i], image_file))
                for mask_file in mask_files:
                    self.val_mask_paths.append(os.path.join(mask_dirs[i], mask_file))
            else:
                for image_file in image_files:
                    self.train_image_paths.append(os.path.join(image_dirs[i], image_file))
                for mask_file in mask_files:
                    self.train_mask_paths.append(os.path.join(mask_dirs[i], mask_file))

        self.mean = [0.3667] 
        self.std = [0.3533]
        self.mode = mode        
    
    def __getitem__(self, i):

        if self.mode == 'train':
            img_path, mask_path = self.train_image_paths[i], self.train_mask_paths[i]
        elif self.mode == 'val':
            img_path, mask_path = self.val_image_paths[i], self.val_mask_paths[i]

        mask = Image.open(mask_path).convert('L')
        mask = transforms.Resize((256,256), interpolation=Image.NEAREST)(mask)
        mask_tensor = transforms.ToTensor()(mask)

        dataset = pydicom.dcmread(img_path)
        HU_img = dataset.RescaleSlope * dataset.pixel_array + dataset.RescaleIntercept
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0
        # change the scope to 0-1
        img = (HU_img - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        img[img>1] = 1.
        img[img<0] = 0
        # resize
        img = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
        img_tensor = transforms.ToTensor()(img)
        img_tensor = transforms.Normalize(mean=self.mean, std=self.std)(img_tensor)

        return {'image': img_tensor, 'mask': mask_tensor}

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_image_paths)
        elif self.mode == 'val':
            return len(self.val_image_paths)

    def cal_mean_std(self):
        image_paths = self.train_image_paths + self.val_image_paths
        image_array = np.zeros((len(image_paths), 512, 512))
        for i in range(len(image_paths)):
            dataset = pydicom.dcmread(image_paths[i])
            HU_img = dataset.RescaleSlope * dataset.pixel_array + dataset.RescaleIntercept
            MIN_BOUND = -1000.0
            MAX_BOUND = 400.0
            img = (HU_img - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
            img[img>1] = 1.
            img[img<0] = 0
            image_array[i,:,:] = img
        return np.mean(image_array), np.std(image_array)

def image_name_key(image_name):
    if image_name[0] == 'i':
        return int(image_name[1:5])
    elif image_name[0:3] == 'IMG':
        return int(image_name[-8:-4])

def mask_name_key(mask_name):
    return int(mask_name[-7:-4])
