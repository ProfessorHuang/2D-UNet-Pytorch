from torch.utils.data import Dataset
import SimpleITK as sitk
from skimage.exposure import equalize_adapthist
import torchvision.transforms as transforms
import os
import numpy as np
import cv2

class Promise12(Dataset):
    
    def __init__(self, data_dir, mode):

        # store data in the npy file
        np_data_path = os.path.join(data_dir, 'npy_image')
        if not os.path.exists(np_data_path):
            os.makedirs(np_data_path)
            data_to_array(data_dir, np_data_path, 256, 256)
        else:
            print('read the data from: {}'.format(np_data_path))
   
        self.mode = mode
        # read the data from npy
        self.X_train = np.load(os.path.join(np_data_path, 'X_train.npy'))
        self.y_train = np.load(os.path.join(np_data_path, 'y_train.npy'))
        self.X_val = np.load(os.path.join(np_data_path, 'X_val.npy'))
        self.y_val = np.load(os.path.join(np_data_path, 'y_val.npy'))
        
        

    def __getitem__(self, i):

        if self.mode == 'train':
            img, mask = self.X_train[i], self.y_train[i]
        elif self.mode == 'val':
            img, mask = self.X_val[i], self.y_val[i]
          
        img_tensor = transforms.ToTensor()(img)
        mask_tensor = transforms.ToTensor()(mask.astype(np.float32))

        return {'image': img_tensor, 'mask': mask_tensor}

    def __len__(self):
        if self.mode == 'train':
            return self.X_train.shape[0]
        elif self.mode == 'val':
            return self.X_val.shape[0]


def data_to_array(base_path, store_path, img_rows, img_cols):

    fileList =  os.listdir(base_path)

    fileList = sorted((x for x in fileList if '.mhd' in x))

    val_list = [5, 15, 25, 35, 45]
    train_list = list(set(range(50)) - set(val_list) )
    count = 0
    for the_list in [train_list,  val_list]:
        images = []
        masks = []

        filtered = [file for file in fileList for ff in the_list if str(ff).zfill(2) in file ]

        for filename in filtered:

            itkimage = sitk.ReadImage(os.path.join(base_path, filename))
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
                masks.append(imgs)
            else:
                imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
                images.append(imgs)

        # images: slices x w x h ==> total number x w x h
        images = np.concatenate(images , axis=0 ).reshape(-1, img_rows, img_cols)
        masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols)
        masks = masks.astype(np.uint8)

        # Smooth images using CurvatureFlow
        images = smooth_images(images)
        images = images.astype(np.float32)

        if count==0: 
            mu = np.mean(images)
            sigma = np.std(images)
            images = (images - mu)/sigma

            np.save(os.path.join(store_path, 'X_train.npy'), images)
            np.save(os.path.join(store_path,'y_train.npy'), masks)
        elif count==1:
            images = (images - mu)/sigma
            np.save(os.path.join(store_path, 'X_val.npy'), images)
            np.save(os.path.join(store_path,'y_val.npy'), masks)
        count+=1

def img_resize(imgs, img_rows, img_cols, equalize=True):

    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist(img, clip_limit=0.05)

        new_imgs[mm] = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST )

    return new_imgs

def smooth_images(imgs, t_step=0.125, n_iter=5):
    """
    Curvature driven image denoising.
    In my experience helps significantly with segmentation.
    """

    for mm in range(len(imgs)):
        img = sitk.GetImageFromArray(imgs[mm])
        img = sitk.CurvatureFlow(image1=img,
                                        timeStep=t_step,
                                        numberOfIterations=n_iter)

        imgs[mm] = sitk.GetArrayFromImage(img)

    return imgs
