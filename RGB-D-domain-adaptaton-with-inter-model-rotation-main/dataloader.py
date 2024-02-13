import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython import display


def pil_loader(path): 
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_sync_dataset(root, label, fold_name='synROD'):
    path01_label2 = []

    with open(label, 'r') as labeltxt:
        for line in labeltxt:
            data = line.strip().split(' ')
            if not is_image(data[0]):
                continue
            if fold_name == 'ROD':
                path = os.path.join(root, '???-washington', data[0])
            else:
                path = os.path.join(root, data[0])

            if fold_name == 'synROD':
                path_rgb = path.replace('***', 'rgb')
                path_depth = path.replace('***', 'depth')
            elif fold_name == 'ROD':
                path_rgb = path.replace('***', 'crop')
                path_rgb = path_rgb.replace('???', 'rgb')
                path_depth = path.replace('***', 'depthcrop')
                path_depth = path_depth.replace('???', 'surfnorm')
            else:
                raise ValueError('Unknown dataset {}. Known datasets are synROD, ROD'.format(fold_name))
            test_label = int(data[1])
            item = (path_rgb, path_depth, test_label)
            path01_label2.append(item)
        return path01_label2


class MyDataset(Dataset):
  def __init__(self, label_path, fold_name = "synROD",flip = False,crop = "center", rotate = False, discrete = True): 
    self.img_path = "ROD-synROD"
    self.label_path = label_path
    self.fold_name = fold_name
    self.crop = crop
    self.discrete = discrete
    self.rotate = rotate
    img_path = os.path.join(self.img_path, self.fold_name)
    imgs =  make_sync_dataset(img_path,self.label_path, fold_name = fold_name)
    self.imgs = imgs
    self.flip = flip 

  def __getitem__(self, index):
    path_rgb, path_depth, label = self.imgs[index] # each "index" corresponds to a specific item i.e. a tuple containing paths and label
    img_rgb = pil_loader(path_rgb)
    img_depth = pil_loader(path_depth)
    #img_rgb.show()
    #img_depth.show() #display before

    img_rgb = TF.resize(img_rgb, (256, 256))
    img_depth = TF.resize(img_depth, (256, 256))
    if self.flip:
      img_rgb = self.flipping(img_rgb)
      img_depth = self.flipping(img_depth)

    if self.rotate == False: 
      img_rgb = self.croping(img_rgb,self.crop)
      img_depth = self.croping(img_depth,self.crop)
      #display(img_rgb)
      #display(img_depth) #display after
      img_rgb = self.to_tensor(img_rgb)
      img_depth = self.to_tensor(img_depth)
      return img_rgb, img_depth, label

    else:
      if self.discrete:
        angles = [0, 90, 180, 270]
        x = random.choice([0, 1, 2, 3])
        y = random.choice([0, 1, 2, 3])
        rot_rgb = angles[x]
        rot_depth = angles[y]
        #print(x)
        #print(y)
        img_rgb = self.rotation(img_rgb, rot_rgb)
        img_depth = self.rotation(img_depth, rot_depth)
        img_rgb = self.croping(img_rgb, self.crop)
        img_depth = self.croping(img_depth, self.crop)
        rel_rot = y - x
        if rel_rot <0 :
          rel_rot +=4
        #img_rgb.show()
        #img_depth.show() #display after
        img_rgb = self.to_tensor(img_rgb)
        img_depth = self.to_tensor(img_depth)
        return img_rgb,img_depth,label, rel_rot 
      else:
        x = random.randint(0, 360)
        y = random.randint(0, 360)
        #x = random.uniform(0,360)
        #y = random.uniform(0,360)     for any float rotation
        img_rgb = self.rotation(img_rgb, x)
        img_depth = self.rotation(img_depth, y)
        img_rgb = self.croping(img_rgb,self.crop)
        img_depth = self.croping(img_depth,self.crop)
        rad_angle = (y - x)*2*np.pi/360
        rad_angle = torch.tensor(rad_angle)
        #display(img_rgb)
        #display(img_depth) #display after
        img_rgb = self.to_tensor(img_rgb)
        img_depth = self.to_tensor(img_depth)
        return img_rgb, img_depth, label, torch.cos(rad_angle), torch.sin(rad_angle)






  def croping(self,img, crop):
    if crop =="center":
        x = (256 - 224)/2
        y = (256 - 224)/2
    else:
        x = random.randint(0, 256 - 224)
        y = random.randint(0, 256 - 224)
    img = TF.crop(img, x, y, 224, 224)
    return img

  def rotation(self,img,rot):
    img = TF.rotate(img, rot)
    return img

  def flipping(self, img):
    if self.flip ==random.choice([True, False]):
      img = TF.hflip(img)
    return img

  def to_tensor(self,img):
    img = TF.to_tensor(img)
    img = TF.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return  img 

  def __len__(self): 
    return len(self.imgs)
      
    
  def __len__(self): 
    return len(self.imgs)
