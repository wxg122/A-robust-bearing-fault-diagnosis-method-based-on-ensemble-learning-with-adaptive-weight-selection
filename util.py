import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

# In[] 用于读取图片
class netDataset(Dataset):
    def __init__(self,path,train_lines,size):
        super(netDataset, self).__init__()
        self.train_path=path
        self.train_lines = train_lines
        self.train_batches = len(train_lines)
        self.size=size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
    def __len__(self):
        return self.train_batches

    def __getitem__(self, index):
#        print(len(self.data_quence))
        annotation_line = self.train_lines[index]
        name = self.train_path+'/'+annotation_line
        # 从文件中读取图像
        jpg = Image.open(name).resize((self.size,self.size), Image.BICUBIC).convert("RGB")
        label=annotation_line.split('.')[0].split('_')[-1]
        label1=int(label)
        if self.transform is not None:
            jpg = self.transform(jpg)
        return jpg, label1
