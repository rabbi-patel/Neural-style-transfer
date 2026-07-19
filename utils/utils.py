'''the classes that inherit datasets you can pass that lass in data loader. 
the data creates batches automatically using multiprocessing.
create just dataset class and it's functions. '''


from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class ImageFolderDataset(Dataset):
    #1.constructor for initialiser
    #2.get item and length item we need to define
    def __init__(self,root,transform):
        super(ImageFolderDataset,self).__init__()
        self.root=root
        self.transform=transform
        self.files=list(os.listdir(root))  #returns all the files and folders inside a dir.
        "Look inside the folder specified by root, collect the names of all files, "
        "store them in a list, and save that list in self.files so the dataset can access"
        " images one by one later."
        #filter for just image files.
        self.files=[p for p in self.files if p.endswith(('.jpg','.png','.jpeg'))]
         
    def __len__(self):  #to get the length of te data
        return len(self.files)
    
    def __getitem__(self, idx):
        image_path=os.path.join(self.root,self.files[idx])
        image=Image.open(image_path).convert('RGB')  #we convert our data to three channels
        
    
        if self.transform:
            image=self.transform(image)
        return image
    
    #self.transform,contains a Compose object.
    '''If a transform has been provided (i.e., it's not None), apply it to the image."
    " Otherwise, leave the image unchanged."
    "self.transform = None → skip transformation."
    "self.transform = transforms.Compose(...) → apply all the preprocessing "
    "steps (resize, crop, tensor conversion, normalization, etc.) to the image before returning it.'''

def get_transform(size,crop,final_size):
    transform_list=[]
    if size>0:
        transform_list.append(transforms.Resize(size))  #transform is torchvision lib element
    if crop==True:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize((final_size, final_size)))
    #convert to tensors
    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)

def adaptive_instance_normalisation(content_feat,style_feat):
    #[batch,channel,height,weight]
    size=content_feat.size()
    style_mean,style_std=cal_mean_std(style_feat)
    content_mean, content_std =cal_mean_std(content_feat)
    normalised_content_feat=(content_feat-content_mean.expand(size))/content_std.expand(size)   #expand=it expans the content mean dim list to content original feature list
    return normalised_content_feat * style_std.expand(size) + style_mean.expand(size)
    

def cal_mean_std(feat,eps=1e-5):
    #[batch,channel,height,weight]
    size=feat.size()
    assert(len(size)==4)
    batch_size,channels=size[:2]
    feat_mean=feat.view(batch_size,channels,-1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean,feat_std
    