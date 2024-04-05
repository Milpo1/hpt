from torchvision import datasets, transforms
from torch.utils.data import  DataLoader, random_split
class AuthorImagesDataset:
    def __init__(self, root_dir, batch_size:int,DataProcent:float,transform=None):
        self.BATCH_SIZE=batch_size
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.dataset = datasets.ImageFolder(root=self.root_dir, transform=self.transform)

        subset_length = int(len(self.dataset) * DataProcent)
        rest_length = len(self.dataset) - subset_length
        self.subset_data, _ = random_split(self.dataset, [subset_length, rest_length])


        test_length = int(len(self.subset_data) * 0.2)
        train_length = len(self.subset_data) - test_length

        # Split the dataset
        self.train_data, self.test_data = random_split(self.subset_data, [train_length, test_length])
        
        self.into_data_loaders()

    def into_data_loaders(self):


        self.train_dataloader= DataLoader(dataset=self.train_data, # use custom created train Dataset
                                     batch_size=self.BATCH_SIZE, # how many samples per batch?
                                    num_workers=4, # how many subprocesses to use for data loading? (higher = more)
                                    # pin_memory=True,
                                     shuffle=True) # shuffle the data?

        self.test_dataloader = DataLoader(dataset=self.test_data, # use custom created test Dataset
                                    batch_size=self.BATCH_SIZE, 
                                    num_workers=4, 
                                    shuffle=False) # don't usually need to shuffle testing data
        
        
        # word,label=next(iter(self.test_dataloader))
        # print(f"shape of dataloader {word.shape} \n and label {label}")

        
    
    def __len__(self):
        return len(self.dataset) 
            


# if __name__=='__main__':
#     t=RawData()
#     t.save_words_to_file(8)
#     s=AuthorImagesDataset(r'Data/Words',transform=transforms.Compose([
#             transforms.Resize(size=(64,64)),
#             transforms.TrivialAugmentWide(num_magnitude_bins=31),
#             transforms.ToTensor()
#         ]))
    

def create_dataloaders(
        DatasetDir:str,
        transform: transforms.Compose, 
        batch_size: int,
        DataProcent:float
):
    # t=RawData()
    # t.save_words_to_file(8)
    s=AuthorImagesDataset(DatasetDir,batch_size,DataProcent,transform)
    
    return s.train_dataloader,s.test_dataloader,s.dataset.classes