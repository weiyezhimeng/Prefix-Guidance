import pandas as pd
from torch.utils.data import Dataset, DataLoader

class mydataset(Dataset):
	def __init__(self, file):
		self.all = pd.read_csv(file)
	def __len__(self):
		return self.all.shape[0]
	def __getitem__(self, idx):
		return self.all.iloc[idx, 0], self.all.iloc[idx, 1]

def load_data(file, batch):
    train_dataset = mydataset(file)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)
    return train_loader