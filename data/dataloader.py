from torch.utils.data import Dataset

class TextOnlyDataset(Dataset):
    def __init__(self, examples, key="passage"):
        self.examples = examples
        self.key = key

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx][self.key]