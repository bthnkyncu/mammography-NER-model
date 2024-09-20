import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class NERDataset(Dataset):
    def __init__(self, bio_data, word_to_ix, tag_to_ix):
        self.data = bio_data
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        report, tags = self.data[idx]
        word_idxs = [self.word_to_ix[word] for word in report.split()]
        tag_idxs = [self.tag_to_ix[tag] for tag in tags]
        return torch.tensor(word_idxs, dtype=torch.long), torch.tensor(tag_idxs, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        sentences, tags = zip(*batch)
        sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
        tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)
        return sentences_padded, tags_paddedS