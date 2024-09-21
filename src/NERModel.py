import torch
import torch.nn as nn

class NERModel(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim):
        super(NERModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, sentences):
        embeds = self.embedding(sentences)  
        lstm_out, _ = self.lstm(embeds)  
        attn_weights = torch.tanh(self.attention(lstm_out))  
        attn_weights = torch.softmax(attn_weights, dim=1)  
        attn_applied = attn_weights * lstm_out  
        tag_space = self.hidden2tag(attn_applied)  
        tag_scores = torch.log_softmax(tag_space, dim=2)  
        return tag_scores