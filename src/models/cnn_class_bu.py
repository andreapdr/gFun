import torch.nn as nn
from torch.nn import functional as F
import torch

class CNN_pdr(nn.Module):

    def __init__(self, output_size, out_channels, compositional_dim, vocab_size, emb_dim, embeddings=None, drop_embedding_range=None,
                 drop_embedding_prop=0, drop_prob=0.5):
        super(CNN_pdr, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.embeddings = torch.FloatTensor(embeddings)
        self.embedding_layer = nn.Embedding(vocab_size, emb_dim, _weight=self.embeddings)
        self.kernel_heights = kernel_heights=[3,5,7]
        self.stride = 1
        self.padding = 0
        self.drop_embedding_range = drop_embedding_range
        self.drop_embedding_prop = drop_embedding_prop
        assert 0 <= drop_embedding_prop <= 1, 'drop_embedding_prop: wrong range'
        self.nC = 73

        self.conv1 = nn.Conv2d(1, compositional_dim, (self.kernel_heights[0], self.emb_dim), self.stride, self.padding)
        self.dropout = nn.Dropout(drop_prob)
        self.label = nn.Linear(len(kernel_heights) * out_channels, output_size)
        self.fC = nn.Linear(compositional_dim + self.nC, self.nC)


    def forward(self, x, svm_output):
        x = torch.LongTensor(x)
        svm_output = torch.FloatTensor(svm_output)
        x = self.embedding_layer(x)
        x = self.conv1(x.unsqueeze(1))
        x = F.relu(x.squeeze(3))
        x = F.max_pool1d(x, x.size()[2]).squeeze(2)
        x = torch.cat((x, svm_output), 1)
        x = F.sigmoid(self.fC(x))
        return x    #.detach().numpy()

        # logits = self.label(x)
        # return logits


