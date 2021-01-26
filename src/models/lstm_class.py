#taken from https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/LSTM.py
from models.helpers import *
from torch.autograd import Variable


class RNNMultilingualClassifier(nn.Module):

    def __init__(self, output_size, hidden_size, lvocab_size, learnable_length, lpretrained=None,
                 drop_embedding_range=None, drop_embedding_prop=0, post_probabilities=True, only_post=False,
                 bert_embeddings=False):

        super(RNNMultilingualClassifier, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.drop_embedding_range = drop_embedding_range
        self.drop_embedding_prop = drop_embedding_prop
        self.post_probabilities = post_probabilities
        self.bert_embeddings = bert_embeddings
        assert 0 <= drop_embedding_prop <= 1, 'drop_embedding_prop: wrong range'

        self.lpretrained_embeddings = nn.ModuleDict()
        self.llearnable_embeddings = nn.ModuleDict()
        self.embedding_length = None
        self.langs = sorted(lvocab_size.keys())
        self.only_post = only_post

        self.n_layers = 1
        self.n_directions = 1

        self.dropout = nn.Dropout(0.6)

        lstm_out = 256
        ff1 = 512
        ff2 = 256

        lpretrained_embeddings = {}
        llearnable_embeddings = {}
        if only_post==False:
            for l in self.langs:
                pretrained = lpretrained[l] if lpretrained else None
                pretrained_embeddings, learnable_embeddings, embedding_length = init_embeddings(
                    pretrained, lvocab_size[l], learnable_length
                )
                lpretrained_embeddings[l] = pretrained_embeddings
                llearnable_embeddings[l] = learnable_embeddings
                self.embedding_length = embedding_length

            # self.lstm = nn.LSTM(self.embedding_length, hidden_size, dropout=0.2 if self.n_layers>1 else 0, num_layers=self.n_layers, bidirectional=(self.n_directions==2))
            self.rnn = nn.GRU(self.embedding_length, hidden_size)
            self.linear0 = nn.Linear(hidden_size * self.n_directions, lstm_out)
            self.lpretrained_embeddings.update(lpretrained_embeddings)
            self.llearnable_embeddings.update(llearnable_embeddings)

        self.linear1 = nn.Linear(lstm_out, ff1)
        self.linear2 = nn.Linear(ff1, ff2)

        if only_post:
            self.label = nn.Linear(output_size, output_size)
        elif post_probabilities and not bert_embeddings:
            self.label = nn.Linear(ff2 + output_size, output_size)
        elif bert_embeddings and not post_probabilities:
            self.label = nn.Linear(ff2 + 768, output_size)
        elif post_probabilities and bert_embeddings:
            self.label = nn.Linear(ff2 + output_size + 768, output_size)
        else:
            self.label = nn.Linear(ff2, output_size)

    def forward(self, input, post, bert_embed, lang):
        if self.only_post:
            doc_embedding = post
        else:
            doc_embedding = self.transform(input, lang)
            if self.post_probabilities:
                doc_embedding = torch.cat([doc_embedding, post], dim=1)
            if self.bert_embeddings:
                doc_embedding = torch.cat([doc_embedding, bert_embed], dim=1)

        logits = self.label(doc_embedding)
        return logits

    def transform(self, input, lang):
        batch_size = input.shape[0]
        input = embed(self, input, lang)
        input = embedding_dropout(input, drop_range=self.drop_embedding_range, p_drop=self.drop_embedding_prop,
                                  training=self.training)
        input = input.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(self.n_layers*self.n_directions, batch_size, self.hidden_size).cuda())
        # c_0 = Variable(torch.zeros(self.n_layers*self.n_directions, batch_size, self.hidden_size).cuda())
        # output, (_, _) = self.lstm(input, (h_0, c_0))
        output, _ = self.rnn(input, h_0)
        output = output[-1, :, :]
        output = F.relu(self.linear0(output))
        output = self.dropout(F.relu(self.linear1(output)))
        output = self.dropout(F.relu(self.linear2(output)))
        return output

    def finetune_pretrained(self):
        for l in self.langs:
            self.lpretrained_embeddings[l].requires_grad = True
            self.lpretrained_embeddings[l].weight.requires_grad = True

    def get_embeddings(self, input, lang):
        batch_size = input.shape[0]
        input = embed(self, input, lang)
        input = embedding_dropout(input, drop_range=self.drop_embedding_range, p_drop=self.drop_embedding_prop,
                                  training=self.training)
        input = input.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size).cuda())
        output, _ = self.rnn(input, h_0)
        output = output[-1, :, :]
        return output.cpu().detach().numpy()

