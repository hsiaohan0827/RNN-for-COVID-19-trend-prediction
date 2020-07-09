import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, rnn_type, input_dim, hidden_dim, layer_dim, dropout):
        super(RNNModel, self).__init__()
        
        self.rnn_type = rnn_type

        # input dimensions
        self.input_dim = input_dim

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        if rnn_type == 'LSTM':
        # batch_first=True (batch_dim, seq_dim, feature_dim)
            self.rnn = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.layer_dim, batch_first=True, dropout=dropout) 
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.layer_dim, batch_first=True, dropout=dropout)
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.layer_dim, batch_first=True, dropout=dropout)


        # classifier
        self.fc = nn.Linear(self.hidden_dim, 1)

    def init_LSTM_hidden(self, batch_size):
        return (torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(torch.device("cuda")),
                torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(torch.device("cuda")))
    def init_RNN_hidden(self, batch_size):
        return torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(torch.device("cuda"))

        
    def forward(self, input):
        if self.rnn_type == 'LSTM':
            hidden = self.init_LSTM_hidden(batch_size=len(input))
        else:
            hidden = self.init_RNN_hidden(batch_size=len(input))
        out, _ = self.rnn(input, hidden)

        # Index hidden state of last time step
        out = out[:, -1, :]
        out = self.fc(out)

        return out