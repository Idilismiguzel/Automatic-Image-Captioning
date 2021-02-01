import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        # Adding a batch normalization with momentum 0.01
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # Define the LSTM
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Define the final fully-connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # self.linear = nn.Linear(hidden_size, vocab_size)
          
    def forward(self, features, captions):
        
        # Create the embedding vectors for each caption using the embedding layer created above
        captions = captions[:, :-1]
        captions = self.embed(captions)
        
        # Concatenate the features, caption inputs and feed to LSTM.
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions), 1)
        lstm_output, _ = self.lstm(inputs)
        
        # Convert LSTM outputs to sentences
        outputs = self.fc(lstm_output)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        sentence =  []
        count = 0
        word_item = None
        
        while (count < max_len and word_item != 1):
            output_lstm, states = self.lstm(inputs, states)
            output = self.fc(output_lstm)
            
            # Get the max probability value
            prob, word = output.max(2)
            
            # Append the word to create the sentences
            word_item = word.item()
            sentence.append(word_item)
            
            inputs = self.embed(word)
            
            count += 1
        
        return sentence