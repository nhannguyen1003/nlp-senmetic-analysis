import torch
import torch.nn as nn
import numpy as np

from .module import IntentClassifier
from gensim.models.keyedvectors import KeyedVectors
from gensim.models import FastText

class ModelCNN_LSTM(nn.Module):
    def __init__(self, args, tokenizer, intent_label_lst):
        super(ModelCNN_LSTM, self).__init__()
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self._initialize_embedding(tokenizer)
        
        # List of CNNs with kernel sizes 3, 4, 5
        self.list_cnn = nn.ModuleList([
            nn.Conv1d(args.embedding_dim, 256, kernel_size, padding='same') for kernel_size in [3, 4, 5]
        ])
        
        cnn_output_size = 256 * len(self.list_cnn)
        self.lstm = nn.LSTM(cnn_output_size, args.hidden_size, batch_first=True)
        
        self.intent_classifier = IntentClassifier(args.hidden_size, self.num_intent_labels, args.dropout_rate)

    def _initialize_embedding(self, tokenizer):
        try:
            word_vectors = KeyedVectors.load("/Users/roy/Documents/nlp/emotion_classification-main/CNN_LSTM/model/vi_cbow/vi-model-CBOW-400.bin")
            word_vectors = word_vectors.wv
        except Exception as e:
            print("Error loading FastText model:", e)
            raise
        
        embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, self.args.embedding_dim))
        for word, i in tokenizer.word_index.items():
            try:
                embedding_matrix[i] = word_vectors.get_vector(word)
            except KeyError:
                embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), self.args.embedding_dim)

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)

    def forward(self, input_ids, intent_label_ids):
        x = self.embedding(input_ids)
        x = torch.transpose(x, 1, 2)
        cnn_outputs = [torch.max(conv(x), dim=2).values for conv in self.list_cnn]
        cnn_output = torch.cat(cnn_outputs, dim=1)
        cnn_output = cnn_output.unsqueeze(1)
        _, (hidden, _) = self.lstm(cnn_output)
        
        intent_logits = self.intent_classifier(hidden.squeeze(0))

        total_loss = 0

        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(
                    intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
                )
            total_loss += intent_loss


        outputs = (intent_logits, hidden.squeeze(0))

        outputs = (total_loss,) + outputs

        return outputs 