import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from pyvi import ViTokenizer
from model import ModelCNN, ModelLSTM, ModelCNN_LSTM
from seqeval.metrics import f1_score, precision_score, recall_score


MODEL_CLASSES = {
    "cnn": ModelCNN,
    "lstm": ModelLSTM,
    "cnn_lstm": ModelCNN_LSTM
}




def load_tokenizer(args):
    data_train = pd.read_csv(
        "/Users/roy/Documents/nlp/emotion_classification-main/dataset/augment_gpt/vlsp_sentiment_train.csv")
    sentences = data_train.iloc[:, 2].values
    sentences_train = []
    for sentence in sentences:
        sentence = ViTokenizer.tokenize(sentence.lower())
        sentences_train.append(sentence.split())

    tokenizer = Tokenizer(num_words=args.max_vocab_size,
                          lower=True, char_level=False)
    tokenizer.fit_on_texts(sentences_train)

    return tokenizer


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_preds, intent_labels):
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)

    results.update(intent_result)

    return results


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {"intent_acc": acc}


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), "r", encoding="utf-8")]
