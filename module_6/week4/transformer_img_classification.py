from datasets import load_dataset
import re
import string
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import os
from module_6 import util


class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length, device='cpu'):
        super(TokenAndPositionEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.device = device

    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand_as(x).to(self.device)
        return self.embedding(x) + self.position_embedding(positions)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        return self.layernorm2(x + self.dropout2(ffn_output))


class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, max_length, num_layers, num_heads, ff_dim, dropout=0.1, device='cpu'):
        super(TransformerEncoder, self).__init__()
        self.embedding = TokenAndPositionEmbedding(src_vocab_size, embed_dim, max_length, device)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        output = self.embedding(x)
        for layer in self.layers:
            output = layer(output)
        return output

class TransformerEncoderCls(nn.Module):
    def __init__(self, vocab_size, max_length, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1, device='cpu'):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_length=max_length,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            device=device
        )
        self.pooling = nn.AvgPool1d(kernel_size=max_length)
        self.fc1 = nn.Linear(in_features=embed_dim, out_features=20)
        self.fc2 = nn.Linear(in_features=20, out_features=2)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.encoder(x)
        output = self.pooling(output.permute(0, 2, 1)).squeeze()
        output = self.dropout(output)
        output = self.fc1(output)
        output = self.dropout(output)
        output = self.fc2(output)
        return output


def collate_batch(batch):
    seq_length = 100
    # create inputs, offsets, labels for batch
    sentences, labels = list(zip(*batch))
    encoded_sentences = [
        sentence+([0] * (seq_length-len(sentence))) if len(sentence) < seq_length else sentence[:seq_length]
        for sentence in sentences
    ]

    encoded_sentences = torch.tensor(encoded_sentences, dtype=torch.int64)
    labels = torch.tensor(labels)

    return encoded_sentences, labels


def yield_tokens(sentences, tokenizer):
    for sentence in sentences:
        yield tokenizer(sentence)


def prepare_dataset(df, vocabulary, tokenizer):
    # create iterator for dataset: (sentence, label)
    for row in df:
        sentence = row['preprocessed_sentence']
        encoded_sentence = vocabulary(tokenizer(sentence))
        label = row['label']
        yield encoded_sentence, label


def preprocess_text(text):
    # remove URLs https://www.
    url_pattern = re.compile(r'https?://\s+\wwww\.\s+')
    text = url_pattern.sub(r" ", text)

    # remove HTML Tags: <>
    html_pattern = re.compile(r'<[^<>]+>')
    text = html_pattern.sub(" ", text)

    # remove puncs and digits
    replace_chars = list(string.punctuation + string.digits)
    for char in replace_chars:
        text = text.replace(char, " ")

    # remove emoji
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U0001F1F2-\U0001F1F4"  # Macau flag
                               u"\U0001F1E6-\U0001F1FF"  # flags
                               u"\U0001F600-\U0001F64F"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U0001F1F2"
                               u"\U0001F1F4"
                               u"\U0001F620"
                               u"\u200d"
                               u"\u2640-\u2642"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r" ", text)

    # normalize whitespace
    text = " ".join(text.split())

    # lowercasing
    text = text.lower()
    return text


def run():
    ds = load_dataset('thainq107/ntc-scv')
    tokenizer = get_tokenizer("basic_english")
    vocab_size = 10000
    vocabulary = build_vocab_from_iterator(
        yield_tokens(ds['train']['preprocessed_sentence'], tokenizer),
        max_tokens=vocab_size,
        specials=["<pad>", "<unk>"]
    )
    vocabulary.set_default_index(vocabulary["<unk>"])
    train_dataset = prepare_dataset(ds['train'], vocabulary, tokenizer)
    train_dataset = to_map_style_dataset(train_dataset)

    valid_dataset = prepare_dataset(ds['valid'], vocabulary, tokenizer)
    valid_dataset = to_map_style_dataset(valid_dataset)

    test_dataset = prepare_dataset(ds['test'], vocabulary, tokenizer)
    test_dataset = to_map_style_dataset(test_dataset)

    batch_size = 128

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0
    )

    _ = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab_size = 10000
    max_length = 100
    embed_dim = 200
    num_layers = 2
    num_heads = 4
    ff_dim = 128
    dropout = 0.1

    model = TransformerEncoderCls(
        vocab_size, max_length, num_layers, embed_dim, num_heads, ff_dim, dropout, device
    )
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=0)

    num_epochs = 50
    save_model = './model'
    os.makedirs(save_model, exist_ok=True)
    model_name = 'model'

    model, _ = util.train(
        model, model_name, save_model, optimizer, criterion, train_dataloader, valid_dataloader, num_epochs, device
    )
