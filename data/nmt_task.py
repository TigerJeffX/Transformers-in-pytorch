# encoding=utf8
import torch
import spacy
import torchtext
import os

from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler, SequentialSampler

abs_dir_path = os.path.dirname(os.path.abspath(__file__))

SOS, EOS, PAD = '<sos>', '<eos>', '<pad>'
MIN_FREQ = 1
MAX_LEN = 60

def get_Multi30K_data_loader(args, device_id):

    def to_map_style_dataset(iter_data):

        class MapStyleDataset(torch.utils.data.Dataset):

            def __init__(self, iter_data):
                self._data = list(iter_data)

            def __len__(self):
                return len(self._data)

            def __getitem__(self, idx):
                return self._data[idx]

        return MapStyleDataset(iter_data)

    # 1. Tokenizer & Field
    spacy_en = spacy.load('en_core_web_sm')
    spacy_de = spacy.load('de_core_news_sm')
    def tokenize_en(text):
        return [t.text for t in spacy_en.tokenizer(text)]
    def tokenize_de(text):
        return [t.text for t in spacy_de.tokenizer(text)]

    SRC_FIELD = torchtext.data.Field(
        tokenize=tokenize_en, eos_token=EOS, pad_token=PAD, lower=True, stop_words=[' '])
    TRG_FIELD = torchtext.data.Field(
        tokenize=tokenize_de, init_token=SOS, eos_token=EOS, pad_token=PAD, lower=True, stop_words=[' '])

    # 2. Dataset & Vocab
    train_iter, val_iter, test_iter = torchtext.datasets.Multi30k.splits(
        root=os.path.join(abs_dir_path, 'multi30k'),
        exts = ('.en', '.de'),
        fields = (SRC_FIELD, TRG_FIELD),
        filter_pred = lambda x:len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN
    )

    # 3. Build Vocab
    if args.share_emb:
        SRC_FIELD.build_vocab(train_iter.src, train_iter.trg, min_freq=MIN_FREQ)
        TRG_FIELD.vocab = SRC_FIELD.vocab
    else:
        SRC_FIELD.build_vocab(train_iter.src, min_freq=MIN_FREQ)
        TRG_FIELD.build_vocab(train_iter.trg, min_freq=MIN_FREQ)

    train_iter_map = to_map_style_dataset(train_iter)
    val_iter_map = to_map_style_dataset(val_iter)
    test_iter_map = (test_iter)

    # 3. Sampler
    train_sampler = DistributedSampler(train_iter_map, num_replicas=args.world_size, rank=device_id)
    val_sampler = SequentialSampler(val_iter_map)
    test_sampler = SequentialSampler(test_iter_map)

    # 4. Loader
    def custom_collate_batch(batch, src_field, trg_field, device_id):
        src = src_field.process([b.src for b in batch]).cuda(device_id)
        trg = trg_field.process([b.trg for b in batch]).cuda(device_id)
        return {'src':src, 'trg':trg}

    def collate_fn(batch):
        return custom_collate_batch(batch, SRC_FIELD, TRG_FIELD, device_id)

    train_loader = DataLoader(
        dataset=train_iter_map,
        sampler=train_sampler,
        shuffle=False,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        dataset=val_iter_map,
        sampler=val_sampler,
        shuffle=False,
        batch_size=args.test_batch_size,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        dataset=test_iter_map,
        sampler=test_sampler,
        shuffle=False,
        batch_size=args.test_batch_size,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, SRC_FIELD.vocab, TRG_FIELD.vocab
