import os
import os.path
import json  # optionally used if you want to save & load tokenized cache
import random
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from transformers import AutoTokenizer
from PIL import Image
from ClipContextual import clip

# -------------------------------------------------------------------------------
# Helper function for cleaning captions
# -------------------------------------------------------------------------------
def pre_caption(caption, max_words=50):
    """
    Cleans and truncates captions for consistency.
    
    Args:
        caption (str): Raw caption text.
        max_words (int): Maximum number of words allowed.
    
    Returns:
        str: Cleaned caption.
    """
    caption = re.sub(r"([.!\"()*#:;~])", " ", caption.lower())
    caption = re.sub(r"\s{2,}", " ", caption).strip()
    words = caption.split()[:max_words]
    return " ".join(words)

# ------------------------------------------------------------------------------
# Custom collate function that extracts IDs (if needed)
# ------------------------------------------------------------------------------
def id_collate(batch):
    new_batch = []
    ids = []
    for _batch in batch:
        new_batch.append(_batch[:-1][0])
        ids.append(_batch[-1])
    return default_collate(new_batch), ids

# ------------------------------------------------------------------------------
# ROCOClipDataset: For training tokenization; returns (clip_tokens, llm_tokens)
# ------------------------------------------------------------------------------
class ROCOClipDataset(Dataset):
    def __init__(self, data_root: str, split: str, category: str, language_model: str, 
                 data_ids=None, train_max_length=25, max_words=50):
        """
        Args:
            data_root (str): Base ROCO dataset directory.
            split (str): One of "train", "validation", or "test".
            category (str): Either "radiology" or "non-radiology".
            language_model (str): Hugging Face model name.
            data_ids (str): Optional path for a pre-tokenized cache file.
            train_max_length (int): Maximum length for LLM tokens.
            max_words (int): Maximum words allowed in cleaned captions.
        """
        # Set up tokenizers
        self.clip_tokenizer = clip.tokenize
        self.tokenizer_llm = AutoTokenizer.from_pretrained(language_model, use_fast=False)
        self.max_seq_len = train_max_length  # for LLM tokens
        self.clip_visual_token_len = 77      # fixed length for CLIP tokens
        
        self.max_words = max_words
        self.trigger = ""  # optional prompt trigger
        
        # Build the CSV file path based on the split and category.
        if split == "train":
            csv_file = os.path.join(data_root, "all_data", split, "radiologytraindata.csv")
        elif split == "validation":
            csv_file = os.path.join(data_root, "all_data", split, "radiologyvaldata.csv")
        elif split == "test":
            csv_file = os.path.join(data_root, "all_data", split, "radiologytestdata.csv")
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Load data using pandas.
        df = pd.read_csv(csv_file)
        if "name" in df.columns and "caption" in df.columns:
            # Create a list of tuples: (image_filename, caption)
            self.annotations = list(zip(df["name"], df["caption"]))
        else:
            raise KeyError("CSV file must contain 'name' and 'caption' columns.")

        # Clean and store only captions.
        self.captions = [pre_caption(cap, self.max_words) for _, cap in self.annotations]
        # Shuffle the captions (and corresponding annotations)
        random.shuffle(self.annotations)
        
        # Prepare a cache for tokenized ids.
        self.cache_ids = [None for _ in range(len(self.annotations))]
        if data_ids and os.path.exists(data_ids):
            print('\nLoading tokenized ids\n')
            with open(data_ids, 'r') as f:
                self.cache_ids = json.load(f)
        
        # If the dataset is used for VQA, modify text accordingly.
        self.is_vqa = "vqa" in csv_file.lower()
        print(f"Total {len(self.captions)} training texts from ROCO.")

    def __len__(self) -> int:
        return len(self.captions)

    def pad_tokens(self, tokens: torch.Tensor, max_len: int, padding_idx=0):
        padding = max_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) + padding_idx))
        elif padding < 0:
            tokens = tokens[:max_len]
        return tokens

    def __getitem__(self, item: int):
        # For training tokenization we use only the caption.
        caption = self.captions[item]
        if self.cache_ids[item] is not None:
            clip_tokens, llm_tokens = self.cache_ids[item]
            clip_tokens = torch.tensor(clip_tokens, dtype=torch.int64)
            llm_tokens = torch.tensor(llm_tokens, dtype=torch.int64)
        else:
            # For VQA, restrict caption to the first sentence.
            if self.is_vqa:
                caption = caption.split('.')[0] + '.'
            try:
                # Tokenize using CLIP tokenizer.
                clip_output = self.clip_tokenizer(caption)
            except Exception as e:
                caption = 'A photo of apple.'
                clip_output = self.clip_tokenizer(caption)
            clip_tokens = self.pad_tokens(
                clip_output.to(torch.int64).squeeze(0),
                self.clip_visual_token_len,
                padding_idx=0
            )
            # Tokenize using the language model.
            llm_output = self.tokenizer_llm.encode(
                self.trigger + caption + '</s>',
                return_tensors="pt"
            ).to(torch.int64).squeeze(0)
            llm_tokens = self.pad_tokens(llm_output, self.max_seq_len, padding_idx=1)
            self.cache_ids[item] = [clip_tokens.tolist(), llm_tokens.tolist()]
        
        return clip_tokens, llm_tokens

# ------------------------------------------------------------------------------
# ROCOCaptionDataset: For evaluation; returns (image_instance, image_name)
# ------------------------------------------------------------------------------
class ROCOCaptionDataset(Dataset):
    def __init__(self, args, split='test', max_words=50):
        """
        Args:
            args: An object with attributes:
                - test_image_prefix_path or val_image_prefix_path (depending on the split)
                - test_path or val_path: path to the CSV file containing annotations.
                - clip_model: name or path to the CLIP model.
            split (str): 'test' or 'val'.
            max_words (int): Maximum words allowed in cleaned captions.
        """
        if split == 'test':
            # The image_prefix_path should point to the folder with test images.
            self.image_path = args.test_image_prefix_path
            csv_file = args.test_path
        elif split == 'val':
            self.image_path = args.val_image_prefix_path
            csv_file = args.val_path
        else:
            raise NotImplementedError
        
        # Load the CSV file.
        df = pd.read_csv(csv_file)
        if "name" in df.columns and "caption" in df.columns:
            # Create a list of dictionaries.
            self.annotations = [{"image_name": name, "caption": cap} for name, cap in zip(df["name"], df["caption"])]
        else:
            raise KeyError("CSV file must contain 'name' and 'caption' columns.")
        
        self.captions = [pre_caption(ann["caption"], max_words) for ann in self.annotations]
        
        # Load CLIP model to get preprocessing function.
        _, self.preprocess = clip.load(args.clip_model)
        print(f"Total {len(self.annotations)} {split} caption pair(s) from ROCO.")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, item: int):
        ann = self.annotations[item]
        image_full_path = os.path.join(self.image_path, ann['image_name'])
        image_instance = Image.open(image_full_path).convert("RGB")
        image_instance = self.preprocess(image_instance)
        return image_instance, ann['image_name']
