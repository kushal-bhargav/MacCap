import torch
import torch.nn as nn
import torch.nn.functional as F
from ClipContextual import clip
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List
from torch import Tensor
import copy
import math
from models.BaseModel import (
    BaseModel,
    MLP,
    TransformerEncoder,
    LayerNorm,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer
)

class ROCO(BaseModel):
    """
    ROCO: Radiology Objects in Context model for medical image captioning.
    This model adapts a similar architecture as MacCap but can be tuned for ROCO-specific nuances.
    """
    def __init__(self, clip_model, llm, tokenizer, args=None, prefix_size: int = 512):
        super(ROCO, self).__init__(clip_model, llm, tokenizer, args)
        
        # Project visual features to token space
        self.align_proj = MLP(prefix_size, prefix_size, self.vocab_dim, 3)
        
        # Number of queries for aligning visual features to text context
        self.num_query = 32
        self.query_fusion = nn.Embedding(self.num_query, prefix_size)
        
        # Use a TransformerDecoder for feature alignment
        decoder_layer = TransformerDecoderLayer(prefix_size, 8)
        decoder_norm = LayerNorm(prefix_size)
        self.aligner = TransformerDecoder(decoder_layer, args.num_decoder_layer, decoder_norm)
        
        # Define additional attributes with safe defaults.
        self.infer_patch_weight = getattr(args, 'infer_patch_weight', 1.0)
        self.train_seq_length = getattr(args, 'train_seq_length', 77)
        self.variance = getattr(args, 'img_noise_variance', 0.0)
        
        # Noise injection parameters from new parser arguments
        self.num_noise = args.num_noise
        self.eval_variance = args.noise_k
        self.sampling_type = args.sampling_type
        
        if self.sampling_type in ['reconstruction', 'reconstruction_rank']:
            self.noise_N_var = args.noise_N_var
        elif self.sampling_type in ['reconstruction_repeat', 
                                    'reconstruction_concat', 
                                    'reconstruction_concat_wo_token_noise']:
            self.noise_N_var = args.noise_N_var
            if len(self.noise_N_var) != 1:
                raise ValueError('unrecognizable args.noise_N_var!')
            self.noise_N_var = self.noise_N_var * args.num_reconstruction
        else:
            # Default assignment if no specific sampling type is set.
            self.noise_N_var = args.noise_N_var

    def generate(self, img, gen_strategy, clip_tokens=None):
        bs = img.shape[0]

        # Get visual representations from CLIP
        cls_features, img_context, img_proj, attn_weight = self.clip_model.visual(img.half(), require_attn=True)
        if clip_tokens is not None:
            text_cls, _, _ = self.clip_model.encode_text(clip_tokens)
            text_cls /= text_cls.norm(dim=-1, keepdim=True)

        cls_features /= cls_features.norm(dim=-1, keepdim=True)
        clip_conx = img_context @ img_proj
        clip_conx /= clip_conx.norm(dim=-1, keepdim=True)
        attn_mask = None

        mixed_patch_feature = []
        # Get attention weights for image patches (assumed shape: bs x 50 x 50)
        cls_weight = attn_weight[:, 0, :]
        top_cls_patch_ids = cls_weight.topk(self.train_seq_length).indices  # using self.train_seq_length

        for idx in range(bs):
            tp_idx = top_cls_patch_ids[idx]
            top_weight = attn_weight[idx, tp_idx].softmax(dim=-1)
            top_features = top_weight @ clip_conx[idx]
            mixed_patch_feature.append(top_features.unsqueeze(0))
        mixed_patch_feature = torch.cat(mixed_patch_feature, dim=0)
        mixed_patch_feature = F.normalize(mixed_patch_feature, dim=-1)

        # Fuse class token features with weighted patch features
        cls_features = cls_features.unsqueeze(1) + mixed_patch_feature * self.infer_patch_weight
        noisy_cls_features = self.noise_injection(cls_features, self.eval_variance)
        noisy_cls_features /= noisy_cls_features.norm(dim=-1, keepdim=True)

        queries = self.query_fusion.weight.unsqueeze(0).repeat(bs, 1, 1)
        inter_hs = self.aligner(
            queries.permute(1, 0, 2),
            noisy_cls_features.permute(1, 0, 2).to(torch.float32),
            pos=None
        )
        # Project aligned features and reshape
        embedding_clip = self.align_proj(inter_hs.permute(1, 0, 2)).view(1, bs, self.num_query, -1)
        embedding_clip = embedding_clip.permute(1, 0, 2, 3).reshape(bs, self.num_query, -1)

        return self.generate_by_strategy(embedding_clip, attn_mask, gen_strategy, cls_features, self.num_query)

    def forward(self, clip_tokens, gpt_tokens):
        bs, token_len = gpt_tokens.shape
        clip_token_len = self.num_query
        device = clip_tokens.device

        with torch.no_grad():
            clip_features, contex_text, text_proj = self.clip_model.encode_text(clip_tokens)
            clip_features /= clip_features.norm(dim=-1, keepdim=True)

        attn_mask = torch.ones(bs, token_len + clip_token_len).to(device)
        for idx, (i, j) in enumerate(zip(clip_tokens, gpt_tokens)):
            valid_llm_len = (j - 1).count_nonzero().item()
            attn_mask[idx][clip_token_len + valid_llm_len:] = 0

        embedding_text = self.llm_vocab(gpt_tokens)
        queries = self.query_fusion.weight.unsqueeze(0).repeat(bs, 1, 1)
        content_feature = self.noise_injection(
            clip_features.unsqueeze(1).repeat(1, self.train_seq_length, 1).to(torch.float32),
            self.variance
        )
        inter_hs = self.aligner(
            queries.permute(1, 0, 2),
            content_feature.permute(1, 0, 2),
            pos=None
        )
        embedding_clip = self.align_proj(inter_hs.permute(1, 0, 2))
        embedding_cat = torch.cat([embedding_clip, embedding_text], dim=1).to(self.llm_dtype)
        label_mask = torch.full((bs, clip_token_len), -100, dtype=torch.int64, device=device)
        labels = torch.cat([label_mask, gpt_tokens], dim=1)

        inputs = {
            'inputs_embeds': embedding_cat,
            'attention_mask': attn_mask,
            'labels': labels,
            'output_hidden_states': True
        }

        out = self.model(**inputs)
        return self.compute_loss(out, clip_token_len, attn_mask, None, gpt_tokens, embedding_clip)

def build_model(args):
    # Load and freeze CLIP model
    clip_model, preprocess = clip.load(args.clip_model)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Initialize the language model and tokenizer
    llm = AutoModelForCausalLM.from_pretrained(args.language_model, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.language_model, use_fast=False)
    if not getattr(args, 'ft_llm', False):
        for p in llm.parameters():
            p.requires_grad = False
    else:
        llm.float()

    model = ROCO(clip_model, llm, tokenizer, args, clip_model.text_projection.shape[0])
    return model
