from transformers import T5EncoderModel, T5Config

import torch
from torch import nn
from torch.nn import functional as tf

import math


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.proj = nn.Linear(2 * input_dim, input_dim)
        self.attention = nn.Linear(input_dim, 1)
        self.scaling = math.sqrt(input_dim)
    
    def forward(self, x, attention_mask=None):
        # x shape: [batch_size, seq_len, input_dim]
        x = self.proj(x)
        scores = self.attention(x)
        scores = scores / self.scaling
        
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask[:, :, None].to(dtype=torch.float32)) * torch.finfo(torch.float32).min
            scores = scores + attention_mask
        
        attention_weights = tf.softmax(scores, dim=1)
        # attention_weights shape: [batch_size, seq_len, 1]
        output = torch.sum(attention_weights * x, dim=1)
        # output shape: [batch_size, input_dim]
        return output


class HousePricePredictionModel(nn.Module):
    def __init__(self, pretrained_model_name):
        super(HousePricePredictionModel, self).__init__()
        
        self.base_model = T5EncoderModel.from_pretrained(pretrained_model_name, output_hidden_states=True)
        
        self.pooler = AttentionPooling(768)
        self.regression_fc = nn.Linear(768, 1)
    
    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        x = encoder_outputs.hidden_states[-2:]
        x = torch.cat([x[0], x[1]], dim=-1)
        x = tf.gelu(self.pooler(x, attention_mask))
        logits = self.regression_fc(x)
        return logits


class PretrainedHousePricePredictionModel(nn.Module):
    def __init__(self, config):
        super(PretrainedHousePricePredictionModel, self).__init__()
        self.base_model = T5EncoderModel(config=config)
        
        self.pooler = AttentionPooling(768)
        self.regression_fc = nn.Linear(768, 1)
    
    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        x = encoder_outputs.hidden_states[-2:]
        x = torch.cat([x[0], x[1]], dim=-1)
        x = tf.gelu(self.pooler(x, attention_mask))
        logits = self.regression_fc(x)
        return logits


def load_pretrained_model(config_path, state_dict_path: str, use_gpu: bool = True, train: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    
    config = T5Config.from_pretrained(config_path)
    config.output_hidden_states = True
    
    model = PretrainedHousePricePredictionModel(config).to(device)
    if state_dict_path is not None and isinstance(state_dict_path, str):
        model.load_state_dict(torch.load(state_dict_path, map_location=device))
    
    if not train:
        model.eval()
        
    return model
