import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel
from cfg import CFG


def init_params(module_lst):
    for module in module_lst:
        for param in module.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
    return


losses_dict = {'smooth-l1': nn.SmoothL1Loss,
               'mse': nn.MSELoss,
               'l1': nn.L1Loss,
               #'huber-loss': nn.HuberLoss,
               }


class Custom_Bert(nn.Module):
    def __init__(self):
        super().__init__()
        config = AutoConfig.from_pretrained(CFG.model_path)
        config.output_hidden_states = True
        config.max_position_embeddings = CFG.max_position_embeddings
        config.attention_probs_dropout_prob = 0
        config.hidden_dropout_prob = 0
        self.num_hidden_layers =  config.num_hidden_layers
        self.backbone = AutoModel.from_pretrained(CFG.model_path, config=config)
        self.backbone.encoder.layer[-1].apply(self.backbone._init_weights)
        self.backbone.encoder.layer[-2].apply(self.backbone._init_weights)
        dim = config.hidden_size
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), # b, s, hidden-size
            nn.Tanh(), # nn.glue()
            nn.Linear(config.hidden_size, 1), #b, s, 1
            nn.Softmax(dim=1) # 归一化
        )
        self.cls = nn.Sequential(
            nn.Linear(dim, CFG.num_labels)
        )
        self._init_weights(self.cls, config)
        self._init_weights(self.attention, config)
        

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask,  token_type_ids=token_type_ids)
        hidden_states = output.last_hidden_state # b, s, hidden_size
        att_weights = self.attention(hidden_states) # b,s,1
        feature =  torch.sum(att_weights * hidden_states, dim=1) # b, hidden-size

        output = self.cls(feature) # b, num_of_labels
        if labels is None:
            return output

        else:
            return (losses_dict[CFG.loss_type]()(output, labels), output)
        
    def _init_weights(self, module, config):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

class Custom_Bert_Simple(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(CFG.model_path)
        config.max_position_embeddings = CFG.max_position_embeddings
        config.num_labels = CFG.num_labels
        config.attention_probs_dropout_prob = 0
        config.hidden_dropout_prob = 0
        self.backbone = AutoModelForSequenceClassification.from_pretrained(CFG.model_path, config=config)

    def forward(self, input_ids, attention_mask, labels=None):
        base_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        output = base_output[0]
        if labels is None:
            return output

        else:
            return (losses_dict[CFG.loss_type]()(output, labels), output)



class Custom_Bert_Mean(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(CFG.model_path)
        config.output_hidden_states = True
        config.max_position_embeddings = CFG.max_position_embeddings
        config.attention_probs_dropout_prob = 0
        config.hidden_dropout_prob = 0
        self.backbone = AutoModel.from_pretrained(CFG.model_path, config=config)
        dim = config.hidden_size
        self.dropout = nn.Dropout(p=0)
        self.cls = nn.Linear(dim, CFG.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        base_output = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask,
                                )

        output = base_output.last_hidden_state
        output = self.cls(self.dropout(torch.mean(output, dim=1)))
        if labels is None:
            return output

        else:
            return (losses_dict[CFG.loss_type]()(output, labels), output)



class Custom_Bert_Mean_Mul_Drop(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(CFG.model_path)
        config.output_hidden_states = True
        config.max_position_embeddings = CFG.max_position_embeddings
        config.attention_probs_dropout_prob = 0
        config.hidden_dropout_prob = 0
        self.backbone = AutoModel.from_pretrained(CFG.model_path, config=config)
        dim = config.hidden_size
        self.dropout = nn.Dropout(p=0)
        self.cls = nn.Linear(dim, CFG.num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        output = 0
        for _ in range(5):
            base_output = self.backbone(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    )
        
            output += base_output.last_hidden_state
        output /= 5
        output = self.cls(self.dropout(torch.mean(output, dim=1)))
        if labels is None:
            return output

        else:
            return (losses_dict[CFG.loss_type]()(output, labels), output)




