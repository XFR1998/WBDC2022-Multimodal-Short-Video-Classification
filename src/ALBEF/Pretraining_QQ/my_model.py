# %%writefile qqmodel/qq_uni_model.py
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append("..")
from Pretraining_QQ.data.masklm import MaskLM, MaskVideo, ShuffleVideo
# from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
# from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from category_id_map import CATEGORY_ID_LIST, lv2id_to_lv1id
from xbert import BertConfig, BertOnlyMLMHead
from xbert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertModel
# from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class QQUniModel(nn.Module):
    def __init__(self, args, task=None, init_from_pretrain=True):
        super().__init__()
        uni_bert_cfg = BertConfig.from_json_file('../config_bert.json')
        if init_from_pretrain:
            self.roberta = UniBertForMaskedLM.from_pretrained(args.bert_dir, config=uni_bert_cfg)
        else:
            self.roberta = UniBertForMaskedLM(uni_bert_cfg)


        self.newfc_hidden = torch.nn.Linear(uni_bert_cfg.hidden_size, 256)

        self.task = set(task)
        print('采用训练任务：', self.task)



        if 'mlm' in task:
            self.lm = MaskLM(tokenizer_path=args.bert_dir)
            self.num_class = len(CATEGORY_ID_LIST)
            self.vocab_size = uni_bert_cfg.vocab_size

        if 'mfm' in task:
            self.vm = MaskVideo()
            self.roberta_mvm_lm_header = VisualOnlyMLMHead(uni_bert_cfg)

        if 'itm' in task:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(uni_bert_cfg.hidden_size, 1)


        # 自己的finetune分类任务
        if 'tag' in task:
            self.classifier = nn.Linear(768, len(CATEGORY_ID_LIST))

    def forward(self, inputs, inference=False, kfold_inference=False):
        loss, pred = 0, None
        masked_vm_loss, masked_lm_loss, itm_loss = 0, 0, 0
        sample_task = self.task

        video_feature = inputs["frame_input"]
        video_mask = inputs["frame_mask"]
        text_input_ids = inputs["title_input"]
        text_mask = inputs["title_mask"]

        # perpare for pretrain task [mask and get task label]: {'mlm': mask title token} {'mfm': mask frame} {'itm': shuffle title-video}
        return_mlm = False
        if 'mlm' in sample_task:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            # lm_label = lm_label[:, 1:].to(text_input_ids.device)  # [SEP] 卡 MASK 大师 [SEP]
            lm_label = lm_label.to(text_input_ids.device)
            return_mlm = True

        if 'mfm' in sample_task:
            vm_input = video_feature
            input_feature, video_label = self.vm.torch_mask_frames(video_feature.cpu(), video_mask.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_label = video_label.to(video_feature.device)

        if 'itm' in sample_task:
            input_feature, video_text_match_label = self.sv.torch_shuf_video(video_feature.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_text_match_label = video_text_match_label.to(video_feature.device)

        # concat features

        features, lm_prediction_scores = self.roberta(video_feature, video_mask, text_input_ids, text_mask,
                                                      return_mlm=return_mlm)

        # compute pretrain task loss
        if 'mlm' in sample_task:
            pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
            masked_lm_loss = nn.CrossEntropyLoss()(pred, lm_label.contiguous().view(-1)) / len(sample_task)
            # print('mlm loss: ', masked_lm_loss)
            loss += masked_lm_loss

        if 'mfm' in sample_task:
            vm_output = self.roberta_mvm_lm_header(features[:, 1:video_feature.size()[1] + 1, :])
            masked_vm_loss = self.calculate_mfm_loss(vm_output, vm_input, video_mask, video_label, normalize=False) / 3 / len(sample_task)
            # print('mfm loss: ',masked_vm_loss* 0.06)
            loss += masked_vm_loss

        if 'itm' in sample_task:
            pred = self.newfc_itm(features[:, 0, :])
            itm_loss = nn.BCEWithLogitsLoss()(pred.view(-1), video_text_match_label.view(-1)) / len(sample_task)  # /100
            loss += itm_loss



        if 'tag' in sample_task:
            mean_output = features.mean(1)
            prediction = self.classifier(mean_output)

            if inference:
                if kfold_inference:
                    return prediction, torch.argmax(prediction, dim=1)
                else:
                    return torch.argmax(prediction, dim=1)
            else:
                return self.cal_loss(prediction, inputs['label'])

        return loss, masked_lm_loss, masked_vm_loss, itm_loss

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        # label_lv1 = label.detach().cpu().numpy()
        # for i,data in enumerate(label_lv1):
        #    label_lv1[i] = lv2id_to_lv1id(data)
        # pdb.set_trace()
        # label_lv1 = torch.tensor(label_lv1,dtype=torch.long).to("cuda:0")
        # loss_lv1 = F.cross_entropy(lv1prediction, label_lv1)

        loss = F.cross_entropy(prediction, label)  # + loss_lv1
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label

    # calc mfm loss
    def calculate_mfm_loss(self, video_feature_output, video_feature_input,
                           video_mask, video_labels_index, normalize=False, temp=0.1):
        if normalize:
            video_feature_output = torch.nn.functional.normalize(video_feature_output, p=2, dim=2)
            video_feature_input = torch.nn.functional.normalize(video_feature_input, p=2, dim=2)

        afm_scores_tr = video_feature_output.view(-1, video_feature_output.shape[-1])

        video_tr = video_feature_input.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        if normalize:
            logits_matrix = logits_matrix / temp

        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != -100)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss


def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, 768, bias=False)
        self.bias = nn.Parameter(torch.zeros(768))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class UniBertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
    def forward(self, video_feature, video_mask, text_input_ids, text_mask, gather_index=None, return_mlm=False):
        # encoder_outputs = self.bert(video_feature, video_mask, text_input_ids, text_mask)

        outputs = self.bert(input_ids=text_input_ids,
                            attention_mask=text_mask,
                            return_dict=True,
                            mode='text')

        outputs = self.bert(encoder_embeds=outputs['last_hidden_state'],
                            attention_mask=text_mask,
                            encoder_hidden_states=video_feature,
                            encoder_attention_mask=video_mask,
                            return_dict=True,
                            mode='fusion')

        encoder_outputs = outputs['last_hidden_state']





        if return_mlm:
            # return encoder_outputs, self.cls(encoder_outputs)[:, 1 + video_feature.size()[1]:, :]
            return encoder_outputs, self.cls(encoder_outputs)
        else:
            return encoder_outputs, None


# class UniBert(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.config = config
#
#         self.bert = BertModel(config=config, add_pooling_layer=False)
#
#         self.init_weights()
#
#     def get_input_embeddings(self):
#         return self.embeddings.word_embeddings
#
#     def set_input_embeddings(self, value):
#         self.embeddings.word_embeddings = value
#
#     def _prune_heads(self, heads_to_prune):
#         for layer, heads in heads_to_prune.items():
#             self.encoder.layer[layer].attention.prune_heads(heads)
#
#     # Copied from transformers.models.bert.modeling_bert.BertModel.forward
#     def forward(self, video_feature, video_mask, text_input_ids, text_mask, gather_index=None):
#
#         outputs = self.bert(input_ids=text_input_ids,
#                             attention_mask=text_mask,
#                             return_dict=True,
#                             mode='text')
#
#         outputs = self.bert(encoder_embeds=outputs['last_hidden_state'],
#                             attention_mask=text_mask,
#                             encoder_hidden_states=video_feature,
#                             encoder_attention_mask=video_mask,
#                             return_dict=True,
#                             mode='fusion')
#
#         last_hidden_state = outputs['last_hidden_state']
#
#
#         return last_hidden_state