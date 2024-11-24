'''
 * The Recognize Anything Model (RAM)
 * Written by Xinyu Huang
'''
import json
import warnings

import numpy as np
import torch
from torch import nn

from .bert import BertConfig, BertLMHeadModel, BertModel
from .swin_transformer import SwinTransformer
from .utils import *
import torch.nn.functional as F

warnings.filterwarnings("ignore")


class RAM(nn.Module):
    def __init__(self,
                 med_config=f'{CONFIG_PATH}/configs/med_config.json',
                 image_size=384,
                 text_encoder_type='bert-base-uncased',
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 prompt='a picture of ',
                 threshold=0.68,
                 delete_tag_index=[],
                 tag_list=f'{CONFIG_PATH}/data/ram_tag_list.txt',
                 tag_list_chinese=f'{CONFIG_PATH}/data/ram_tag_list_chinese.txt',
                 stage='eval'):
        r""" The Recognize Anything Model (RAM) inference module.
        RAM is a strong image tagging model, which can recognize any common category with high accuracy.
        Described in the paper " Recognize Anything: A Strong Image Tagging Model" https://recognize-anything.github.io/
        
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
            threshold (int): tagging threshold
            delete_tag_index (list): delete some tags that may disturb captioning
        """
        super().__init__()

        # create image encoder
        if vit == 'swin_b':
            if image_size == 224:
                vision_config_path = f'{CONFIG_PATH}/configs/swin/config_swinB_224.json'
            elif image_size == 384:
                vision_config_path = f'{CONFIG_PATH}/configs/swin/config_swinB_384.json'
            vision_config = read_json(vision_config_path)
            assert image_size == vision_config['image_res']
            # assert config['patch_size'] == 32
            vision_width = vision_config['vision_width']

            self.visual_encoder = SwinTransformer(
                img_size=vision_config['image_res'],
                patch_size=4,
                in_chans=3,
                embed_dim=vision_config['embed_dim'],
                depths=vision_config['depths'],
                num_heads=vision_config['num_heads'],
                window_size=vision_config['window_size'],
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.0,
                drop_path_rate=0.1,
                ape=False,
                patch_norm=True,
                use_checkpoint=False)
            
            if stage == 'train_from_scratch':
                # download from https://github.com/microsoft/Swin-Transformer
                state_dict = torch.load(vision_config['ckpt'], map_location="cpu")['model']

                for k in list(state_dict.keys()):
                    if 'relative_position_bias_table' in k:
                        dst_num_pos = (2 * vision_config['window_size'] - 1) ** 2
                        state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
                    elif ('relative_position_index' in k) or ('attn_mask' in k):
                        del state_dict[k]

                print("### Load Vision Backbone", vit)
                msg = self.visual_encoder.load_state_dict(state_dict, strict = False)
                print("missing_keys: ", msg.missing_keys)
                print("unexpected_keys: ", msg.unexpected_keys)

        elif vit == 'swin_l':
            if image_size == 224:
                vision_config_path = f'{CONFIG_PATH}/configs/swin/config_swinL_224.json'
            elif image_size == 384:
                vision_config_path = f'{CONFIG_PATH}/configs/swin/config_swinL_384.json'
            vision_config = read_json(vision_config_path)
            assert image_size == vision_config['image_res']
            # assert config['patch_size'] == 32
            vision_width = vision_config['vision_width']

            self.visual_encoder = SwinTransformer(
                img_size=vision_config['image_res'],
                patch_size=4,
                in_chans=3,
                embed_dim=vision_config['embed_dim'],
                depths=vision_config['depths'],
                num_heads=vision_config['num_heads'],
                window_size=vision_config['window_size'],
                mlp_ratio=4.,
                qkv_bias=True,
                drop_rate=0.0,
                drop_path_rate=0.1,
                ape=False,
                patch_norm=True,
                use_checkpoint=False)
            
            if stage == 'train_from_scratch':
                # download from https://github.com/microsoft/Swin-Transformer
                state_dict = torch.load(vision_config['ckpt'], map_location="cpu")['model']

                for k in list(state_dict.keys()):
                    if 'relative_position_bias_table' in k:
                        dst_num_pos = (2 * vision_config['window_size'] - 1) ** 2
                        state_dict[k] = interpolate_relative_pos_embed(state_dict[k], dst_num_pos, param_name=k)
                    elif ('relative_position_index' in k) or ('attn_mask' in k):
                        del state_dict[k]

                print("### Load Vision Backbone", vit)
                msg = self.visual_encoder.load_state_dict(state_dict, strict = False)
                print("missing_keys: ", msg.missing_keys)
                print("unexpected_keys: ", msg.unexpected_keys)

        else:
            self.visual_encoder, vision_width = create_vit(
                vit, image_size, vit_grad_ckpt, vit_ckpt_layer)

        # create tokenzier
        self.tokenizer = init_tokenizer(text_encoder_type)

        # Tag2Text employ encoder-decoder architecture for image-tag-text generation: image-tag interaction encoder and image-tag-text decoder
        # create image-tag interaction encoder
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = 512
        self.tag_encoder = BertModel(config=encoder_config,
                                     add_pooling_layer=False)

        # create image-tag-text decoder
        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

        self.delete_tag_index = delete_tag_index
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        # load tag list
        self.tag_list = self.load_tag_list(tag_list)
        self.tag_list_chinese = self.load_tag_list(tag_list_chinese)

        # create image-tag recognition decoder
        self.threshold = threshold
        self.num_class = len(self.tag_list)
        q2l_config = BertConfig.from_json_file(f'{CONFIG_PATH}/configs/q2l_config.json')
        q2l_config.encoder_width = 512
        self.tagging_head = BertModel(config=q2l_config,
                                      add_pooling_layer=False)
        self.tagging_head.resize_token_embeddings(len(self.tokenizer))

        if stage == 'train_from_scratch':
            self.label_embed = nn.Parameter(torch.load(f'{CONFIG_PATH}/data/frozen_tag_embedding/ram_tag_embedding_class_4585.pth',map_location='cpu').float())
        else:
            # when eval with pretrained RAM model, directly load from ram_swin_large_14m.pth
            self.label_embed = nn.Parameter(torch.zeros(self.num_class, q2l_config.encoder_width))

        if q2l_config.hidden_size != 512:
            self.wordvec_proj = nn.Linear(512, q2l_config.hidden_size)
        else:
            self.wordvec_proj = nn.Identity()

        self.fc = nn.Linear(q2l_config.hidden_size, 1)

        self.del_selfattention()

        self.tagging_loss_function = AsymmetricLoss(gamma_neg=7,
                                                    gamma_pos=0,
                                                    clip=0.05)

        # share weights of the lowest 2-layer of "image-tag interaction encoder" with the "image-tag recogntion decoder"
        tie_encoder_decoder_weights(self.tag_encoder, self.tagging_head, '',
                                    ' ')
        self.image_proj = nn.Linear(vision_width, 512)
        # self.label_embed = nn.Parameter(torch.load(f'{CONFIG_PATH}/data/textual_label_embedding.pth',map_location='cpu').float())

        # adjust thresholds for some tags
        self.class_threshold = torch.ones(self.num_class) * self.threshold
        ram_class_threshold_path = f'{CONFIG_PATH}/data/ram_tag_list_threshold.txt'
        with open(ram_class_threshold_path, 'r', encoding='utf-8') as f:
            ram_class_threshold = [float(s.strip()) for s in f]
        for key,value in enumerate(ram_class_threshold):
            self.class_threshold[key] = value

    def load_tag_list(self, tag_list_file):
        with open(tag_list_file, 'r', encoding="utf-8") as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list

    # delete self-attention layer of image-tag recognition decoder to reduce computation, follower Query2Label
    def del_selfattention(self):
        del self.tagging_head.embeddings
        for layer in self.tagging_head.encoder.layer:
            del layer.attention

    def forward(self, image, caption, image_tag, parse_tag, clip_feature):
        """
        call function as forward

        Args:
            image: type: torch.Tensor  shape: batch_size * 3 * 384 * 384
            caption: type: list[string]  len: batch_size
            tag: type: torch.Tensor   shape: batch * class_num (e.g. 3429)   value: positive sample is 1.0, negative sample is 0.0

        Returns:
            loss: type: torch.Tensor
        """

        label_embed = torch.nn.functional.relu(self.wordvec_proj(self.label_embed))

        image_embeds = self.image_proj(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        ##================= Distillation from CLIP ================##
        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        loss_dis = F.l1_loss(image_cls_embeds, clip_feature)

        ##================= Image Tagging ================##
        bs = image_embeds.shape[0]
        label_embed = label_embed.unsqueeze(0).repeat(bs, 1, 1)

        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )

        logits = self.fc(tagging_embed[0]).squeeze(-1)

        loss_tag = self.tagging_loss_function(logits, image_tag)

        ##================= Image-Tag-Text Generation ================##
        tag = parse_tag.cpu().numpy()
        tag_input = []
        for b in range(bs):
            index = np.argwhere(tag[b] == 1)
            token = self.tag_list[index].squeeze(axis=1)
            tag_input.append(' | '.join(token))
        
        # tokenizer input tags
        tag_input_tokenzier = self.tokenizer(tag_input,
                                             padding='max_length',
                                             truncation=True,
                                             max_length=40,
                                             return_tensors="pt").to(
                                                 image.device)
        encoder_input_ids = tag_input_tokenzier.input_ids
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id

        # put input tag into image-tag interaction encoder to interact with image embeddings
        output_tagembedding = self.tag_encoder(
            encoder_input_ids,
            attention_mask=tag_input_tokenzier.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        text = self.tokenizer(caption,
                              padding='longest',
                              truncation=True,
                              max_length=40,
                                return_tensors="pt").to(
                                    image.device)
        
        decoder_input_ids = text.input_ids
        decoder_input_ids[:,0] = self.tokenizer.bos_token_id

        decoder_targets = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100) 
        decoder_targets[:,:self.prompt_length] = -100
        
        decoder_output = self.text_decoder(decoder_input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = output_tagembedding.last_hidden_state,
                                           encoder_attention_mask = None,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
        
        loss_t2t = decoder_output.loss

        return loss_t2t, loss_tag, loss_dis

    def generate_tag(self,
                 image,
                 threshold=0.68,
                 tag_input=None,
                 ):
            
        label_embed = torch.nn.functional.relu(self.wordvec_proj(self.label_embed))

        image_embeds = self.image_proj(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        # recognized image tags using image-tag recogntiion decoder
        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]
        label_embed = label_embed.unsqueeze(0).repeat(bs, 1, 1)
        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )

        logits = self.fc(tagging_embed[0]).squeeze(-1)

        targets = torch.where(
            torch.sigmoid(logits) > self.class_threshold.to(image.device),
            torch.tensor(1.0).to(image.device),
            torch.zeros(self.num_class).to(image.device))

        tag = targets.cpu().numpy()
        tag[:,self.delete_tag_index] = 0
        tag_output = []
        tag_output_chinese = []
        for b in range(bs):
            index = np.argwhere(tag[b] == 1)
            token = self.tag_list[index].squeeze(axis=1)
            tag_output.append(' | '.join(token))
            token_chinese = self.tag_list_chinese[index].squeeze(axis=1)
            tag_output_chinese.append(' | '.join(token_chinese))


        return tag_output, tag_output_chinese

    def generate_tag_openset(self,
                 image,
                 threshold=0.68,
                 tag_input=None,
                 ):
            
        label_embed = torch.nn.functional.relu(self.wordvec_proj(self.label_embed))

        image_embeds = self.image_proj(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        # recognized image tags using image-tag recogntiion decoder
        image_cls_embeds = image_embeds[:, 0, :]
        image_spatial_embeds = image_embeds[:, 1:, :]

        bs = image_spatial_embeds.shape[0]
        label_embed = label_embed.unsqueeze(0).repeat(bs, 1, 1)
        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )

        logits = self.fc(tagging_embed[0]).squeeze(-1)

        targets = torch.where(
            torch.sigmoid(logits) > self.class_threshold.to(image.device),
            torch.tensor(1.0).to(image.device),
            torch.zeros(self.num_class).to(image.device))

        tag = targets.cpu().numpy()
        tag[:,self.delete_tag_index] = 0
        tag_output = []
        for b in range(bs):
            index = np.argwhere(tag[b] == 1)
            token = self.tag_list[index].squeeze(axis=1)
            tag_output.append(' | '.join(token))

        return tag_output


# load RAM pretrained model parameters
def ram(pretrained='', **kwargs):
    model = RAM(**kwargs)
    if pretrained:
        if kwargs['vit'] == 'swin_b':
            model, msg = load_checkpoint_swinbase(model, pretrained, kwargs)
        elif kwargs['vit'] == 'swin_l':
            model, msg = load_checkpoint_swinlarge(model, pretrained, kwargs)
        else:
            model, msg = load_checkpoint(model, pretrained)
        print('vit:', kwargs['vit'])
#         print('msg', msg)
    return model
