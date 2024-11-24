'''
 * The Tag2Text Model
 * Written by Xinyu Huang
'''
import numpy as np
import json
import torch
import warnings

from torch import nn
from .bert import BertConfig, BertModel, BertLMHeadModel
from .swin_transformer import SwinTransformer

from .utils import *

warnings.filterwarnings("ignore")


class Tag2Text(nn.Module):

    def __init__(self,
                 med_config=f'{CONFIG_PATH}/configs/med_config.json',
                 image_size=384,
                 text_encoder_type='bert-base-uncased',
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 prompt='a picture of ',
                 threshold=0.68,
                 delete_tag_index=[127,2961, 3351, 3265, 3338, 3355, 3359],
                 tag_list=f'{CONFIG_PATH}/data/tag2text_ori_tag_list.txt',
                 stage='eval'):
        r""" Tag2Text inference module, both captioning and tagging are included.
        Tag2Text is an efficient and controllable vision-language pre-training framework.
        Described in the paper "Tag2Text: Guiding Vision-Language Model via Image Tagging" https://arxiv.org/abs/2303.05657

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

        else:
            self.visual_encoder, vision_width = create_vit(
                vit, image_size, vit_grad_ckpt, vit_ckpt_layer)

        # create tokenzier
        self.tokenizer = init_tokenizer(text_encoder_type)

        # Tag2Text employ encoder-decoder architecture for image-tag-text generation: image-tag interaction encoder and image-tag-text decoder
        # create image-tag interaction encoder
        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.tag_encoder = BertModel(config=encoder_config,
                                     add_pooling_layer=False)

        # create image-tag-text decoder
        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config)

        # delete some tags that may disturb captioning
        # 127: "quarter"; 2961: "back"; 3351: "two"; 3265: "three"; 3338: "four"; 3355: "five"; 3359: "one"
        self.delete_tag_index = delete_tag_index
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids) - 1

        # load tag list
        self.tag_list = self.load_tag_list(tag_list)

        # create image-tag recognition decoder
        self.threshold = threshold
        self.num_class = len(self.tag_list)
        q2l_config = BertConfig.from_json_file(f'{CONFIG_PATH}/configs/q2l_config.json')
        q2l_config.encoder_width = vision_width
        self.tagging_head = BertModel(config=q2l_config,
                                      add_pooling_layer=False)
        self.tagging_head.resize_token_embeddings(len(self.tokenizer))
        self.label_embed = nn.Embedding(self.num_class, q2l_config.hidden_size)
        self.fc = GroupWiseLinear(self.num_class,
                                  q2l_config.hidden_size,
                                  bias=True)
        self.del_selfattention()

        self.tagging_loss_function = AsymmetricLoss(gamma_neg=7,
                                                    gamma_pos=0,
                                                    clip=0.05)

        # share weights of the lowest 2-layer of "image-tag interaction encoder" with the "image-tag recogntion decoder"
        tie_encoder_decoder_weights(self.tag_encoder, self.tagging_head, '',
                                    ' ')

        # adjust thresholds for some tags
        # default threshold: 0.68
        # 2701: "person"; 2828: "man"; 1167: "woman"; 
        tag_thrshold = {2701:0.7, 2828: 0.7, 1167: 0.7}
        self.class_threshold = torch.ones(self.num_class) * self.threshold
        for key,value in tag_thrshold.items():
            self.class_threshold[key] = value

    def load_tag_list(self, tag_list_file):
        with open(tag_list_file, 'r') as f:
            tag_list = f.read().splitlines()
        tag_list = np.array(tag_list)
        return tag_list

    # delete self-attention layer of image-tag recognition decoder to reduce computation, follower Query2Label
    def del_selfattention(self):
        del self.tagging_head.embeddings
        for layer in self.tagging_head.encoder.layer:
            del layer.attention
    

    def forward(self, image, caption, tag):
        """
        call function as forward

        Args:
            image: type: torch.Tensor  shape: batch_size * 3 * 384 * 384
            caption: type: list[string]  len: batch_size
            tag: type: torch.Tensor   shape: batch * class_num (e.g. 3429)   value: positive sample is 1.0, negative sample is 0.0

        Returns:
            loss: type: torch.Tensor
        """

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        ##================= Image Tagging ================##
        bs = image_embeds.shape[0]
        label_embed = self.label_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        tagging_embed = self.tagging_head(
            encoder_embeds=label_embed,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode='tagging',
        )

        logits = self.fc(tagging_embed[0])

        loss_tag = self.tagging_loss_function(logits, tag)

        ##================= Image-Tag-Text Generation ================##
        tag = tag.cpu().numpy()
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

        return loss_t2t, loss_tag


    def generate(self,
                 image,
                 sample=False,
                 num_beams=3,
                 max_length=30,
                 min_length=10,
                 top_p=0.9,
                 repetition_penalty=1.0,
                 tag_input=None,
                 return_tag_predict=False):

        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

        # if not user specified tags, recognized image tags using image-tag recogntiion decoder
        if tag_input == None:

            bs = image_embeds.shape[0]
            label_embed = self.label_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            tagging_embed = self.tagging_head(
                encoder_embeds=label_embed,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=False,
                mode='tagging',
            )

            logits = self.fc(tagging_embed[0])

            targets = torch.where(
                torch.sigmoid(logits) > self.class_threshold.to(image.device),
                torch.tensor(1.0).to(image.device),
                torch.zeros(self.num_class).to(image.device))

            tag = targets.cpu().numpy()

            # delete some tags that may disturb captioning
            tag[:, self.delete_tag_index] = 0

            tag_input = []
            for b in range(bs):
                index = np.argwhere(tag[b] == 1)
                token = self.tag_list[index].squeeze(axis=1)
                tag_input.append(' | '.join(token))
                
        tag_output = tag_input

        # beam search for text generation(default)
        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
            tag_input_temp = []
            for tag in tag_input:
                for i in range(num_beams):
                    tag_input_temp.append(tag)
            tag_input = tag_input_temp

        image_atts = torch.ones(image_embeds.size()[:-1],
                                dtype=torch.long).to(image.device)

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

        # prompt trick for better captioning, followed BLIP
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
            image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1]

        if sample:
            # nucleus sampling
            model_kwargs = {
                "encoder_hidden_states": output_tagembedding.last_hidden_state,
                "encoder_attention_mask": None
            }
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.1,
                **model_kwargs)
        else:
            # beam search (default)
            model_kwargs = {
                "encoder_hidden_states": output_tagembedding.last_hidden_state,
                "encoder_attention_mask": None
            }
            outputs = self.text_decoder.generate(
                input_ids=input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                eos_token_id=self.tokenizer.sep_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs)

        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        if return_tag_predict == True:
            return  captions, tag_output
        return captions


# load Tag2Text pretrained model parameters
def tag2text(pretrained='', **kwargs):
    model = Tag2Text(**kwargs)
    if pretrained:
        if kwargs['vit'] == 'swin_b':
            model, msg = load_checkpoint_swinbase(model, pretrained, kwargs)
        else:
            model, msg = load_checkpoint(model, pretrained)
        print('vit:', kwargs['vit'])
#         print('msg', msg)
    return model

