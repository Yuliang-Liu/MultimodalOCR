import argparse
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
from ..process import pad_image, resize_image
from PIL import Image
import torch
class MiniGPT4:
    def __init__(self, args, device='cuda:0') -> None:
        args.cfg_path = args.MiniGPT4_cfg_path
        args.options=None
        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = int(device[-1])
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.chat = Chat(model, vis_processor, device = device)
    def generate(self, image, question, name= 'resize', *kargs):
        chat_state = CONV_VISION.copy()
        num_beams = 1
        temperature = 0.9
        img_list = []
        image = Image.open(image).convert('RGB')
        if name == 'resize':
            image = resize_image(image, (224,224))
        llm_message = self.chat.upload_img(image, chat_state, img_list)
        self.chat.ask(question, chat_state)
        llm_message = self.chat.answer(conv=chat_state,
                                    img_list=img_list,
                                    num_beams=num_beams,
                                    temperature=temperature,
                                    max_new_tokens=128,
                                    max_length=640)[0]
        return llm_message

