import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import requests
from PIL import Image
from io import BytesIO
from fastapi import FastAPI, HTTPException, Body, File, UploadFile
from transformers import StoppingCriteriaList
from fastapi.responses import JSONResponse

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
app = FastAPI()
# def parse_args():
#     parser = argparse.ArgumentParser(description="Demo")
#     parser.add_argument("--cfg-path", default = 'eval_configs/minigpt4_eval.yaml', help="path to configuration file.")
#     parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
#     parser.add_argument(
#         "--options",
#         nargs="+",
#         help="override some settings in the used config, the key-value pair "
#         "in xxx=yyy format will be merged into config file (deprecate), "
#         "change to --cfg-options instead.",
#     )
#     args = parser.parse_args()
#     return args

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('Initializing Chat')
# args = parse_args()
cfg = Config()

model_config = cfg.model_cfg
model_config.device_8bit = 0
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(0))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(0)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

chat = Chat(model, vis_processor, device='cuda:{}'.format(0), stopping_criteria=stopping_criteria)
print('Initialization Finished')

# img_url = 'https://media.istockphoto.com/id/1181241726/photo/young-man-walking-at-crosswalk-on-a-sao-paulos-street.jpg?s=612x612&w=0&k=20&c=BbYxvaMZazg6gppfk1l9Rj5Q25f6hsaA3NbBcmPdGsM='
# my_res = requests.get(img_url)
# my_img = Image.open(BytesIO(my_res.content))

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(BytesIO(content))
    
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(image, chat_state, img_list)
    chat.encode_img(img_list)
    
    user_message = 'Please explain this image as detailed as possible'
    chat.ask(user_message, chat_state)
    chatbot = user_message
    num_beams = 1
    temperature =1
    llm_message = chat.answer(conv=chat_state,
                                  img_list=img_list,
                                  num_beams=num_beams,
                                  temperature=temperature,
                                  max_new_tokens=300,
                                  max_length=2000)[0]
    
    print(llm_message)
    return JSONResponse(content={"text": llm_message}, status_code=200)

