import torch
import math
import os
import PIL
import time
import lmdb
import six
import logging
import sys
import traceback
import torch.distributed as dist
from multiprocessing import Queue, Process

torch.multiprocessing.set_sharing_strategy('file_system')


def pad_image(image, target_size):
 
    """
    :param image: input image
    :param target_size: a tuple (num,num)
    :return: new image
    """
 
    iw, ih = image.size 
    w, h = target_size  
 
    scale = min(w / iw, h / ih) 
 
    nw = int(iw * scale+0.5)
    nh = int(ih * scale+0.5)

    w += 128
    h += 128
 

    image = image.resize((nw, nh), PIL.Image.BICUBIC) 
    new_image = PIL.Image.new('RGB', (w, h), (0, 0, 0)) 
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2)) 

    return new_image


def process_data(_quene, path, batch_size):
    from lavis.models import load_model_and_preprocess
    _, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=torch.device("cpu"))
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
    txn = env.begin(write=False)
    length = int(txn.get('num-samples'.encode()))
    print("the length of dataset:", length)
    batch_image = []
    batch_text = []
    idx_list = []
    for idx in range(length):
        image_key, label_key = f'image-{idx + 1:09d}', f'label-{idx + 1:09d}'
        imgbuf = txn.get(image_key.encode())  # image
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        image = PIL.Image.open(buf).convert("RGB")
        image = pad_image(image, (224, 224))
        label = str(txn.get(label_key.encode()), 'utf-8').strip()
        batch_image.append(vis_processors["eval"](image).unsqueeze(0))
        batch_text.append(label)
        idx_list.append(idx)
        if len(batch_image) >= batch_size:
            assert len(batch_image) == len(batch_text)
            batch_image_tensor = torch.cat(batch_image, dim=0)
            batch = {'text_input': batch_text, 'image': batch_image_tensor, 'idx_list': idx_list}
            _quene.put(batch)
            batch_text = []
            batch_image = []
            idx_list = []
    if len(batch_image) > 0:
        assert len(batch_image) == len(batch_text)
        batch_image_tensor = torch.cat(batch_image, dim=0)
        batch = {'text_input': batch_text, 'image': batch_image_tensor, 'idx_list': idx_list}
        _quene.put(batch)
    _quene.put(None)
    while True:
        pass


def process_by_model(cuda_idx, _quene_get, _quene_put):
    from lavis.models import load_model_and_preprocess
    logging.info('init cuda:{}'.format(cuda_idx))
    device = torch.device("cuda:{}".format(cuda_idx))
    model, _, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
    logging.info('cuda {} ready'.format(cuda_idx))
    while True:
        batch = _quene_get.get(True)
        if batch is None:
            _quene_put.put(None)
            while True:
                pass
        text = batch['text_input']
        image = batch['image'].to(device)
        idx_list = batch['idx_list']
        batch_size = len(text)
        assert batch_size == image.shape[0]

        with torch.no_grad():
            # What is the text of the picture? 66.92/81.39
            # What is the content of the text?
            # What does the text in the picture say?  66.81/82.39 66.87/82.39(32)
            # What is written on the picture? 68.80/81.78 68.86/81.78(32)
            answer = model.predict_answers(samples={"image": image, "text_input": ['Question: What does the text in the picture say? Short answer:'] * batch_size}, inference_method="generate", max_len=32)
        
        _quene_put.put([idx_list, text, answer])

if __name__ == '__main__':


    path = sys.argv[1]

    queue_data = Queue(maxsize=32)
    queue_result = Queue()


    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][line:%(lineno)d][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    batch_size = 128
    
    data_process = Process(target=process_data, args=(queue_data, path, batch_size))
    data_process.start()
    
    model_process_list = []
    for i in range(1):
        model_process = Process(target=process_by_model, args=(i, queue_data, queue_result))
        model_process.start()
        model_process_list.append(model_process)
        # time.sleep(20)

    save_all = []
    last_time = time.time()
    while True:
        batch_data = queue_result.get(True)
        if batch_data is None:
            break
        for i in range(len(batch_data[0])):
            print('Label: {} Answer: {}'.format(batch_data[1][i], batch_data[2][i]))
            save_all.append([batch_data[1][i], batch_data[2][i]])
    
    right_num = 0.0
    in_num = 0.0
    for label, answer in save_all:
        label = label.lower()
        answer = answer.lower()
        if label == answer or label == answer.split(' ')[0]:
            right_num += 1
        if label in answer.split(' ') or label in answer or label in answer.replace(' ', '').replace('\'', ''):
            in_num += 1
        else:
            print('[error] Label: {} Answer: {}'.format(label, answer))
    print(right_num / len(save_all), right_num, len(save_all))
    print('in', in_num / len(save_all), in_num, len(save_all))