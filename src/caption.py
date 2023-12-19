import cv2
import numpy as np
import torch
import argparse

import av
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

def captioning(video_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load pretrained processor, tokenizer, and model
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

    # load video
    container = av.open(video_path)

    # extract evenly spaced frames from video
    seg_len = container.streams.video[0].frames
    clip_len = model.config.encoder.num_frames
    indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))

    # generate caption
    gen_kwargs = {
        "min_length": 10, 
        "max_length": 20, 
        "num_beams": 8,
    }
    pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
    tokens = model.generate(pixel_values, **gen_kwargs)
    caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
    print(caption) 
    return caption

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()
    caption = captioning(args.video_path)
    file1 = open(args.output_path,"a")
    file1.write(caption)
    file1.close()


if __name__ == '__main__':
    main()

