import os
import torch
import numpy as np
import cv2
from torchvision.transforms.functional import to_pil_image
from matplotlib import pyplot as plt

# 确保这些变量被定义和设置
# image_path = 'testing_scripts/examples/FFHQ_256_sample_384000_8.jpg'  # 修改为你的图片路径
# prompt = 'left eye'  # 你的提示文本
# filename = 'FFHQ_256_sample_384000_8_{}.jpg'.format(prompt)  # 图片的文件名
# outdir = ''  # 输出目录

import math
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from einops import repeat
from ldm_seg.util import instantiate_from_config


# # 路径变量
# CONFIG_FILE = '/public/hezhenliang/users/gaoge/VIPLFaceMDT/LD-ZNet/configs/ldznet/phrasecut.yaml'
# CKPT_FILE = "/public/hezhenliang/users/gaoge/VIPLFaceMDT/LD-ZNet/pretrained_model/global_step_145999_e_000015.ckpt"
# # CKPT_FILE = "/public/hezhenliang/users/gaoge/VIPLFaceMDT/LD-ZNet/pretrained_model/sd-v1-4-full-ema.ckpt"
# # IMAGE_PATH = '/public/hezhenliang/users/gaoge/VIPLFaceMDT/LD-ZNet/testing_scripts/examples/FFHQ_256_sample_384000_8.jpg'
# IMAGE_PATH = '/public/hezhenliang/users/gaoge/VIPLFaceMDT/LD-ZNet/testing_scripts/examples/00007.png'
# OUTDIR = "/public/hezhenliang/users/gaoge/VIPLFaceMDT/LD-ZNet/testing_scripts/examples/output/"
#
# # 其他变量
# SEED = 42
# PROMPT = 'left_eye'


def parse_args():
    parser = argparse.ArgumentParser(description="Generate masks for given images.")
    parser.add_argument("--config_file", default='/public/hezhenliang/users/gaoge/VIPLFaceMDT/LD-ZNet/configs/ldznet/phrasecut.yaml', type=str, help="Path to the configuration file.")
    parser.add_argument("--ckpt_file", default="/public/hezhenliang/users/gaoge/VIPLFaceMDT/LD-ZNet/pretrained_model/global_step_145999_e_000015.ckpt", type=str, help="Path to the checkpoint file.")
    parser.add_argument("--image_path", default='/public/hezhenliang/users/gaoge/VIPLFaceMDT/LD-ZNet/testing_scripts/examples/FFHQ_256_sample_384000_8.jpg', type=str, help="Path to the input image.")
    parser.add_argument("--outdir", default="/public/hezhenliang/users/gaoge/VIPLFaceMDT/LD-ZNet/testing_scripts/examples/output/", type=str, help="Output directory for the generated masks.")
    parser.add_argument("--seed", default=42, type=int, help="Seed for random number generators.")
    parser.add_argument("--prompt", default='left_eye', type=str, help="Prompt for the generation task.")
    return parser.parse_args()


def load_model_from_config(config, ckpt, verbose=False):
    """从配置文件和检查点中加载模型"""
    model = instantiate_from_config(config.model)
    ldm_weights = torch.load(config.model.params.sd_features_stage_config.params.path, map_location="cpu")["state_dict"]
    ldm_weights_keys_updated = {}
    for k, v in ldm_weights.items():
        if '.diffusion_model' in k:
            k = 'sd_features_stage_model.model.' + k
        ldm_weights_keys_updated[k] = v
    # model.load_state_dict(ldm_weights_keys_updated, strict=False)
    # pl_sd = torch.load(ckpt, map_location="cpu")
    # sd = pl_sd["state_dict"]
    # model.load_state_dict(sd, strict=False)
    # model.cuda()
    # model.eval()
    # return model
    m, u = model.load_state_dict(ldm_weights_keys_updated,
                                 strict=False)  # loading the both stages of LDM along with the clip text encoder

    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    m, u = model.load_state_dict(sd, strict=False)  # loading just the segmentation model
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # num_trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    # print("Num of trainable parameters: "+str(num_trainable_params))

    model.cuda()
    model.eval()
    return model

def load_img(path):
    """加载并预处理图像"""
    image = Image.open(path).convert("RGB")
    w, h = image.size
    aspect_ratio = float(h) / w
    new_w = int(math.sqrt(384 * 384 / aspect_ratio))
    new_h = int(aspect_ratio * new_w)
    new_w, new_h = map(lambda x: x - x % 64, (new_w, new_h))
    image = image.resize((new_w, new_h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.

def setup_directories(outdir):
    """设置输出目录"""
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "Visualizations"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "Masks"), exist_ok=True)

def main():

    args = parse_args()

    seed_everything(args.seed)

    config = OmegaConf.load(args.config_file)
    model = load_model_from_config(config, args.ckpt_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    setup_directories(args.outdir)

    filename = os.path.basename(args.image_path)

    assert os.path.isfile(args.image_path)
    init_image = load_img(args.image_path).to(device)

    prompt = args.prompt

    # seed_everything(SEED)
    #
    # config = OmegaConf.load(CONFIG_FILE)
    # model = load_model_from_config(config, CKPT_FILE)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    #
    # setup_directories(OUTDIR)
    #
    # filename = os.path.basename(IMAGE_PATH)
    #
    # assert os.path.isfile(IMAGE_PATH)
    # init_image = load_img(IMAGE_PATH).to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=1)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
    init_image_arr = (255*(1+init_image[0])/2).cpu().permute(1,2,0).numpy().astype(np.uint8)[:,:,::-1]
    t = torch.randint(0, 1, (init_image.shape[0],)).cuda().long()

    with torch.no_grad():
        with model.ema_scope():
            prompts = [prompt]
            prompts = model.sample_prompts(prompts)
            c = model.get_learned_conditioning(prompts)
            sd_features = model.encode_sd_features_stage(init_latent, c)
            pred = model.apply_model(init_latent, t, c, sd_features)
            pred = torch.sigmoid(pred)
            pred = pred.cpu().permute(0, 2, 3, 1).numpy().squeeze()
            # output_filename = filename.replace('.jpg', '_' + prompt + '.png')
            output_filename = filename.replace('.', '_' + prompt + '.')
            print("++++++++++++++++++++++++++", output_filename)
            output_filename = output_filename.replace('.jpg','.png')

            heatmap = cv2.applyColorMap((255*pred).astype(np.uint8), cv2.COLORMAP_JET)
            attention_overlay = cv2.addWeighted(heatmap, 0.5, init_image_arr, 0.5, 0)
            cv2.imwrite(os.path.join(args.outdir, output_filename), attention_overlay)

    plt.imshow(attention_overlay[:, :, ::-1])
    plt.show()

if __name__ == '__main__':
    main()

