import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import random

imagenet_templates_small = [
    'a {} in the style of {} over a simple background',
    'a photo of a {} in the style of {} over a simple background',
    'a rendering of a {} in the style of {} over a simple background',
    'a cropped photo pf a {} in the style of {} over a simple background',
    'the photo of a {} in the style of {} over a simple background',
    'a clean photo of a {} in the style of {} over a simple background',
    'a dark photo of a {} in the style of {} over a simple background',
    'a picture of a {} in the style of {} over a simple background',
    'a cool photo of a {} in the style of {} over a simple background',
    'a close-up photo of a {} in the style of {} over a simple background',
    'a bright photo of a {} in the style of {} over a simple background',
    'a cropped photo of a {} in the style of {} over a simple background',
    'a good photo of a {} in the style of {} over a simple background',
    'a close-up photo of a {} in the style of {} over a simple background',
    'a nice photo of a {} in the style of {} over a simple background',
    'a small photo of a {} in the style of {} over a simple background',
    'a weird photo of a {} in the style of {} over a simple background',
    'a large photo of a {} in the style of {} over a simple background',
]

imagenet_dual_templates_small = [
    'a photo in the style of {} with {}',
    'a rendering in the style of {} with {}',
    'a cropped photo in the style of {} with {}',
    'the photo in the style of {} with {}',
    'a clean photo in the style of {} with {}',
    'a dirty photo in the style of {} with {}',
    'a dark photo in the style of {} with {}',
    'a cool photo in the style of {} with {}',
    'a close-up photo in the style of {} with {}',
    'a bright photo in the style of {} with {}',
    'a cropped photo in the style of {} with {}',
    'a good photo in the style of {} with {}',
    'a photo of one {} in the style of {}',
    'a nice photo in the style of {} with {}',
    'a small photo in the style of {} with {}',
    'a weird photo in the style of {} with {}',
    'a large photo in the style of {} with {}',
]

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=100,
                 interpolation="bicubic",
                 flip_p=0.5,
                 set="train",
                 placeholder_token="*",
                 per_image_tokens=False,
                 center_crop=False,
                 ):

        self.data_root = data_root

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images 

        self.placeholder_token = placeholder_token

        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i % self.num_images]
        image = Image.open(image_path)
        class_name = os.path.basename(image_path).split('_')[0]

        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self.per_image_tokens and np.random.uniform() < 0.25:
            text = random.choice(imagenet_dual_templates_small).format(self.placeholder_token, per_img_token_list[i % self.num_images])
        else:
            text = random.choice(imagenet_templates_small).format(class_name, self.placeholder_token)
            if class_name == 'airplane':
                text.replace('a airplane', 'an airplane')
            
        example["caption"] = text

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        
        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example