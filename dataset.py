import os.path
from itertools import product

from random import choice
import numpy as np
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset
import json
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomHorizontalFlip,
                                    RandomPerspective, RandomRotation, Resize,
                                    ToTensor)
from torchvision.transforms.transforms import RandomResizedCrop

BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def transform_image(split="train", imagenet=False):
    if imagenet:
        # from czsl repo.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform = Compose(
            [
                RandomResizedCrop(n_px),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(
                    mean,
                    std,
                ),
            ]
        )
        return transform

    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        transform = Compose(
            [
                # RandomResizedCrop(n_px, interpolation=BICUBIC),
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                RandomHorizontalFlip(),
                RandomPerspective(),
                RandomRotation(degrees=5),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    return transform

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


class CompositionDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='compositional-split-natural',
            open_world=False,
            imagenet=False,
            same_prim_sample=False
    ):
        self.root = root
        self.phase = phase
        self.split = split
        self.open_world = open_world
        self.same_prim_sample = same_prim_sample

        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + '/images/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        if self.open_world:
            self.pairs = list(product(self.attrs, self.objs))

        self.train_data, self.val_data, self.test_data = self.get_split_info()
        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

        if self.open_world:
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.

            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)

        if self.phase == 'train' and self.same_prim_sample:
            self.same_attr_diff_obj_dict = {pair: list() for pair in self.train_pairs}
            self.same_obj_diff_attr_dict = {pair: list() for pair in self.train_pairs}
            for i_sample, sample in enumerate(self.train_data):
                sample_attr, sample_obj = sample[1], sample[2]
                for pair_key in self.same_attr_diff_obj_dict.keys():
                    if (pair_key[1] == sample_obj) and (pair_key[0] != sample_attr):
                        self.same_obj_diff_attr_dict[pair_key].append(i_sample)
                    elif (pair_key[1] != sample_obj) and (pair_key[0] == sample_attr):
                        self.same_attr_diff_obj_dict[pair_key].append(i_sample)


    def get_split_info(self):
        data = torch.load(self.root + '/metadata_{}.t7'.format(self.split), weights_only=True)
        train_data, val_data, test_data = [], [], []
        for instance in data:
            image, attr, obj, settype = instance['image'], instance[
                'attr'], instance['obj'], instance['set']

            if attr == 'NA' or (attr,
                                obj) not in self.pairs or settype == 'NA':
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if settype == 'train':
                train_data.append(data_i)
            elif settype == 'val':
                val_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                pairs = f.read().strip().split('\n')
                # pairs = [t.split() if not '_' in t else t.split('_') for t in pairs]
                if 'vaw-czsl' in self.root:
                    pairs = [t.split('+') for t in pairs]
                else:
                    pairs = [t.split() for t in pairs]
                # pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs, all_objs = sorted(
            list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
                list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def revise_image_path(self, image):
        if 'mit-states' in self.root and not 'mit-states-star' in self.root:
            pair, img = image.split('/')
            pair = pair.replace('_', ' ')
            image = pair + '/' + img
        return image

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        image = self.revise_image_path(image)
        img = self.loader(image)
        img = self.transform(img)

        if self.phase == 'train':
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]
        else:
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

        if self.phase == 'train' and self.same_prim_sample:
            [same_attr_image, same_attr, diff_obj], same_attr_mask = self.same_A_diff_B(label_A=attr, label_B=obj, phase='attr')
            [same_obj_image, diff_attr, same_obj], same_obj_mask = self.same_A_diff_B(label_A=obj, label_B=attr, phase='obj')
            same_attr_img = self.transform(self.loader(same_attr_image))
            same_obj_img = self.transform(self.loader(same_obj_image))
            data += [same_attr_img, self.attr2idx[same_attr], self.obj2idx[diff_obj], 
                     self.train_pair_to_idx[(same_attr, diff_obj)], same_attr_mask,
                     same_obj_img, self.attr2idx[diff_attr], self.obj2idx[same_obj], 
                     self.train_pair_to_idx[(diff_attr, same_obj)], same_obj_mask]

        return data

    def same_A_diff_B(self, label_A, label_B, phase='attr'):
        if phase=='attr':
            candidate_list = self.same_attr_diff_obj_dict[(label_A, label_B)]
        else:
            candidate_list = self.same_obj_diff_attr_dict[(label_B, label_A)]
        if len(candidate_list) != 0:
            idx = choice(candidate_list)
            mask = 1
        else:
            idx = choice(list(range(len(self.data))))
            mask = 0
        return self.data[idx], mask

    def __len__(self):
        return len(self.data)


class MultiAttributeDataset(Dataset):
    def __init__(
            self,
            root,
            phase,
            split='generalized-split',
            imagenet=False
    ):
        self.root = root
        self.phase = phase
        self.split = split

        self.feat_dim = None
        self.transform = transform_image(phase, imagenet=imagenet)
        self.loader = ImageLoader(self.root + '/image/')

        self.attrs, self.objs, self.pairs, \
                self.train_pairs, self.val_pairs, \
                self.test_pairs = self.parse_split()

        self.train_data, self.val_data, self.test_data = self.get_split_info()

        if self.phase == 'train':
            self.data = self.train_data
        elif self.phase == 'val':
            self.data = self.val_data
        else:
            self.data = self.test_data

        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print('# train pairs: %d | # val pairs: %d | # test pairs: %d' % (len(
            self.train_pairs), len(self.val_pairs), len(self.test_pairs)))
        print('# train images: %d | # val images: %d | # test images: %d' %
              (len(self.train_data), len(self.val_data), len(self.test_data)))

        self.train_pair_to_idx = dict(
            [(pair, idx) for idx, pair in enumerate(self.train_pairs)]
        )

    def get_split_info(self):
        train_data, val_data, test_data = [], [], []
        label_root = os.path.join(self.root, 'label')
        image_root = os.path.join(self.root, 'image')
        for obj in os.listdir(label_root):
            obj_name = obj.strip().replace(' ', '-')
            obj_label_dir = os.path.join(label_root, obj)
            obj_image_dir = os.path.join(image_root, obj)
            if not os.path.isdir(obj_label_dir):
                continue
            for fname in os.listdir(obj_label_dir):
                if not fname.endswith('.json'):
                    continue

                json_path = os.path.join(obj_label_dir, fname)
                try:
                    with open(json_path, 'r') as f:
                        label_data = json.load(f)
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")
                    continue
                # 解析 object 名称
                # object_name = label_data.get("object", {}).get("plant", obj)  # fallback to folder name

                # 解析 attributes
                attributes = label_data.get("attributes", [])
                attr_parts = []
                for attr in attributes:
                    for k, v in attr.items():
                        attr_parts.extend(v)  # 可以拓展为 "green grainy unripe"
                attr_text = ' '.join(attr_parts).strip()

                # 验证是否在对应组合中
                image_filename = fname.replace('.json', '')
                image_path = os.path.join(obj_image_dir, image_filename)
                if os.path.exists(image_path):
                    if (attr_text, obj_name) in self.train_pairs:
                        # train_data.append({
                        #     'image_path': image_path,
                        #     'attr': attr_text,
                        #     'obj': obj_name,
                        #     'pair': (attr_text, obj_name)
                        # })
                        train_data.append([image_path, attr_text, obj_name, (attr_text, obj_name)])
                    elif (attr_text, obj_name) in self.val_pairs:
                        # val_data.append({
                        #     'image_path': image_path,
                        #     'attr': attr_text,
                        #     'obj': obj_name,
                        #     'pair': (attr_text, obj_name)
                        # })
                        val_data.append([image_path, attr_text, obj_name, (attr_text, obj_name)])
                    elif (attr_text, obj_name) in self.test_pairs:
                        # test_data.append({
                        #     'image_path': image_path,
                        #     'attr': attr_text,
                        #     'obj': obj_name,
                        #     'pair': (attr_text, obj_name)
                        # })
                        test_data.append([image_path, attr_text, obj_name, (attr_text, obj_name)])
                    else:
                        print(f"Unknown pair! {(attr_text, obj)}")

        return train_data, val_data, test_data

    def parse_split(self):
        def parse_pairs(pair_list):
            with open(pair_list, 'r') as f:
                lines = f.read().strip().split('\n')
                pairs = []
                for line in lines:
                    tokens = line.strip().split()
                    if len(tokens) < 2:
                        continue  # 跳过无效行
                    attr = ' '.join(tokens[:-1])  # 前面的作为属性
                    obj = tokens[-1]  # 最后一个是对象
                    pairs.append((attr, obj))
                attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(
            '%s/%s/train_pairs.txt' % (self.root, self.split))
        vl_attrs, vl_objs, vl_pairs = parse_pairs(
            '%s/%s/val_pairs.txt' % (self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs(
            '%s/%s/test_pairs.txt' % (self.root, self.split))

        all_attrs = sorted(list(set(tr_attrs + vl_attrs + ts_attrs)))
        all_objs = sorted(list(set(tr_objs + vl_objs + ts_objs)))
        all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs

    def revise_image_path(self, image):
        if 'mit-states' in self.root:
            pair, img = image.split('/')
            pair = pair.replace('_', ' ')
            image = pair + '/' + img
        return image

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        # pair, img = image.split('/')
        # pair = pair.replace('_', ' ')
        # image = pair + '/' + img
        image = self.revise_image_path(image)
        img = self.loader(image)
        img = self.transform(img)

        if self.phase == 'train':
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]
        else:
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

        if self.phase == 'train' and self.same_prim_sample:
            [same_attr_image, same_attr, diff_obj], same_attr_mask = self.same_A_diff_B(label_A=attr, label_B=obj, phase='attr')
            [same_obj_image, diff_attr, same_obj], same_obj_mask = self.same_A_diff_B(label_A=obj, label_B=attr, phase='obj')
            same_attr_img = self.transform(self.loader(same_attr_image))
            same_obj_img = self.transform(self.loader(same_obj_image))
            data += [same_attr_img, self.attr2idx[same_attr], self.obj2idx[diff_obj],
                     self.train_pair_to_idx[(same_attr, diff_obj)], same_attr_mask,
                     same_obj_img, self.attr2idx[diff_attr], self.obj2idx[same_obj],
                     self.train_pair_to_idx[(diff_attr, same_obj)], same_obj_mask]

        return data