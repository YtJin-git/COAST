# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Reorganize the UT-Zappos dataset to resemble the MIT-States dataset
root/attr_obj/img1.jpg
root/attr_obj/img2.jpg
root/attr_obj/img3.jpg
...
"""

import os
import argparse
import torch
import shutil
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-folder",
    default="./datasets",
    help="Directory that contains the ut-zap50k folder.",
)
args = parser.parse_args()

root = os.path.join(args.data_folder, "ut-zap50k")
print(f"root: {root}")
os.makedirs(os.path.join(root, "images"), exist_ok=True)

data = torch.load(os.path.join(root, "metadata_compositional-split-natural.t7"))
for instance in tqdm.tqdm(data):
	image, attr, obj = instance['_image'], instance['attr'], instance['obj']
	old_file = os.path.join(root, "_images", image)
	new_dir = os.path.join(root, "images", f"{attr}_{obj}")
	os.makedirs(new_dir, exist_ok=True)
	shutil.copy(old_file, new_dir)
