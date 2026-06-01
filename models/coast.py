import os
import torch
import torch.nn as nn
from clip_modules.clip_model import load_clip, QuickGELU
from clip_modules.tokenization_clip import SimpleTokenizer
from models.common import *
from torch.nn.modules.loss import CrossEntropyLoss
import operator
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

class ThreeBranchesPretrain(nn.Module):
    def __init__(self, config, dset):
        super(ThreeBranchesPretrain, self).__init__()
        self.device = f"cuda:{config.cuda_device}" if torch.cuda.is_available() else "cpu"
        self.clip = load_clip(name=config.clip_arch, context_length=config.context_length, device=self.device)
        self.tokenizer = SimpleTokenizer()
        self.config = config

        allattrs = dset.attrs
        allobj = dset.objs
        classes = [cla.replace(".", " ").lower() for cla in allobj]
        attributes = [attr.replace(".", " ").lower() for attr in allattrs]
        offset = len(attributes)
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.num_attrs, self.num_objs, self.num_pairs = len(dset.attrs), len(dset.objs), len(dset.pairs)
        # self.full_pairs = list(product(dset.attrs, dset.objs))
        self.pairs = dset.pairs
        self.attr_idx = dset.attr2idx
        self.obj_idx = dset.obj2idx
        self.pair_idx = dset.pair2idx

        all_element_words = list(dset.attrs) + list(dset.objs)
        self.attr_obj_displacement = len(dset.attrs)
        self.element_pair_displacement = len(all_element_words)

        self.dict_Obj2IDX = {word: idx for idx, word in enumerate(dset.objs)}
        self.dict_Attr2IDX = {word: idx for idx, word in enumerate(dset.attrs)}

        self.token_ids, self.soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        dtype = self.clip.dtype
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype

        self.text_encoder = CustomTextEncoder(self.clip, self.tokenizer, self.dtype)
        # freeze CLIP's parameters
        for p in self.parameters():
            p.requires_grad = False

        # only consider ViT as visual encoder
        assert 'ViT' in config.clip_model
        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.comp_ctx_vectors = nn.Parameter(comp_ctx_vectors).to(self.device)
        self.attr_ctx_vectors = nn.Parameter(attr_ctx_vectors).to(self.device)
        self.obj_ctx_vectors = nn.Parameter(obj_ctx_vectors).to(self.device)

        self.additional_visual_params = self.add_visual_tunable_params()
        output_dim = self.clip.visual.output_dim

        self.attr_disentangler = Disentangler(output_dim).to(self.device)
        self.obj_disentangler = Disentangler(output_dim).to(self.device)

    def release_text_encoder(self):
        del self.token_ids
        del self.soft_att_obj
        del self.comp_ctx_vectors
        del self.tokenizer
        del self.text_encoder
        # del self.soft_a_and_o_embedding
        torch.cuda.empty_cache()

    def add_visual_tunable_params(self):
        adapter_num = 2 * self.clip.visual.transformer.layers
        params = nn.ModuleList([Adapter(d_model=self.clip.visual.transformer.width,
                                        bottleneck=self.config.adapter_dim,
                                        dropout=self.config.adapter_dropout
                                        ) for _ in range(adapter_num)])
        return params

    def encode_image(self, x: torch.Tensor):
        return self.encode_image_with_adapter(x)

    def encode_image_with_adapter(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                        device=x.device), x],
            dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # img_feature = self.clip.visual.transformer(x)
        for i_block in range(self.clip.visual.transformer.layers):
            # MHA
            adapt_x = self.additional_visual_params[i_block](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            x = x + adapt_x + residual
            # x = x + residual

            # FFN
            i_adapter = i_block + self.clip.visual.transformer.layers
            adapt_x = self.additional_visual_params[i_adapter](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual
            # x = x + residual

        img_feature = x.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.visual.ln_post(img_feature)
        if self.clip.visual.proj is not None:
            img_feature = img_feature @ self.clip.visual.proj
        return img_feature[:, 0, :], img_feature

    def construct_soft_prompt(self):
        # token_ids indicates the position of [EOS]
        token_ids = self.tokenizer(self.config.prompt_template,
                                   context_length=self.config.context_length).to(self.device)

        tokenized = torch.cat(
            [
                self.tokenizer(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.to(self.device))
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = self.config.ctx_init
        assert isinstance(ctx_init, list)
        n_ctx = [len(ctx.split()) for ctx in ctx_init]
        prompt = self.tokenizer(ctx_init,
                                context_length=self.config.context_length).to(self.device)
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)

        comp_ctx_vectors = embedding[0, 1: 1 + n_ctx[0], :].to(self.clip.dtype)
        attr_ctx_vectors = embedding[1, 1: 1 + n_ctx[1], :].to(self.clip.dtype)
        obj_ctx_vectors = embedding[2, 1: 1 + n_ctx[2], :].to(self.clip.dtype)

        return token_ids, soft_att_obj, comp_ctx_vectors, attr_ctx_vectors, obj_ctx_vectors

    def encode_text_for_open(self, idx):
        token_tensors = self.construct_token_tensors(idx)
        text_features = []
        for i_element in range(self.token_ids.shape[0]):
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            text_features.append(idx_text_features)
        return text_features

    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        token_tensor, num_elements = list(), [len(pair_idx), self.offset, len(self.classes)]
        for i_element in range(self.token_ids.shape[0]):
            class_token_ids = self.token_ids[i_element].repeat(num_elements[i_element], 1)
            token_tensor.append(self.clip.token_embedding(
                class_token_ids.to(self.device)
            ).type(self.clip.dtype))

        eos_idx = [int(self.token_ids[i_element].argmax()) for i_element in range(self.token_ids.shape[0])]
        soft_att_obj = self.attr_dropout(self.soft_att_obj).to(self.device)
        # comp
        token_tensor[0][:, eos_idx[0] - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[0][:, eos_idx[0] - 1, :] = soft_att_obj[
            obj_idx + self.offset
            ].type(self.clip.dtype)
        token_tensor[0][
            :, 1: len(self.comp_ctx_vectors) + 1, :
        ] = self.comp_ctx_vectors.type(self.clip.dtype)

        # attr
        token_tensor[1][:, eos_idx[1] - 1, :] = soft_att_obj[
            :self.offset
        ].type(self.clip.dtype)
        token_tensor[1][
            :, 1: len(self.attr_ctx_vectors) + 1, :
        ] = self.attr_ctx_vectors.type(self.clip.dtype)

        # obj
        token_tensor[2][:, eos_idx[2] - 1, :] = soft_att_obj[
            self.offset:
        ].type(self.clip.dtype)
        token_tensor[2][
            :, 1: len(self.obj_ctx_vectors) + 1, :
        ] = self.obj_ctx_vectors.type(self.clip.dtype)

        return token_tensor

    def encode_text(self, token_ids, token_tensors=None, enable_pos_emb=False):
        return self.text_encoder(token_ids, token_tensors, enable_pos_emb)

    def loss_calu(self, predict, target, train_pairs=None):
        loss_fn = CrossEntropyLoss()
        _, batch_attr, batch_obj, batch_target = target
        batch_attr = batch_attr.to(self.device)
        batch_obj = batch_obj.to(self.device)
        batch_target = batch_target.to(self.device)
        loss_comp = loss_fn(predict['comp_logits'], batch_target)
        loss_attr = loss_fn(predict['attr_logits'], batch_attr)
        loss_obj = loss_fn(predict['obj_logits'], batch_obj)

        loss = loss_comp * self.config.pair_loss_weight + \
               loss_attr * self.config.attr_loss_weight + \
               loss_obj * self.config.obj_loss_weight

        if self.training:
            attr_pred_softmax = F.softmax(predict['attr_logits'], dim=-1)  # [B, num_attrs]
            obj_pred_softmax = F.softmax(predict['obj_logits'], dim=-1)  # [B, num_objs]
            comp_pred_softmax = F.softmax(predict['comp_logits'], dim=-1)  # [B, num_pairs]

            attr_idx = train_pairs[:, 0]
            obj_idx = train_pairs[:, 1]
            attr_obj_reorder = attr_pred_softmax[:, attr_idx] * obj_pred_softmax[:, obj_idx]

            attr_obj_reorder = attr_obj_reorder / (attr_obj_reorder.sum(dim=-1, keepdim=True) + 1e-8)

            kl_loss = F.kl_div(
                (attr_obj_reorder + 1e-8).log(),
                comp_pred_softmax,
                reduction='batchmean',
                log_target=False
            )
            loss = loss + kl_loss * self.config.kl_loss_weight

        return loss

    def logit_infer(self, predict, pairs):
        attr_logits = predict['attr_logits'] * self.config.attr_inference_weight
        obj_logits = predict['obj_logits'] * self.config.obj_inference_weight
        attr_pred = F.softmax(attr_logits, dim=-1)
        obj_pred = F.softmax(obj_logits, dim=-1)
        for i_comp in range(predict['comp_logits'].shape[-1]):
            weighted_attr_pred = 1 if self.config.attr_inference_weight == 0 else attr_pred[:, pairs[i_comp][0]]
            weighted_obj_pred = 1 if self.config.obj_inference_weight == 0 == 0 else obj_pred[:, pairs[i_comp][1]]
            predict['comp_logits'][:, i_comp] = predict['comp_logits'][
                                                    :, i_comp] * self.config.pair_inference_weight + weighted_attr_pred * weighted_obj_pred

        return predict['comp_logits']

    def forward_for_open(self, batch, text_feats):
        batch_img = batch[0].to(self.device)
        b = batch_img.shape[0]
        # l, _ = idx.shape
        batch_img, batch_patch = self.encode_image(batch_img.type(self.clip.dtype))
        batch_img_features = [batch_img, self.attr_disentangler(batch_img), self.obj_disentangler(batch_img)]
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        logits = {}
        key = ["comp_logits", "attr_logits", "obj_logits"]
        for i_element in range(self.token_ids.shape[0]):
            idx_text_features = text_feats[i_element]
            logits[key[i_element]] = torch.einsum("bd, kd->bk", normalized_img_features[i_element],
                                                  idx_text_features * self.clip.logit_scale.exp())
        return logits

    def forward(self, batch, pairs_idx):
        batch_img = batch[0].to(self.device)

        batch_img, batch_patch = self.encode_image(batch_img.type(self.clip.dtype))
        batch_img_features = [batch_img, self.attr_disentangler(batch_img), self.obj_disentangler(batch_img)]
        normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

        logits = {}
        key = ["comp_logits", "attr_logits", "obj_logits"]
        token_tensors = self.construct_token_tensors(pairs_idx)  # c, a, o
        for i_element in range(self.token_ids.shape[0]):
            _text_features, _ = self.encode_text(
                self.token_ids[i_element],
                token_tensors[i_element],
                enable_pos_emb=self.enable_pos_emb,
            )

            idx_text_features = _text_features / _text_features.norm(
                dim=-1, keepdim=True
            )
            logits[key[i_element]] = torch.einsum("bd, kd->bk", normalized_img_features[i_element],
                                                  idx_text_features * self.clip.logit_scale.exp())

        return logits

class TextKAM(nn.Module):
    def __init__(self, text_feats, device):
        super(TextKAM, self).__init__()
        self.dim1, self.dim2 = text_feats.shape
        self.residual = nn.Parameter(torch.zeros([self.dim1, self.dim2], dtype=text_feats.dtype).to(device), requires_grad=True)

    def forward(self, x, weight):
        weight_reshaped = weight.view(-1, 1)
        x_new = x.clone() + weight_reshaped * self.residual
        x_new = F.normalize(x_new, dim=-1)
        return x_new

class VisualKAM(nn.Module):
    def __init__(self, text_feats, device):
        super(VisualKAM, self).__init__()
        self.dim1, self.dim2 = text_feats.shape
        self.residual = nn.Parameter(torch.zeros([self.dim1, self.dim2], dtype=text_feats.dtype).to(device), requires_grad=True)

    def forward(self, x, weight, all_classes):
        weight_reshaped = weight.view(-1, 1)
        x_new = x.clone() + weight_reshaped * self.residual[all_classes]
        x_new = F.normalize(x_new, dim=-1)
        return x_new

def contrastive_loss( x: torch.Tensor, y: torch.Tensor, temperature):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    batch_size = x.shape[0]

    similarity_matrix = torch.mm(x, y.t()) * temperature.to(dtype=x.dtype)  # (batch_size, batch_size)

    labels = torch.arange(batch_size, device=x.device)

    loss_x = F.cross_entropy(similarity_matrix, labels)
    loss_y = F.cross_entropy(similarity_matrix.t(), labels)

    return (loss_x + loss_y) / 2

def self_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

def _normalize_torch_device(device):
    if isinstance(device, int):
        return torch.device(f"cuda:{device}")
    return torch.device(device)

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity."""
    with torch.no_grad():
        feature = features_loss[0].detach()
        loss_value = features_loss[1]
        if torch.is_tensor(loss_value):
            loss_value = loss_value.detach().float().mean().item()
        else:
            loss_value = float(loss_value)

        item = [feature, loss_value]
        if include_prob_map:
            prob_map = features_loss[2]
            item.append(prob_map.detach() if torch.is_tensor(prob_map) else prob_map)

        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif loss_value < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]
        return

def cache_key_value(image_features, cache, clip_weights, device):
    """Compute logits using positive/negative cache."""
    with torch.no_grad():
        target_device = _normalize_torch_device(device)
        cache_keys = []
        cache_class_indices = []
        all_classes = []
        for class_index in sorted(cache.keys()):
            num_items = len(cache[class_index])
            image_prototype = torch.zeros_like(image_features)
            for item in cache[class_index]:
                image_prototype += item[0] / num_items
            cache_keys.append(image_prototype)
            cache_class_indices.append(class_index)
            all_classes.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0)
        cache_values = (
            torch.tensor(cache_class_indices, dtype=torch.long, device=target_device),
            clip_weights.size(0),
        )
        return cache_keys, cache_values, all_classes

def compute_cache_logits(image_features, cache_keys, cache_values, alpha, beta, device):
    affinity = image_features @ cache_keys.T
    cache_scores = ((-1) * (beta - beta * affinity)).exp()
    if isinstance(cache_values, tuple):
        class_indices, num_classes = cache_values
        class_indices = class_indices.to(cache_scores.device)
        cache_logits = torch.zeros(
            cache_scores.shape[0],
            num_classes,
            dtype=cache_scores.dtype,
            device=cache_scores.device,
        )
        cache_logits.scatter_add_(1, class_indices.expand(cache_scores.shape[0], -1), cache_scores)
    else:
        cache_logits = cache_scores @ cache_values
    return alpha * cache_logits.to(_normalize_torch_device(device))

def adaptive_update_weight(img_feats, text_feats, alpha=10):
    similarity = img_feats @ text_feats.T
    x = 1 / (1 + torch.exp(alpha * similarity))
    return x

def get_clip_logits_three_branches(img_feats, model, text_feats):
    logits = torch.einsum("bd, kd->bk", img_feats,
                               text_feats * model.clip.logit_scale.exp())
    entropy = self_entropy(logits)
    pred = logits.argmax(dim=1)[0].item()
    return logits, entropy, pred

def predict_logits_text_first_with_coast(model, dataset, config):
    """
    Function to predict the cosine similarities between the images and the attribute-object representations.
    Incorporates Hierarchical TTA: Dynamic Routing (Entropy-based), Margin-Aware Gating, Primitive Anchoring,
    and Decoupled Marginal Adaptation.
    """
    model.eval()
    device = config.cuda_device
    cpu = config.cpu_cache
    use_cache = config.use_img_cache

    routing_temp = config.routing_temp
    distill_weight = config.distill_weight
    kl_entropy_threshold = config.kl_entropy_threshold

    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                          for attr, obj in dataset.pairs]).to(device)

    attr_idx, obj_idx = pairs[:, 0], pairs[:, 1]
    logit_scale_exp = model.clip.logit_scale.exp().item()

    dataloader = DataLoader(dataset, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.num_workers)

    all_logits_list = []
    all_attr_gt, all_obj_gt, all_pair_gt = [], [], []

    with torch.no_grad():
        text_feats = [[], [], []]
        for i in range(0, pairs.shape[0], config.text_encoder_batch_size):
            cur_pair = pairs[i: i + config.text_encoder_batch_size]
            cur_text_feats = model.encode_text_for_open(cur_pair)

            text_feats[0].append(cur_text_feats[0])
            if not text_feats[1]:
                text_feats[1].append(cur_text_feats[1])
                text_feats[2].append(cur_text_feats[2])

        text_feats_c = torch.cat(text_feats[0], dim=0)[:model.num_pairs]
        text_feats_a = text_feats[1][0][:model.num_attrs]
        text_feats_o = text_feats[2][0][:model.num_objs]

        text_feats_c = F.normalize(text_feats_c, p=2, dim=-1)
        text_feats_a = F.normalize(text_feats_a, p=2, dim=-1)
        text_feats_o = F.normalize(text_feats_o, p=2, dim=-1)

    model.release_text_encoder()

    pos_params = {
        'shot_capacity': config.shot_capacity,
        'alpha': config.alpha,
        'beta': config.beta,
    }

    text_kam_c = TextKAM(text_feats_c, device)
    text_kam_a = TextKAM(text_feats_a, device)
    text_kam_o = TextKAM(text_feats_o, device)

    optimizer_t = torch.optim.AdamW([
        {'params': text_kam_c.parameters(), 'lr': config.text_lr, 'eps': config.eps, 'weight_decay': config.wd},
        {'params': text_kam_a.parameters(), 'lr': config.text_lr, 'eps': config.eps, 'weight_decay': config.wd},
        {'params': text_kam_o.parameters(), 'lr': config.text_lr, 'eps': config.eps, 'weight_decay': config.wd}
    ])

    if use_cache:
        visual_kam_c = VisualKAM(text_feats_c, 'cpu' if cpu else device)
        pos_cache_c = {}
        optimizer_i = torch.optim.AdamW([
            {'params': visual_kam_c.parameters(), 'lr': config.image_lr, 'eps': config.eps, 'weight_decay': config.wd}
        ])

    total_true, update_sample_num = 0, 0

    for idx, data in tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
        img = data[0].to(device)

        with torch.no_grad():
            img_feats, _ = model.encode_image(img.type(model.clip.dtype))
            batch_img_features = [img_feats, model.attr_disentangler(img_feats), model.obj_disentangler(img_feats)]
            normalized_img_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in batch_img_features]

            text_weight_c = adaptive_update_weight(normalized_img_features[0], text_feats_c, config.theta)
            text_weight_a = adaptive_update_weight(normalized_img_features[1], text_feats_a, config.theta)
            text_weight_o = adaptive_update_weight(normalized_img_features[2], text_feats_o, config.theta)

        new_text_feats_c = text_kam_c(text_feats_c, text_weight_c)
        new_text_feats_a = text_kam_a(text_feats_a, text_weight_a)
        new_text_feats_o = text_kam_o(text_feats_o, text_weight_o)

        logits_c = (normalized_img_features[0] @ new_text_feats_c.t()) * logit_scale_exp
        logits_a = (normalized_img_features[1] @ new_text_feats_a.t()) * logit_scale_exp
        logits_o = (normalized_img_features[2] @ new_text_feats_o.t()) * logit_scale_exp

        pred_c_idx = logits_c.argmax(dim=-1)
        pred_a_idx = logits_a.argmax(dim=-1)
        pred_o_idx = logits_o.argmax(dim=-1)

        is_consistent = (attr_idx[pred_c_idx] == pred_a_idx) & (obj_idx[pred_c_idx] == pred_o_idx)
        pred_c = pred_c_idx[0].item()

        probs_c_init = F.softmax(logits_c, dim=-1)
        entropy_c = -torch.sum(probs_c_init * torch.log(probs_c_init + 1e-8), dim=-1)

        logits_ao = (logits_a[:, attr_idx] + logits_o[:, obj_idx]) / 2.0
        probs_ao = F.softmax(logits_ao, dim=-1)
        entropy_ao = -torch.sum(probs_ao * torch.log(probs_ao + 1e-8), dim=-1)

        delta_entropy = entropy_c - entropy_ao
        ao_weight = torch.sigmoid(delta_entropy / routing_temp).unsqueeze(-1)

        logits_fused = (1.0 - ao_weight) * logits_c + ao_weight * logits_ao

        if use_cache:
            if is_consistent[0].item():
                feat_for_cache = normalized_img_features[0].to('cpu') if cpu else normalized_img_features[0]

                entropy_for_cache = entropy_c.to('cpu') if cpu else entropy_c

                update_cache(pos_cache_c, pred_c, [feat_for_cache, entropy_for_cache], pos_params['shot_capacity'])

                pos_cache_keys_c, pos_cache_values_c, all_classes_c = cache_key_value(
                    feat_for_cache, pos_cache_c, new_text_feats_c, 'cpu' if cpu else device)

                with torch.no_grad():
                    cache_weight_c = adaptive_update_weight(feat_for_cache, pos_cache_keys_c, config.theta)

                new_pos_cache_keys_c = visual_kam_c(pos_cache_keys_c, cache_weight_c, all_classes_c)

                logits_c = logits_c + compute_cache_logits(
                    feat_for_cache, new_pos_cache_keys_c, pos_cache_values_c, pos_params['alpha'], pos_params['beta'],
                    device)

                probs_c_new = F.softmax(logits_c, dim=-1)
                entropy_c_new = -torch.sum(probs_c_new * torch.log(probs_c_new + 1e-8), dim=-1)

                delta_entropy_new = entropy_c_new - entropy_ao
                ao_weight_new = torch.sigmoid(delta_entropy_new / routing_temp).unsqueeze(-1)
                logits_fused = (1.0 - ao_weight_new) * logits_c + ao_weight_new * logits_ao

        probs_fused = F.softmax(logits_fused, dim=-1)
        entropy_fused = -torch.sum(probs_fused * torch.log(probs_fused + 1e-8), dim=-1)
        topk_probs_fused, _ = torch.topk(probs_fused, k=2, dim=-1)
        margin_fused = topk_probs_fused[:, 0] - topk_probs_fused[:, 1]
        valid_mask = (entropy_fused < config.entropy_threshold) & (margin_fused > config.margin_threshold)

        probs_a = F.softmax(logits_a, dim=-1)
        probs_o = F.softmax(logits_o, dim=-1)
        entropy_a = -torch.sum(probs_a * torch.log(probs_a + 1e-8), dim=-1)
        entropy_o = -torch.sum(probs_o * torch.log(probs_o + 1e-8), dim=-1)

        valid_a_mask = entropy_a < config.entropy_threshold
        valid_o_mask = entropy_o < config.entropy_threshold

        if is_consistent[0].item() and valid_mask[0].item():
            update_sample_num += 1
            if probs_fused.argmax(-1).cpu().equal(data[3]):
                total_true += 1

            loss = entropy_fused.mean()
            # loss = 0.0

            pseudo_a_labels, pseudo_o_labels = pred_a_idx, pred_o_idx

            loss_distill_a = F.cross_entropy(logits_a, pseudo_a_labels)
            loss_distill_o = F.cross_entropy(logits_o, pseudo_o_labels)
            loss = loss + distill_weight * (loss_distill_a + loss_distill_o)

            if config.use_align_loss and use_cache:
                image2text_loss_c = contrastive_loss(new_pos_cache_keys_c.to(config.cuda_device),
                                                     new_text_feats_c[all_classes_c, :], model.clip.logit_scale.exp())
                loss = loss + config.align_loss_weight * image2text_loss_c

            if config.use_kl_loss:
                attr_obj_reorder = probs_a[:, attr_idx] * probs_o[:, obj_idx]
                attr_obj_reorder = attr_obj_reorder / (attr_obj_reorder.sum(dim=-1, keepdim=True) + 1e-8)

                ao_entropy = -torch.sum(attr_obj_reorder * torch.log(attr_obj_reorder + 1e-8), dim=-1)

                if (ao_entropy < kl_entropy_threshold)[0].item():
                    comp_pred_softmax = F.softmax(logits_c, dim=-1)
                    kl_loss_raw = F.kl_div(
                        (attr_obj_reorder + 1e-8).log(),
                        comp_pred_softmax,
                        reduction='batchmean',
                        log_target=False)
                    loss = loss + kl_loss_raw

            optimizer_t.zero_grad()
            if config.use_img_cache:
                optimizer_i.zero_grad()
            loss.backward()
            optimizer_t.step()
            if config.use_img_cache:
                optimizer_i.step()

        else:
            loss_marginal = 0.0
            update_flag = False

            if valid_a_mask[0].item():
                loss_dist_a_local = F.cross_entropy(logits_a, pred_a_idx)
                loss_marginal = loss_marginal + distill_weight * loss_dist_a_local
                update_flag = True

            if valid_o_mask[0].item():
                loss_dist_o_local = F.cross_entropy(logits_o, pred_o_idx)
                loss_marginal = loss_marginal + distill_weight * loss_dist_o_local
                update_flag = True

            if update_flag:
                optimizer_t.zero_grad()
                loss_marginal.backward()
                optimizer_t.step()

        all_logits_list.append(logits_fused.detach().cpu())
        all_attr_gt.append(data[1])
        all_obj_gt.append(data[2])
        all_pair_gt.append(data[3])

    all_logits = torch.cat(all_logits_list, dim=0)
    all_attr_gt = torch.cat(all_attr_gt).to("cpu")
    all_obj_gt = torch.cat(all_obj_gt).to("cpu")
    all_pair_gt = torch.cat(all_pair_gt).to("cpu")

    print(f"total_true: {total_true}")
    update_precision = total_true / len(dataloader)
    update_coverage = update_sample_num / len(dataloader)
    print("Update_precision: ", update_precision)
    print("Update_coverage: ", update_coverage)

    return all_logits, all_attr_gt, all_obj_gt, all_pair_gt

