from stringprep import b1_set
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torchvision import models
import argparse
import math
import numpy as np
from collections import OrderedDict
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class CustomTextEncoder(torch.nn.Module):
    def __init__(self, clip_model, tokenizer, dtype=torch.float16):
        super().__init__()
        self.dtype = dtype

        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding

        self.tokenizer = tokenizer

    def tokenize(self, text):
        return torch.cat([self.tokenizer(tok) for tok in text])

    def encode_text(self, text, enable_pos_emb=True):
        token_ids = self.tokenize(text)
        text_features = self.forward(token_ids, None, enable_pos_emb)
        return text_features

    def forward(self, token_ids, token_tensors, enable_pos_emb):
        """The forward function to compute representations for the prompts.

        Args:
            token_ids (torch.tensor): the token ids, which
                contains the <eos> token.
            token_tensors (torch.Tensor, optional): the tensor
                embeddings for the token ids. Defaults to None.
            enable_pos_emb (bool, optional): adds the learned
                positional embeddigngs if true. Defaults to False.

        Returns:
            torch.Tensor: the vector representation of the prompt.
        """
        if token_tensors is not None:
            text_features = token_tensors
        else:
            text_features = self.token_embedding(token_ids)

        if len(text_features.shape) == 3:
            text_features = text_features.type(self.dtype)
            x = (
                text_features + self.positional_embedding.type(self.dtype)
                if enable_pos_emb
                else text_features
            )
            x = x.permute(1, 0, 2)
            text_feature = self.transformer(x)

            x = text_feature.permute(1, 0, 2)
            x = self.ln_final(x)
            tf = (
                x[
                    torch.arange(x.shape[0]), token_ids.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.text_projection
            )
        else:
            tfs, text_features_ = list(), list()
            text_features = text_features.type(self.dtype)
            x = (
                text_features + self.positional_embedding.type(self.dtype)
                if enable_pos_emb
                else text_features
            )
            for idx in range(x.shape[0]):
                t = x[idx].permute(1, 0, 2)
                text_feature = self.transformer(t)

                t = text_feature.permute(1, 0, 2)
                t = self.ln_final(t)
                tf = (
                        t[
                            torch.arange(t.shape[0]), token_ids.argmax(dim=-1)
                        ]  # POS of <EOS>
                        @ self.text_projection
                )
                tfs.append(tf)
                text_features_.append(text_feature)
            tf = torch.stack(tfs)
            text_feature = torch.stack(text_features_)

        return tf, text_feature


class MLP(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    '''
    def __init__(self, inp_dim, out_dim, num_layers = 1, relu = True, bias = True, dropout = False, norm = False, layers = []):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers.pop(0)
            mod.append(nn.Linear(incoming, outgoing, bias = bias))
            
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
                # mod.append(nn.BatchNorm1d(outgoing))
            mod.append(nn.ReLU(inplace = True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            if dropout:
                mod.append(nn.Dropout(p = 0.3))

        mod.append(nn.Linear(incoming, out_dim, bias = bias))

        if relu:
            mod.append(nn.ReLU(inplace = True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        self.mod = nn.Sequential(*mod)
    
    def forward(self, x):
        return self.mod(x)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("drop", nn.Dropout(0.3)),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CrossResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_x = LayerNorm(d_model)
        self.ln_y = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, y: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, y, y, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x = x + self.attention(self.ln_x(x), self.ln_y(y))
        x = x + self.mlp(self.ln_2(x))
        return x



class FusionTextImageBlock(nn.Module):
    def __init__(self, 
                 width_img: int, 
                 width_txt: int, 
                 attributes: int, 
                 classes: int, 
                 layers: int, 
                 attn_mask: torch.Tensor = None,
                 context_length: int = 8, 
                 fusion: str = "BiFusion"):
        super().__init__()
        self.fusion = fusion
        self.width_img = width_img
        self.width_txt = width_txt
        self.layers = layers
        self.context_length = context_length
        self.attributes = attributes
        self.classes = classes
        self.img2txt_transform_layer1 = nn.Linear(width_img, width_txt)
        self.img2txt_transform_layer2 = nn.Linear(257, context_length * (attributes + classes))
        self.txt2img_transform_layer1 = nn.Linear(width_txt, width_img)
        self.txt2img_transform_layer2 = nn.Linear(context_length * (attributes + classes), 257)
        self.dropout = nn.Dropout(0.3)
        self.crossblock_img = CrossResidualAttentionBlock(width_img, width_img//64, attn_mask)
        self.crossblock_txt = CrossResidualAttentionBlock(width_txt, width_txt//64, attn_mask)
        self.resblocks_img = nn.Sequential(*[ResidualAttentionBlock(width_img, width_img//64, attn_mask) for _ in range(layers)])
        self.resblocks_txt = nn.Sequential(*[ResidualAttentionBlock(width_txt, width_txt//64, attn_mask) for _ in range(layers)])
        self.txt_fine_tune = nn.Linear(self.width_txt, self.width_txt)


    def decompose(self, text_feature, idx):
        t, l, c = text_feature.shape
        att_idx, obj_idx = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
        text_att = torch.zeros(t, self.attributes, c).cuda()
        text_obj = torch.zeros(t, self.classes, c).cuda()
        for i in range(self.attributes):
            text_att[:, i, :] = text_feature[:, np.where(att_idx==i)[0], :].mean(-2)
        for i in range(self.classes):
            text_obj[:, i, :] = text_feature[:, np.where(obj_idx==i)[0], :].mean(-2)    
        text_decom_feature = torch.cat([text_att, text_obj], dim=1)
        return text_decom_feature


    def compose(self, text_feature, idx):
        t, l, c = text_feature.shape
        att_idx, obj_idx = idx[:, 0].cpu().numpy(), idx[:, 1].cpu().numpy()
        text_com_feature = torch.zeros(t, len(idx), c).cuda()
        text_com_feature = text_feature[:, att_idx, :] * text_feature[:, obj_idx + self.attributes, :]
        text_com_feature = self.txt_fine_tune(text_com_feature)
        return text_com_feature



    def img2txt(self, x: torch.Tensor):
        x = self.img2txt_transform_layer1(x)
        x = x.permute(2,1,0)
        x = self.img2txt_transform_layer2(x)
        x = x.permute(2,1,0).reshape(-1, (self.attributes + self.classes), self.width_txt)
        x = self.dropout(x)
        return x

    def txt2img(self, x:torch.Tensor, idx, b: int):    
        x = self.decompose(x, idx)
        x = self.txt2img_transform_layer1(x)
        x = rearrange(x, 't l c -> c (t l)')
        x = self.txt2img_transform_layer2(x)
        x = self.dropout(x)
        x = x.permute(1,0).unsqueeze(1).repeat(1,b,1)
        return x
        

    def forward(self, x_image: torch.Tensor, x_text: torch.Tensor, idx, b: int):
        if self.fusion == "BiFusion":
            x_img = self.crossblock_img(x_image, self.txt2img(x_text, idx, b))
            x_txt = self.img2txt(x_image)
            x_text = self.decompose(x_text, idx)
            x_txt = self.crossblock_txt(x_text.repeat(b, 1, 1), x_txt)
            x_txt = self.resblocks_txt(x_txt)
            x_txt = self.compose(x_txt, idx)
            x_txt = x_txt.reshape(b, self.context_length, -1, self.width_txt)
            x_img = self.resblocks_img(x_img)
            return x_img, x_txt
        elif self.fusion == "img2txt":
            x_txt = self.img2txt(x_image)
            x_text = self.decompose(x_text, idx)
            x_txt = self.crossblock_txt(x_text.repeat(b, 1, 1), x_txt)
            x_txt = self.resblocks_txt(x_txt)
            x_txt = self.compose(x_txt, idx)
            x_txt = x_txt.reshape(b, self.context_length, -1, self.width_txt)
            x_img = self.resblocks_img(x_image)
            return x_img, x_txt
        elif self.fusion == "txt2img":
            x_img = self.crossblock_img(x_image, self.txt2img(x_text, idx, b))
            x_img = self.resblocks_img(x_img)
            x_txt = self.resblocks_txt(x_text)
            return x_img, x_txt
        elif self.fusion == "OnlySPM":
            return x_image, x_text


class Adapter(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        self.init_option = init_option

        self._reset_parameters()

    def _reset_parameters(self):
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class Disentangler(nn.Module):
    def __init__(self, emb_dim):
        super(Disentangler, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim)
        self.bn1_fc = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        # residual = x
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        # return x + residual
        return x


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # print(self._num_experts)
        # sort experts
        # print('gates', gates.shape) # 64, 22
        # [[0.0000, 0.0000, 0.5146, 0.4854, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        #         [0.0000, 0.0000, 0.0000, 0.0000, 0.4666, 0.5334, 0.0000, 0.0000, 0.0000]]
        # print(torch.nonzero(gates).shape)  # torch.Size([128, 2])
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)

        # print(sorted_experts.shape, index_sorted_experts.shape) # torch.Size([128, 2]) torch.Size([128, 2])
        # [[0, 2],[0, 3],[1, 4],[1, 5]] sorted_experts 将feature和experts匹配上
        # [[1, 0],[0, 1],[2, 2],[3, 3]]

        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # print(self._batch_index)
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes

        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space

        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)  # weight


        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), device=stitched.device)
        # combine samples that have been processed by the same k experts

        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # back to log space
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoEAdapter(nn.Module):
    # Reference: https://github.com/JiazuoYu/MoE-Adapters4CL
    # Written by Yutang Jin: https://github.com/YtJin-git
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.1,
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none",
                 noisy_gating=True,
                 top_k=2,
                 experts_num=8,
                 init_option="lora"):
        super().__init__()
        assert experts_num >= 1, "MoEAdapter must have at least 1 expert."
        self.d_model = d_model
        self.bottleneck = bottleneck
        self.experts_num = experts_num
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.init_option = init_option
        self.adapter_scalar = adapter_scalar
        self.adapter_layernorm_option = adapter_layernorm_option
        self.dropout = dropout
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        # MoE start

        # Router parameters
        self.router = nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(d_model, self.experts_num), requires_grad=True)

        self.experts_list = nn.ModuleList()

        for i in range(self.experts_num):  # Expert number
            adaptmlp = Adapter(d_model=d_model, dropout=self.dropout, bottleneck=self.bottleneck,
                               init_option=self.init_option, adapter_scalar=self.adapter_scalar,
                               adapter_layernorm_option=self.adapter_layernorm_option)
            self.experts_list.append(adaptmlp)
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

        # MoE end

        self.dropout = dropout

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        normal = Normal(self.mean, self.std)

        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, w_gate, w_noise, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = x @ w_gate.to(x)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ w_noise.to(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.experts_num), dim=1)
        top_k_logits = top_logits[:, :self.top_k]
        top_k_indices = top_indices[:, :self.top_k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        if self.noisy_gating and self.top_k < self.experts_num and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x):
        x_re = x.permute(1, 0, 2)[:, 0, :]
        gates, load = self.noisy_top_k_gating(x=x_re, train=True, w_gate=self.router, w_noise=self.w_noise)

        dispatcher = SparseDispatcher(self.experts_num, gates)
        expert_inputs = dispatcher.dispatch(x.permute(1, 0, 2).reshape(x.shape[1], -1))  #
        expert_outputs = [self.experts_list[i](expert_inputs[i].view(expert_inputs[i].shape[0],
                                                                      x.shape[0], x.shape[2]).to(x), add_residual=False)
                          for i in range(self.experts_num)]  # 11 experts 1 router
        i = 0
        while i < len(expert_outputs):
            if expert_outputs[i].shape[0] == 0:
                expert_outputs.pop(i)
            else:
                expert_outputs[i] = expert_outputs[i].view(expert_outputs[i].shape[0], -1)
                i += 1

        y = dispatcher.combine(expert_outputs)
        y = y.view(x.shape[1], x.shape[0], x.shape[2]).permute(1, 0, 2)
        return y


class LoRA(nn.Module):
    # Low-Rank Adaptation implementation
    def __init__(self, d_model=None, lora_rank=8, lora_alpha=16, lora_dropout=0.0,
                 adapter_scalar="0.1", adapter_layernorm_option="none"):
        super().__init__()
        self.d_model = d_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scale = lora_alpha / lora_rank
        
        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.d_model)
        
        if adapter_scalar == "learnable_scalar":
            self.adapter_scale = nn.Parameter(torch.ones(1))
        else:
            self.adapter_scale = float(adapter_scalar)
        
        # LoRA weights
        self.lora_A = nn.Linear(d_model, lora_rank, bias=False)
        self.lora_B = nn.Linear(lora_rank, d_model, bias=False)
        self.dropout = nn.Dropout(lora_dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        
        # LoRA computation
        x_lora = self.lora_A(x)
        x_lora = self.dropout(x_lora)
        x_lora = self.lora_B(x_lora)
        x_lora = x_lora * self.scale * self.adapter_scale
        
        if self.adapter_layernorm_option == 'out':
            x_lora = self.adapter_layer_norm_before(x_lora)
        
        if add_residual:
            output = x_lora + residual
        else:
            output = x_lora
        
        return output

class CustomTextEncoderMoEAdapter(torch.nn.Module):
    def __init__(self, clip_model, tokenizer, dtype=torch.float16):
        super().__init__()
        self.dtype = dtype

        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding

        self.tokenizer = tokenizer

    def tokenize(self, text):
        return torch.cat([self.tokenizer(tok) for tok in text])

    def encode_text(self, text, enable_pos_emb=True):
        token_ids = self.tokenize(text)
        text_features = self.forward(token_ids, None, enable_pos_emb)
        return text_features

    def forward(self, token_ids, moe_adapter, token_tensors, enable_pos_emb):
        """The forward function to compute representations for the prompts.

        Args:
            token_ids (torch.tensor): the token ids, which
                contains the <eos> token.
            token_tensors (torch.Tensor, optional): the tensor
                embeddings for the token ids. Defaults to None.
            enable_pos_emb (bool, optional): adds the learned
                positional embeddigngs if true. Defaults to False.

        Returns:
            torch.Tensor: the vector representation of the prompt.
        """
        if token_tensors is not None:
            text_features = token_tensors
        else:
            text_features = self.token_embedding(token_ids)

        if len(text_features.shape) == 3:
            text_features = text_features.type(self.dtype)
            x = (
                text_features + self.positional_embedding.type(self.dtype)
                if enable_pos_emb
                else text_features
            )
            x = x.permute(1, 0, 2)
            # text_feature = self.transformer(x)
            # MoE Adapter
            for i_block in range(self.transformer.layers):
                # MHA
                residual = x
                x = self.transformer.resblocks[i_block].attention(
                    self.transformer.resblocks[i_block].ln_1(x)
                )
                x = x + residual

                # MoE
                residual = x
                adapt_x = moe_adapter[i_block](x)
                x = self.transformer.resblocks[i_block].mlp(
                    self.transformer.resblocks[i_block].ln_2(x)
                )
                x = x + adapt_x + residual

            text_feature = x

            x = text_feature.permute(1, 0, 2)
            x = self.ln_final(x)
            tf = (
                x[
                    torch.arange(x.shape[0]), token_ids.argmax(dim=-1)
                ]  # POS of <EOS>
                @ self.text_projection
            )
        else:
            tfs, text_features_ = list(), list()
            text_features = text_features.type(self.dtype)
            x = (
                text_features + self.positional_embedding.type(self.dtype)
                if enable_pos_emb
                else text_features
            )
            for idx in range(x.shape[0]):
                t = x[idx].permute(1, 0, 2)
                text_feature = self.transformer(t)

                t = text_feature.permute(1, 0, 2)
                t = self.ln_final(t)
                tf = (
                        t[
                            torch.arange(t.shape[0]), token_ids.argmax(dim=-1)
                        ]  # POS of <EOS>
                        @ self.text_projection
                )
                tfs.append(tf)
                text_features_.append(text_feature)
            tf = torch.stack(tfs)
            text_feature = torch.stack(text_features_)

        return tf, text_feature