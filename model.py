import math

from einops import rearrange
import numpy as np
import torch as pt
import torch.nn as nn
import torchvision.models as ptvm

from utils import register_module


#### initializers


def lecun_normal_(tensor, scale=1.0):
    """timm.models.layers.lecun_normal_, for conv/transposed-conv"""

    def _calculate_fanin_and_fanout(tensor):
        ndim = tensor.dim()
        assert ndim >= 2
        receptive_field_size = np.prod(tensor.shape[2:]) if ndim > 2 else 1
        fan_in = tensor.size(1) * receptive_field_size
        fan_out = tensor.size(0) * receptive_field_size
        return fan_in, fan_out

    def _trunc_normal_(tensor, mean, std, a, b):
        norm_cdf = lambda x: (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
        assert not ((mean < a - 2 * std) or (mean > b + 2 * std))
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)

    fan_in, fan_out = _calculate_fanin_and_fanout(tensor)
    denom = fan_in
    variance = scale / denom
    std = math.sqrt(variance) / 0.87962566103423978
    with pt.no_grad():
        _trunc_normal_(tensor, 0, 1.0, -2, 2)
        tensor.mul_(std).add_(0)


#### layers


Conv2d = nn.Conv2d


class GroupNorm2d(nn.GroupNorm):
    """"""

    def __init__(self, num_channels, num_groups=32):
        super().__init__(num_groups, num_channels)  # , eps=1e-6


class LayerNorm2d(nn.Module):
    """"""

    def forward(self, input):
        return nn.functional.layer_norm(input, input.size()[1:])  # , eps=1e-6


#### models


class SimpleCnn(nn.Sequential):
    """
    As for the hyperparams of ConvTranpose2d, refere to
        https://blog.csdn.net/pl3329750233/article/details/130283512.
    """

    conv_types = {
        0: nn.Conv2d,
        1: lambda *a, **k: nn.ConvTranspose2d(*a, **k, output_padding=1),
    }

    def __init__(self, in_channel, channels, kernels, strides, tranposes):
        layers = []
        ci = in_channel
        for c, k, s, t in zip(channels, kernels, strides, tranposes):
            layers += [
                self.conv_types[t](ci, c, k, stride=s, padding=k // 2),
                nn.ReLU(inplace=True),
            ]
            ci = c
        super().__init__(*layers)


class ResNet(nn.Sequential):
    """Without GAP and classification head."""

    ARCH_DICT = {
        10: dict(block=ptvm.resnet.BasicBlock, layers=[1, 1, 1, 1]),
        18: dict(block=ptvm.resnet.BasicBlock, layers=[2, 2, 2, 2]),
        34: dict(block=ptvm.resnet.BasicBlock, layers=[3, 4, 6, 3]),
        50: dict(block=ptvm.resnet.Bottleneck, layers=[3, 4, 6, 3]),
    }

    def __init__(self, depth, weights="IMAGENET1K_V1", norm_layer=GroupNorm2d, dx=32):
        resnet = ptvm.ResNet(**self.ARCH_DICT[depth], norm_layer=norm_layer)
        if weights:
            weights = ptvm.__dict__[f"ResNet{depth}_Weights"][weights]
            resnet.load_state_dict(weights.get_state_dict(progress=True), strict=False)
        layers = [
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        ]
        if dx == 32:
            ...
        elif dx == 16:
            layers.pop(3)
        elif dx == 8:
            layers[0] = nn.Conv2d(3, 64, 5, stride=1, padding=2, bias=False)
            layers[0].weight.data[...] = resnet.conv1.weight[:, :, 1:6, 1:6]
            layers.pop(3)
        elif dx == 4:
            layers[0] = nn.Conv2d(3, 64, 5, stride=1, padding=2, bias=False)
            layers[0].weight.data[...] = resnet.conv1.weight[:, :, 1:6, 1:6]
            layers.pop(3)
            resnet.layer4[0].conv1.stride = (1, 1)
            resnet.layer4[0].conv1.padding = (2, 2)
            resnet.layer4[0].conv1.dilation = (2, 2)
            resnet.layer4[0].downsample[0].stride = (1, 1)
        else:
            raise "NotImplemented"
        super().__init__(*layers)

    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                for param in m.parameters():
                    param.requires_grad = False
        self[0].weight.requires_grad = True
        return self


class PositionEmbedding2d(nn.Module):
    """nn.Embedding"""

    def __init__(self, num_channel):
        super().__init__()
        self.project = nn.Linear(2, num_channel)

    @staticmethod
    def embed(shape, value_range=(-1.0, 1.0)) -> pt.Tensor:
        """Create a tensor with equidistant entries. Return a tensor in shape (*shape, len(shape))."""
        s = [np.linspace(*value_range, _) for _ in shape]
        pe = np.stack(np.meshgrid(*s, sparse=False, indexing="ij"), axis=-1)
        return pt.tensor(pe, requires_grad=False)

    def forward(self, input: pt.Tensor) -> pt.Tensor:
        """in (b,h,w,c), out (b,h,w,c)"""
        assert len(input.shape) == 4
        emb = self.embed(input.shape[1:3], [-1, 1]).to(input)[None, ...]
        output = input + self.project(emb)
        return output


class MLP(nn.Sequential):
    """depth = len(mid_channels) + 1"""

    def __init__(
        self,
        in_channel: int,
        mid_channels: list,
        out_channel: int = None,
        act_type=nn.ReLU,
        norm=None,
    ):
        out_channel = out_channel or in_channel  # TODO XXX split into LayerNorm + MLP
        layers = []
        if norm == "pre":
            layers += [nn.LayerNorm(in_channel)]  # , eps=1e-6
        layers += [nn.Linear(in_channel, mid_channels[0])]
        for i, mc in enumerate(mid_channels[1:]):
            layers += [act_type(), nn.Linear(mid_channels[i], mc)]
        layers += [act_type(), nn.Linear(mid_channels[-1], out_channel)]
        if norm == "post":
            layers += [nn.LayerNorm(out_channel)]  # , eps=1e-6
        super().__init__(*layers)


class BboxEmbedding(nn.Module):
    """State init that encodes bounding box coordinates as conditional input."""

    def __init__(self, mid_dim, out_dim, prepend_background=True, center_of_mass=False):
        super().__init__()
        self.prepend_background = prepend_background
        self.center_of_mass = center_of_mass
        self.background_value = 0.0
        self.embed_transform = MLP(4, [mid_dim], out_dim)

    def forward(self, input) -> pt.Tensor:
        """in (b,n,4), out (b,n+1,c)"""
        if self.prepend_background:  # add background box [0, 0, 0, 0] at the beginning.
            box_bg = pt.full([input.size(0), 1, 4], self.background_value).to(input)
            input = pt.concat([box_bg, input], dim=1)
        if self.center_of_mass:
            y_pos = (input[:, :, 0] + input[:, :, 2]) / 2
            x_pos = (input[:, :, 1] + input[:, :, 3]) / 2
            input = pt.stack([y_pos, x_pos], dim=-1).to(input)
        slots = self.embed_transform(input)  # (b,n+1,c)
        return slots


class GRUCell(nn.Module):
    """Cause nn.GRUCell is different from that of Flax."""

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pir = nn.Linear(input_size, hidden_size, bias)
        self.piz = nn.Linear(input_size, hidden_size, bias)
        self.phr = nn.Linear(hidden_size, hidden_size, False)
        self.phz = nn.Linear(hidden_size, hidden_size, False)
        self.pin = nn.Linear(input_size, hidden_size, bias)
        self.phn = nn.Linear(hidden_size, hidden_size, bias)

    def forward(self, input, hx):
        r = nn.functional.sigmoid(self.pir(input) + self.phr(hx))
        z = nn.functional.sigmoid(self.piz(input) + self.phz(hx))
        n = nn.functional.tanh(self.pin(input) + r * self.phn(hx))
        hx_new = (1 - z) * n + z * hx
        return hx_new


class SlotAttention(nn.Module):
    """Based on Inverted Scaled Dot Product Attention, which is an RNN block.

    Here, qi_dim=?, ki_dim==vi_dim; qo_dim==ko_vim, vo_dim=?.
    But for ``torch.nn.MultiHeadAttention``: embed_dim==qi_dim==vo_dim

    So strange that ``nn.GRUCell`` gets bad ari/train,
    while ari/val, ari_fg/train and ari_fg/val are all good.
    """

    def __init__(self, num_iter, qi_dim, kvi_dim, qko_dim=None, vo_dim=None):
        super().__init__()
        vo_dim = vo_dim or qi_dim
        qko_dim = qko_dim or qi_dim
        # assert qi_dim == vo_dim
        self.num_iter = num_iter
        self.norm_q = nn.LayerNorm(qi_dim)  # , eps=1e-6
        self.norm_kv = nn.LayerNorm(kvi_dim)  # , eps=1e-6
        self.proj_q = nn.Linear(qi_dim, qko_dim, bias=False)
        self.proj_k = nn.Linear(kvi_dim, qko_dim, bias=False)
        self.proj_v = nn.Linear(kvi_dim, vo_dim, bias=False)
        self.gru = GRUCell(vo_dim, vo_dim)  # nn.GRUCell

    def forward(self, query, input) -> pt.Tensor:  # TODO 加上最新的改进
        b, n, _ = query.shape
        x = self.norm_kv(input)
        k = self.proj_k(x)
        v = self.proj_v(x)
        for _ in range(self.num_iter):
            q = self.norm_q(query)
            q = self.proj_q(q)
            update = self.inverted_sdpa(q, k, v)  # TODO multi-head; then fuse with fc
            query = self.gru(  # TODO XXX TODO XXX TODO XXX 原来这里弄错了位置 XXX
                update.view(b * n, -1),
                query.view(b * n, -1),
            ).view(b, n, -1)
        return query

    @staticmethod
    def inverted_sdpa(q, k, v, eps=1e-8):
        # temperature normalization
        q = q / q.size(-1) ** 0.5
        # *inverted: softmax over query
        a = pt.einsum("bmc,bnc->bmn", q, k).softmax(dim=1)
        # *renormalize keys
        a = a / (a.sum(dim=2, keepdim=True) + eps)
        # aggregate values
        return pt.einsum("bmn,bnc->bmc", a, v)


class MultiheadAttention(nn.Module):
    """nn.MultiheadAttention: ``embed_dim==q_dim==out_dim``"""

    def __init__(
        self,
        embed_dim,
        num_head=1,
        dropout=0.0,
        q_dim=None,
        kv_dim=None,
        out_dim=None,
        eps=1e-8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.head_dim = embed_dim // num_head
        assert embed_dim % num_head == 0
        self.dropout = dropout
        self.eps = eps

        q_dim = q_dim or embed_dim
        kv_dim = kv_dim or q_dim
        out_dim = out_dim or q_dim

        self.proj_q = nn.Linear(q_dim, embed_dim, bias=False)
        self.proj_k = nn.Linear(kv_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(kv_dim, embed_dim, bias=False)
        self.proj_o = nn.Linear(embed_dim, out_dim)

        """ self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_q.weight)
        nn.init.xavier_uniform_(self.proj_k.weight)
        nn.init.xavier_uniform_(self.proj_v.weight)
        nn.init.constant_(self.proj_o.bias, 0.0) """

    def forward(self, query, key, value, return_attent=False):
        assert len(query.shape) == len(key.shape) == len(value.shape) == 3
        assert key.size(1) == value.size(1)

        q = self.proj_q(query)
        k = self.proj_k(key)
        v = self.proj_v(value)

        if return_attent:
            q = rearrange(q, "b n (h d) -> b n h d", h=self.num_head, d=self.head_dim)
            k = rearrange(k, "b n (h d) -> b n h d", h=self.num_head, d=self.head_dim)
            v = rearrange(v, "b n (h d) -> b n h d", h=self.num_head, d=self.head_dim)

            q = q / q.size(-1) ** 0.5
            attent = pt.einsum("bqhd,bkhd->bqkh", q, k)
            attent = pt.softmax(attent, dim=2)
            if self.training and self.dropout > 0:
                attent = pt.dropout(attent, p=self.dropout)
            output = pt.einsum("bqvh,bvhd->bqhd", attent, v)

            output = rearrange(output, "b n h d -> b n (h d)")
            output = self.proj_o(output)

            attent = attent.mean(dim=3)

        else:
            q = rearrange(q, "b n (h d) -> b h n d", h=self.num_head, d=self.head_dim)
            k = rearrange(k, "b n (h d) -> b h n d", h=self.num_head, d=self.head_dim)
            v = rearrange(v, "b n (h d) -> b h n d", h=self.num_head, d=self.head_dim)

            self_dropout = self.dropout if self.training else 0
            output = nn.functional.scaled_dot_product_attention(
                q, k, v, None, self_dropout, False
            )

            output = rearrange(output, "b h n d -> b n (h d)")
            output = self.proj_o(output)

            attent = None

        return output, attent


class TransformBlock(nn.Module):
    """So strange that ``nn.TransformerEncoderLayer`` gets bad ari/train,
    while ari/val, ari_fg/train and ari_fg/val are all good.

    class TransformBlock(nn.TransformerEncoderLayer):
        def __init__(
            self, embed_dim=128, num_head=4, ffn_dim=256, act_fn=pt.relu, dropout=0.01, pre_norm=False,
        ):
            super().__init__(
                d_model=embed_dim, nhead=num_head, dim_feedforward=ffn_dim, dropout=dropout,
                activation=act_fn, batch_first=True, norm_first=pre_norm,
            )
    """

    def __init__(
        self,
        embed_dim=128,
        num_head=4,
        dropout=0.0,
        q_dim=256,
        ffn_dim=256,
        pre_norm=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.attn = MultiheadAttention(embed_dim, num_head, dropout, q_dim)
        self.norm1 = nn.LayerNorm(q_dim)  # , eps=1e-6

        self.ffn = MLP(q_dim, [ffn_dim], q_dim)
        self.norm2 = nn.LayerNorm(q_dim)  # , eps=1e-6

    def forward(self, query: pt.Tensor, input=None) -> pt.Tensor:
        if self.pre_norm:
            x = self.norm1(query)
            x = query + self.attn(x, x, x)[0]
            y = x
            z = self.norm2(y)
            z = y + self.ffn(z)
        else:
            x = query + self.attn(query, query, query)[0]
            x = self.norm1(x)  # TODO move ``query+`` out of ``self.norm1``
            y = x
            z = y + self.ffn(y)
            z = self.norm2(z)
        return z


class Savi(nn.Module):
    """Slot Attention for Video (SAVi)."""

    def __init__(
        self,
        enc_backbone=SimpleCnn(
            3, [32, 32, 32, 32], [5, 5, 5, 5], [1, 1, 1, 1], [0, 0, 0, 0]
        ),
        enc_posit_embed=PositionEmbedding2d(32),
        enc_project=MLP(32, [64], 32, norm="pre"),
        proc_initialize=BboxEmbedding(
            256, 128, prepend_background=True, center_of_mass=False
        ),
        proc_correct=SlotAttention(1, 128, 32, 128, 128),
        # proc_predict=TransformBlock(128, 4, q_dim=128, ffn_dim=256, pre_norm=False),
        proc_predict=TransformBlock(128, 4, ffn_dim=256, pre_norm=False),
        dec_resolut=(8, 8),
        dec_posit_embed=PositionEmbedding2d(128),
        dec_backbone=SimpleCnn(
            128, [64, 64, 64, 64], [5, 5, 5, 5], [2, 2, 2, 1], [1, 1, 1, 0]
        ),
        dec_readout=Conv2d(64, 4, 1),
    ):
        super().__init__()
        self.encoder = nn.ModuleDict(
            dict(
                backbone=enc_backbone, posit_embed=enc_posit_embed, project=enc_project
            )
        )
        self.processor = nn.ModuleDict(
            dict(initialize=proc_initialize, correct=proc_correct, predict=proc_predict)
        )
        self.dec_resolut = dec_resolut
        self.decoder = nn.ModuleDict(
            dict(
                posit_embed=dec_posit_embed, backbone=dec_backbone, readout=dec_readout
            )
        )
        self.init_weights()  # seems not that necessary

    def init_weights(self):  # TODO XXX 真的发挥作用了吗？检查！
        for k, m in self.named_modules():
            if isinstance(m, nn.modules.conv._ConvNd):
                print(k)
                lecun_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                print(k)
                lecun_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, frames, condition):
        """
        frames: in shape (b,t,c,h,w)
        condition: in shape (b,t,n,4)
        """
        b, t = frames.shape[:2]

        frames_ = rearrange(frames, "b t c h w -> (b t) c h w")
        encoded_ = self._encode(frames_)
        encoded = rearrange(encoded_, "(b t) h w c -> b t (h w) c", b=b, t=t)

        states = self._process(encoded, condition)
        states_ = rearrange(states, "t b n c -> (b t) n c")

        images_, segment_, logit_ = self._decode(states_)
        images = rearrange(images_, "(b t) c h w -> b t c h w", b=b, t=t)
        segment = rearrange(segment_, "(b t) c h w -> b t c h w", b=b, t=t).squeeze(2)

        return images, segment, states_, logit_

    def _encode(self, input):
        """
        input: in shape (bt,c,h,w)
        output: in shape (bt,h,w,c)
        """
        x = self.encoder.backbone(input)
        x = x.permute(0, 2, 3, 1)

        x = self.encoder.posit_embed(x)
        output = self.encoder.project(x)

        return output

    def _process(self, encoded, condition):
        """
        encoded: in shape (b,t,n,c)
        condition: in shape (b,t,m,4)
        corrects: in shape [(b,m,c)]*t
        """
        states = []

        state = self.processor.initialize(condition[:, 0, ...])  # (b,n+1,c)
        for t in range(encoded.size(1)):
            corrected_state = self.processor.correct(state, encoded[:, t, ...])
            predicted_state = self.processor.predict(corrected_state)
            state = predicted_state

            states.append(corrected_state)

        return states

    def _decode(self, slots):
        """
        slots: in shape (bt,n,c)
        image: in shape (bt,3,h,w)
        segment: in shape (bt,h,w)
        """
        b, n, _ = slots.shape
        x = slots.view(b * n, 1, 1, -1).repeat(1, *self.dec_resolut, 1)  # (b*n,8,8,c)

        x = self.decoder.posit_embed(x).permute(0, 3, 1, 2)  # (b*n,c,8,8)
        x = self.decoder.backbone(x)  # (b*n,c,h,w)
        _, _, h, w = x.shape
        logit = self.decoder.readout(x).view(b, n, -1, h, w)

        image = (logit[:, :, 3:].softmax(dim=1) * logit[:, :, :3]).sum(dim=1)
        segment = logit[:, :, 3:].argmax(dim=1)  # (b,1,h,w)

        return image, segment, logit


class ModelHelper(nn.Module):
    """Wrap a model so that its inputs and outputs are in expected dict struct."""

    def __init__(self, model, in_keys: dict or list, out_keys: list, device: str):
        """
        - in_keys: dict or list.
            If keys in batch mismatches with keys in model.forward, use dict, ie, {key_in_batch: key_in_forward};
            If not, use list.
        - out_keys: list
        """
        super().__init__()
        assert isinstance(in_keys, (dict, list, tuple))
        assert isinstance(out_keys, (list, tuple))
        self.model = model
        self.in_keys = in_keys if isinstance(in_keys, dict) else {_: _ for _ in in_keys}
        self.out_keys = out_keys
        self.device = pt.device(device)

    def forward(self, input: dict) -> dict:
        input2 = {k: input[v].to(self.device) for k, v in self.in_keys.items()}
        output = self.model(**input2)
        if not isinstance(output, (list, tuple)):
            output = [output]
        assert len(self.out_keys) == len(output)
        output2 = dict(zip(self.out_keys, output))
        return output2


[register_module(_) for _ in locals().values() if isinstance(_, type)]
