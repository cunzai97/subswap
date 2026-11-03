#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LaMa（big-lama）原生 PyTorch 实现，支持 conv 半精度 + FFT 分支强制 fp32。

big-lama.pt 是 TorchScript 导出，模型的 FFC 全局分支走 torch.fft，
CUDA 半精度下 ComplexHalf 会产 NaN，所以整模型 fp16 不可用。
这里按官方 advimman/lama 的 FFCResNetGenerator 重建，权重从 TorchScript
state_dict 直接加载；FourierUnit 内部强制 fp32，其余卷积可安全半精度。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')


class FourierUnit(nn.Module):
    """官方实现，fft 段强制 fp32（CUDA ComplexHalf 不可用）。"""

    def __init__(self, in_channels, out_channels, groups=1, fft_norm='ortho'):
        super().__init__()
        self.groups = groups
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2,
                                    kernel_size=1, stride=1, padding=0,
                                    groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        in_dtype = x.dtype
        with torch.autocast(device_type='cuda', enabled=False):
            x = x.float()
            batch = x.shape[0]
            fft_dim = (-2, -1)
            ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
            ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
            ffted = ffted.view((batch, -1,) + ffted.size()[3:])
            ffted = self.conv_layer(ffted)
            ffted = self.relu(self.bn(ffted))
            ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
                0, 1, 3, 4, 2).contiguous()
            ffted = torch.complex(ffted[..., 0], ffted[..., 1])
            output = torch.fft.irfftn(ffted, s=x.shape[-2:], dim=fft_dim,
                                      norm=self.fft_norm)
        return output.to(in_dtype)


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1,
                 enable_lfu=False, **fu_kwargs):
        super().__init__()
        self.enable_lfu = enable_lfu
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity()
        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True))
        self.fu = FourierUnit(out_channels // 2, out_channels // 2,
                              groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1,
                               groups=groups, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0
        return self.conv2(x + output + xs)


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin,
                 ratio_gout, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, enable_lfu=True, padding_type='reflect', gated=False,
                 **spectral_kwargs):
        super().__init__()
        assert stride == 1 or stride == 2
        self.stride = stride
        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size, stride, padding,
                              dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size, stride, padding,
                              dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size, stride, padding,
                              dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(in_cg, out_cg, stride,
                              1 if groups == 1 else groups // 2,
                              enable_lfu, **spectral_kwargs)
        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0
        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)
            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1
        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)
        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin,
                 ratio_gout, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.Identity, padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super().__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin,
                       ratio_gout, stride, padding, dilation, groups, bias,
                       enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)
        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU,
                 dilation=1, spatial_transform_kwargs=None, inline=False,
                 **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation,
                                dilation=dilation, norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type, **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation,
                                dilation=dilation, norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type, **conv_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], \
                x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))
        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3,
                 n_blocks=18, norm_layer=nn.BatchNorm2d, padding_type='reflect',
                 activation_layer=nn.ReLU, up_norm_layer=nn.BatchNorm2d,
                 up_activation=nn.ReLU(True), init_conv_kwargs=None,
                 downsample_conv_kwargs=None, resnet_conv_kwargs=None,
                 add_out_act='sigmoid', max_features=1024):
        super().__init__()
        init_conv_kwargs = init_conv_kwargs or {}
        downsample_conv_kwargs = downsample_conv_kwargs or {}
        resnet_conv_kwargs = resnet_conv_kwargs or {}
        assert n_blocks >= 0
        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0,
                            norm_layer=norm_layer,
                            activation_layer=activation_layer,
                            **init_conv_kwargs)]
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]
        feats_num_bottleneck = min(max_features, ngf * 2 ** n_downsampling)
        for _ in range(n_blocks):
            model += [FFCResnetBlock(feats_num_bottleneck,
                                     padding_type=padding_type,
                                     activation_layer=activation_layer,
                                     norm_layer=norm_layer,
                                     **resnet_conv_kwargs)]
        model += [ConcatTupleLayer()]
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class LaMaNative(nn.Module):
    """与 TorchScript 版同接口：forward(image[0..1], mask[0/1])。"""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, image, mask):
        x = torch.cat([image * (1 - mask), mask], dim=1)
        out = self.generator(x)
        # 与 TorchScript 版一致：掩码外原样返回，掩码内用生成结果
        return out * mask + image * (1 - mask)


def build_generator():
    zero = {'ratio_gin': 0, 'ratio_gout': 0, 'enable_lfu': False}
    return FFCResNetGenerator(
        input_nc=4, output_nc=3, ngf=64, n_downsampling=3, n_blocks=18,
        add_out_act='sigmoid',
        init_conv_kwargs=dict(zero),
        downsample_conv_kwargs=dict(zero),
        resnet_conv_kwargs={'ratio_gin': 0.75, 'ratio_gout': 0.75,
                            'enable_lfu': False})


def load_native(path, device='cuda', half=False):
    ref = torch.jit.load(path, map_location='cpu')
    sd = ref.state_dict()
    gen = build_generator()
    prefix = 'model.generator.'
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = gen.load_state_dict(sub, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'state dict mismatch missing={missing[:5]} '
                           f'unexpected={unexpected[:5]}')
    model = LaMaNative(gen).to(device).eval()
    if half:
        model = model.half()
        # FFT 分支必须保持 fp32（ComplexHalf 不可用）
        for m in model.modules():
            if isinstance(m, FourierUnit):
                m.float()
    return model


if __name__ == '__main__':
    import time
    path = '/media/pan/学习/work/video-erase/models/big-lama.pt'
    torch.manual_seed(0)
    img = torch.rand(2, 3, 128, 256, device='cuda')
    msk = (torch.rand(2, 1, 128, 256, device='cuda') > 0.7).float()
    ref = torch.jit.load(path, map_location='cuda').eval()
    with torch.inference_mode():
        out_ref = ref(img, msk)
    for half in (False, True):
        m = load_native(path, 'cuda', half=half)
        with torch.inference_mode():
            out = m(img.half() if half else img, msk.half() if half else msk)
        out = out.float()
        diff = (out - out_ref).abs()
        print(f'half={half} maxdiff={diff.max().item():.2e} '
              f'mean={diff.mean().item():.2e} '
              f'finite={torch.isfinite(out).all().item()}')
    # 吞吐
    for half in (False, True):
        m = load_native(path, 'cuda', half=half)
        for h, w, bs in ((256, 1024, 4), (256, 1024, 8), (208, 1024, 8)):
            x = torch.rand(bs, 3, h, w, device='cuda')
            k = torch.zeros(bs, 1, h, w, device='cuda')
            if half:
                x, k = x.half(), k.half()
            with torch.inference_mode():
                for _ in range(3):
                    m(x, k)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(10):
                    m(x, k)
                torch.cuda.synchronize()
            print(f'half={half} {w}x{h} b{bs}: '
                  f'{(time.time()-t0)/10*1000:.1f} ms '
                  f'{(time.time()-t0)/10*bs:.1f} img/s')
