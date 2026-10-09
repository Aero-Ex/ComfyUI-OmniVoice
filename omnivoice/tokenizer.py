import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.audio
import comfy.ops
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention


def _conv_output_length(length, kernel, stride, padding):
    return (length + 2 * padding - kernel) // stride + 1


class Snake1d(nn.Module):
    def __init__(self, channels, device=None, dtype=None):
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, channels, 1, device=device, dtype=dtype))

    def forward(self, x):
        shape = x.shape
        x = x.reshape(shape[0], shape[1], -1)
        x = x + (comfy.ops.cast_to_input(self.alpha, x) + 1e-9).reciprocal() * torch.sin(comfy.ops.cast_to_input(self.alpha, x) * x).pow(2)
        return x.reshape(shape)


class DacResidualUnit(nn.Module):
    def __init__(self, dim, dilation=1, device=None, dtype=None, operations=None):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dim, device=device, dtype=dtype)
        self.conv1 = operations.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad, device=device, dtype=dtype)
        self.snake2 = Snake1d(dim, device=device, dtype=dtype)
        self.conv2 = operations.Conv1d(dim, dim, kernel_size=1, device=device, dtype=dtype)

    def forward(self, x):
        out = self.conv1(self.snake1(x))
        out = self.conv2(self.snake2(out))
        padding = (x.shape[-1] - out.shape[-1]) // 2
        if padding > 0:
            x = x[..., padding:-padding]
        return x + out


class DacEncoderBlock(nn.Module):
    def __init__(self, stride, dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.res_unit1 = DacResidualUnit(dim // 2, dilation=1, device=device, dtype=dtype, operations=operations)
        self.res_unit2 = DacResidualUnit(dim // 2, dilation=3, device=device, dtype=dtype, operations=operations)
        self.res_unit3 = DacResidualUnit(dim // 2, dilation=9, device=device, dtype=dtype, operations=operations)
        self.snake1 = Snake1d(dim // 2, device=device, dtype=dtype)
        self.conv1 = operations.Conv1d(
            dim // 2, dim, kernel_size=2 * stride, stride=stride,
            padding=math.ceil(stride / 2), device=device, dtype=dtype,
        )

    def forward(self, x):
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.snake1(self.res_unit3(x))
        return self.conv1(x)


class DacEncoder(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        encoder_hidden = config.get("encoder_hidden_size", 64)
        strides = config.get("downsampling_ratios", [8, 5, 4, 2, 3])
        hidden_size = config.get("hidden_size", 256)

        self.conv1 = operations.Conv1d(1, encoder_hidden, kernel_size=7, padding=3, device=device, dtype=dtype)
        blocks = []
        dim = encoder_hidden
        for stride in strides:
            dim *= 2
            blocks.append(DacEncoderBlock(stride, dim, device=device, dtype=dtype, operations=operations))
        self.block = nn.ModuleList(blocks)
        self.snake1 = Snake1d(dim, device=device, dtype=dtype)
        self.conv2 = operations.Conv1d(dim, hidden_size, kernel_size=3, padding=1, device=device, dtype=dtype)

    def conv_output_length(self, length):
        layers = [self.conv1] + [b.conv1 for b in self.block] + [self.conv2]
        for layer in layers:
            w = layer.weight if not hasattr(layer, "parametrizations") else layer.parametrizations.weight.original1
            length = _conv_output_length(length, w.shape[-1], layer.stride[0], layer.padding[0])
        return length

    def forward(self, x):
        x = self.conv1(x)
        for block in self.block:
            x = block(x)
        x = self.snake1(x)
        return self.conv2(x)


class DacDecoderBlock(nn.Module):
    def __init__(self, stride, input_dim, output_dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.snake1 = Snake1d(input_dim, device=device, dtype=dtype)
        self.conv_t1 = operations.ConvTranspose1d(
            input_dim, output_dim, kernel_size=2 * stride, stride=stride,
            padding=math.ceil(stride / 2), output_padding=stride % 2, device=device, dtype=dtype,
        )
        self.res_unit1 = DacResidualUnit(output_dim, dilation=1, device=device, dtype=dtype, operations=operations)
        self.res_unit2 = DacResidualUnit(output_dim, dilation=3, device=device, dtype=dtype, operations=operations)
        self.res_unit3 = DacResidualUnit(output_dim, dilation=9, device=device, dtype=dtype, operations=operations)

    def forward(self, x):
        x = self.snake1(x)
        x = self.conv_t1(x)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        return self.res_unit3(x)


class DacDecoder(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        input_channels = config.get("hidden_size", 256)
        channels = config.get("decoder_hidden_size", 1024)
        strides = config.get("upsampling_ratios", [8, 5, 4, 2, 3])

        self.conv1 = operations.Conv1d(input_channels, channels, kernel_size=7, padding=3, device=device, dtype=dtype)
        blocks = []
        dim = channels
        for stride in strides:
            blocks.append(DacDecoderBlock(stride, dim, dim // 2, device=device, dtype=dtype, operations=operations))
            dim //= 2
        self.block = nn.ModuleList(blocks)
        self.snake1 = Snake1d(dim, device=device, dtype=dtype)
        self.conv2 = operations.Conv1d(dim, 1, kernel_size=7, padding=3, device=device, dtype=dtype)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.block:
            x = block(x)
        x = self.snake1(x)
        return self.conv2(x)


class EuclideanCodebook(nn.Module):
    def __init__(self, codebook_size, codebook_dim, device=None, dtype=None):
        super().__init__()
        self.codebook_size = codebook_size
        self.register_buffer("inited", torch.tensor([True], device=device, dtype=dtype))
        self.register_buffer("cluster_size", torch.zeros(codebook_size, device=device, dtype=dtype))
        self.register_buffer("embed", torch.zeros(codebook_size, codebook_dim, device=device, dtype=dtype))
        self.register_buffer("embed_avg", torch.zeros(codebook_size, codebook_dim, device=device, dtype=dtype))

    def quantize(self, x):
        embed = comfy.ops.cast_to_input(self.embed, x).t()
        scaled = x.pow(2).sum(1, keepdim=True)
        dist = -(scaled - 2 * x @ embed + embed.pow(2).sum(0, keepdim=True))
        return dist.max(dim=-1).indices

    def encode(self, x):
        shape = x.shape
        return self.quantize(x.reshape((-1, shape[-1]))).view(*shape[:-1])

    def decode(self, indices):
        return F.embedding(indices, self.embed.to(indices.device))


class VectorQuantization(nn.Module):
    def __init__(self, hidden_size, codebook_size, codebook_dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.codebook = EuclideanCodebook(codebook_size, codebook_dim, device=device, dtype=dtype)
        self.project_in = operations.Linear(hidden_size, codebook_dim, device=device, dtype=dtype)
        self.project_out = operations.Linear(codebook_dim, hidden_size, device=device, dtype=dtype)

    def encode(self, x):
        return self.codebook.encode(self.project_in(x.permute(0, 2, 1)))

    def decode(self, indices):
        return self.project_out(self.codebook.decode(indices)).permute(0, 2, 1)


class ResidualVectorQuantization(nn.Module):
    def __init__(self, num_quantizers, hidden_size, codebook_size, codebook_dim, device=None, dtype=None, operations=None):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.quantizers = nn.ModuleList([
            VectorQuantization(hidden_size, codebook_size, codebook_dim, device=device, dtype=dtype, operations=operations)
            for _ in range(num_quantizers)
        ])

    def encode(self, embeddings):
        residual = embeddings
        indices = []
        for quantizer in self.quantizers:
            embed_ind = quantizer.encode(residual)
            quantized = quantizer.decode(embed_ind)
            residual = residual - quantized
            indices.append(embed_ind)
        return torch.stack(indices)

    def decode(self, codes):
        quantized_out = None
        for i, indices in enumerate(codes):
            quantized = self.quantizers[i].decode(indices)
            quantized_out = quantized if quantized_out is None else quantized_out + quantized.to(quantized_out.device)
        return quantized_out


class SemanticResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3, device=None, dtype=None, operations=None):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.activation = nn.ELU()
        self.conv1 = operations.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=padding, dilation=dilation, bias=False, device=device, dtype=dtype)
        self.conv2 = operations.Conv1d(out_channels, out_channels, kernel_size=1, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        return x + self.conv2(out)


class SemanticEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilations, device=None, dtype=None, operations=None):
        super().__init__()
        self.res_units = nn.ModuleList([
            SemanticResidualUnit(in_channels, in_channels, dilation, device=device, dtype=dtype, operations=operations)
            for dilation in dilations
        ])
        kernel = 3 if stride == 1 else 2 * stride
        padding = (kernel - 1) // 2
        self.conv = operations.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding, bias=True, device=device, dtype=dtype)

    def forward(self, x):
        for unit in self.res_units:
            x = unit(x)
        return self.conv(x)


class SemanticEncoder(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        semantic_hidden = config["semantic_hidden_size"]
        strides = config["strides"]
        ratios = config["channel_ratios"]
        dilations = config["block_dilations"]
        kernel_size = config["kernel_size"]

        self.conv = operations.Conv1d(semantic_hidden, semantic_hidden, kernel_size, 1, kernel_size // 2, bias=False, device=device, dtype=dtype)
        blocks = []
        in_channels = semantic_hidden
        for i, stride in enumerate(strides):
            out_channels = int(semantic_hidden * ratios[i])
            blocks.append(SemanticEncoderBlock(in_channels, out_channels, stride, dilations, device=device, dtype=dtype, operations=operations))
            in_channels = out_channels
        self.conv_blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.conv(x)
        for block in self.conv_blocks:
            x = block(x)
        return x


class SemanticDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilations, device=None, dtype=None, operations=None):
        super().__init__()
        if stride == 1:
            self.conv = operations.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, device=device, dtype=dtype)
        else:
            self.conv = operations.ConvTranspose1d(
                in_channels, out_channels, 2 * stride, stride, (stride + 1) // 2,
                output_padding=1 if stride % 2 == 1 else 0, bias=False, device=device, dtype=dtype,
            )
        self.res_units = nn.ModuleList([
            SemanticResidualUnit(out_channels, out_channels, dilation, device=device, dtype=dtype, operations=operations)
            for dilation in dilations
        ])

    def forward(self, x):
        x = self.conv(x)
        for unit in self.res_units:
            x = unit(x)
        return x


class SemanticDecoder(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        semantic_hidden = config["semantic_hidden_size"]
        strides = config["strides"]
        ratios = config["channel_ratios"]
        dilations = config["block_dilations"]
        kernel_size = config["kernel_size"]

        self.conv1 = operations.Conv1d(semantic_hidden, int(semantic_hidden * ratios[0]), kernel_size, 1, kernel_size // 2, bias=False, device=device, dtype=dtype)
        blocks = []
        for i, stride in enumerate(strides):
            in_channels = int(semantic_hidden * ratios[i])
            if i < len(ratios) - 1:
                out_channels = int(semantic_hidden * ratios[i + 1])
            else:
                out_channels = semantic_hidden
            blocks.append(SemanticDecoderBlock(in_channels, out_channels, stride, dilations, device=device, dtype=dtype, operations=operations))
        self.conv_blocks = nn.ModuleList(blocks)
        self.conv2 = operations.Conv1d(semantic_hidden, semantic_hidden, kernel_size, 1, kernel_size // 2, bias=False, device=device, dtype=dtype)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.conv_blocks:
            x = block(x)
        return self.conv2(x)


class HubertGroupNormConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv = operations.Conv1d(in_dim, out_dim, kernel_size=kernel, stride=stride, bias=False, device=device, dtype=dtype)
        self.activation = nn.GELU()
        self.layer_norm = nn.GroupNorm(num_groups=out_dim, num_channels=out_dim, affine=True, device=device, dtype=dtype)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer_norm(x)
        return self.activation(x)


class HubertNoLayerNormConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv = operations.Conv1d(in_dim, out_dim, kernel_size=kernel, stride=stride, bias=False, device=device, dtype=dtype)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.conv(x))


class HubertFeatureEncoder(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        conv_dim = config["conv_dim"]
        conv_kernel = config["conv_kernel"]
        conv_stride = config["conv_stride"]
        layers = [HubertGroupNormConvLayer(1, conv_dim[0], conv_kernel[0], conv_stride[0], device=device, dtype=dtype, operations=operations)]
        for i in range(1, len(conv_dim)):
            layers.append(HubertNoLayerNormConvLayer(conv_dim[i - 1], conv_dim[i], conv_kernel[i], conv_stride[i], device=device, dtype=dtype, operations=operations))
        self.conv_layers = nn.ModuleList(layers)

    def forward(self, x):
        x = x[:, None]
        for layer in self.conv_layers:
            x = layer(x)
        return x


class HubertFeatureProjection(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config["conv_dim"][-1], eps=config.get("layer_norm_eps", 1e-5), device=device, dtype=dtype)
        self.projection = operations.Linear(config["conv_dim"][-1], config["hidden_size"], device=device, dtype=dtype)
        self.dropout = nn.Identity()

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.projection(x)
        return self.dropout(x)


class HubertPositionalConvEmbedding(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.conv = operations.Conv1d(
            config["hidden_size"], config["hidden_size"],
            kernel_size=config["num_conv_pos_embeddings"],
            padding=config["num_conv_pos_embeddings"] // 2,
            groups=config["num_conv_pos_embedding_groups"],
            device=device, dtype=dtype,
        )
        self.conv = torch.nn.utils.parametrizations.weight_norm(self.conv, name="weight", dim=2)
        self.num_pad_remove = 1 if config["num_conv_pos_embeddings"] % 2 == 0 else 0
        self.activation = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        if self.num_pad_remove > 0:
            x = x[:, :, :-self.num_pad_remove]
        x = self.activation(x)
        return x.transpose(1, 2)


class HubertAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, device=None, dtype=None, operations=None):
        super().__init__()
        self.comfy_attention = ComfyAttention()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = operations.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.k_proj = operations.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.v_proj = operations.Linear(hidden_size, hidden_size, device=device, dtype=dtype)
        self.out_proj = operations.Linear(hidden_size, hidden_size, device=device, dtype=dtype)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        xq = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xk = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xv = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = AttentionTensorContainer(xq), AttentionTensorContainer(xk), AttentionTensorContainer(xv)
        out = optimized_attention(q, k, v, self.num_heads, skip_reshape=True, preferred_attention=self.comfy_attention)
        return self.out_proj(out)


class HubertFeedForward(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.intermediate_dense = operations.Linear(config["hidden_size"], config["intermediate_size"], device=device, dtype=dtype)
        self.intermediate_act_fn = nn.GELU()
        self.output_dense = operations.Linear(config["intermediate_size"], config["hidden_size"], device=device, dtype=dtype)

    def forward(self, x):
        x = self.intermediate_act_fn(self.intermediate_dense(x))
        return self.output_dense(x)


class HubertEncoderLayer(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.attention = HubertAttention(config["hidden_size"], config["num_attention_heads"], device=device, dtype=dtype, operations=operations)
        self.layer_norm = nn.LayerNorm(config["hidden_size"], eps=config.get("layer_norm_eps", 1e-5), device=device, dtype=dtype)
        self.feed_forward = HubertFeedForward(config, device=device, dtype=dtype, operations=operations)
        self.final_layer_norm = nn.LayerNorm(config["hidden_size"], eps=config.get("layer_norm_eps", 1e-5), device=device, dtype=dtype)

    def forward(self, x):
        residual = x
        x = self.attention(x)
        x = residual + x
        x = self.layer_norm(x)
        x = x + self.feed_forward(x)
        return self.final_layer_norm(x)


class HubertEncoder(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.pos_conv_embed = HubertPositionalConvEmbedding(config, device=device, dtype=dtype, operations=operations)
        self.layer_norm = nn.LayerNorm(config["hidden_size"], eps=config.get("layer_norm_eps", 1e-5), device=device, dtype=dtype)
        self.dropout = nn.Identity()
        self.layers = nn.ModuleList([
            HubertEncoderLayer(config, device=device, dtype=dtype, operations=operations)
            for _ in range(config["num_hidden_layers"])
        ])

    def forward(self, x):
        x = x + self.pos_conv_embed(x)
        x = self.dropout(self.layer_norm(x))
        hidden_states = [x]
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)
        return hidden_states


class SemanticModel(nn.Module):
    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.feature_extractor = HubertFeatureEncoder(config, device=device, dtype=dtype, operations=operations)
        self.feature_projection = HubertFeatureProjection(config, device=device, dtype=dtype, operations=operations)
        self.encoder = HubertEncoder(config, device=device, dtype=dtype, operations=operations)

    def forward(self, x):
        x = self.feature_extractor(x).transpose(1, 2)
        x = self.feature_projection(x)
        return self.encoder(x)


class OmniVoiceTokenizer(nn.Module):
    """HiggsAudioV2 neural audio codec: HuBERT semantic features + DAC acoustic codec + RVQ."""

    def __init__(self, config, device=None, dtype=None, operations=None):
        super().__init__()
        self.config = config
        self.device = device
        self.dtype = dtype
        self.operations = operations

        acoustic_config = config["acoustic_model_config"]
        semantic_config = config["semantic_model_config"]

        self.sample_rate = config.get("sample_rate", 24000)
        self.semantic_sample_rate = config.get("semantic_sample_rate", 16000)
        self.semantic_downsample_factor = config.get("semantic_downsample_factor", 2)
        self.num_quantizers = config.get("num_quantizers", 8)
        self.hidden_size = acoustic_config.get("hidden_size", 256) + semantic_config.get("hidden_size", 768)
        self.codebook_size = config.get("codebook_size", 1024)
        self.codebook_dim = config.get("codebook_dim", 64)
        self.pad = acoustic_config.get("hop_length", 960) // 2

        ops = operations
        self.acoustic_encoder = DacEncoder(acoustic_config, device=device, dtype=dtype, operations=ops)
        self.acoustic_decoder = DacDecoder(acoustic_config, device=device, dtype=dtype, operations=ops)
        self.encoder_semantic = SemanticEncoder({
            "semantic_hidden_size": semantic_config.get("hidden_size", 768),
            "strides": config.get("strides", [1, 1]),
            "channel_ratios": config.get("channel_ratios", [1, 1]),
            "block_dilations": config.get("block_dilations", [1, 1]),
            "kernel_size": config.get("kernel_size", 3),
        }, device=device, dtype=dtype, operations=ops)
        self.decoder_semantic = SemanticDecoder({
            "semantic_hidden_size": semantic_config.get("hidden_size", 768),
            "strides": config.get("strides", [1, 1]),
            "channel_ratios": config.get("channel_ratios", [1, 1]),
            "block_dilations": config.get("block_dilations", [1, 1]),
            "kernel_size": config.get("kernel_size", 3),
        }, device=device, dtype=dtype, operations=ops)
        self.semantic_model = SemanticModel(semantic_config, device=device, dtype=dtype, operations=ops)
        self.fc = ops.Linear(self.hidden_size, self.hidden_size, device=device, dtype=dtype)
        self.fc1 = ops.Linear(self.hidden_size, semantic_config.get("hidden_size", 768), device=device, dtype=dtype)
        self.fc2 = ops.Linear(self.hidden_size, acoustic_config.get("hidden_size", 256), device=device, dtype=dtype)
        self.quantizer = ResidualVectorQuantization(
            self.num_quantizers, self.hidden_size, self.codebook_size, self.codebook_dim,
            device=device, dtype=dtype, operations=ops,
        )

    @property
    def frame_rate(self):
        return math.ceil(self.sample_rate / (self.pad * 2))

    def extract_semantic(self, input_values):
        if self.sample_rate != self.semantic_sample_rate:
            input_values = comfy.audio.resample(input_values, self.sample_rate, self.semantic_sample_rate)
        input_values = F.pad(input_values[:, 0, :], (160, 160))
        hidden_states = self.semantic_model(input_values)
        stacked = torch.stack([h.to(input_values.device) for h in hidden_states], dim=1)
        semantic_features = stacked.mean(dim=1)
        if self.semantic_downsample_factor > 1:
            semantic_features = semantic_features[:, ::self.semantic_downsample_factor, :]
        return semantic_features

    def encode(self, input_values):
        """Encode mono waveform [batch, 1, samples] to codes [batch, num_quantizers, length]."""
        e_semantic = self.extract_semantic(input_values).detach().transpose(1, 2)
        e_semantic = self.encoder_semantic(e_semantic)

        if self.acoustic_encoder.conv_output_length(input_values.shape[2]) != e_semantic.shape[2]:
            e_acoustic = self.acoustic_encoder(F.pad(input_values, (self.pad, self.pad)))
        else:
            e_acoustic = self.acoustic_encoder(input_values)

        embeddings = torch.cat([e_acoustic.to(e_semantic.device), e_semantic], dim=1)
        embeddings = self.fc(embeddings.transpose(1, 2)).transpose(1, 2)
        return self.quantizer.encode(embeddings).transpose(0, 1)

    def decode(self, audio_codes):
        """Decode codes [batch, num_quantizers, length] to waveform [batch, 1, samples]."""
        if audio_codes.shape[1] != self.num_quantizers and audio_codes.shape[-1] == self.num_quantizers:
            audio_codes = audio_codes.transpose(1, 2)
        quantized = self.quantizer.decode(audio_codes.transpose(0, 1))
        quantized_acoustic = self.fc2(quantized.transpose(1, 2)).transpose(1, 2)
        return self.acoustic_decoder(quantized_acoustic)

    @classmethod
    def from_state_dict(cls, sd, config, device=None, dtype=None, operations=None):
        return cls(config, device=device, dtype=dtype, operations=operations)
