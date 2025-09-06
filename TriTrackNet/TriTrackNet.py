import torch
import random
import numpy as np

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader

from .utils.attention import scaled_dot_product_attention
from .utils.dataset import LabeledDataset
from .utils.revin import RevIN
from .utils.perturbopt import perturbopt


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.all_head_dim = self.head_dim * num_heads

        assert self.all_head_dim == embed_dim, "Embedding dimension must be divisible by number of heads"

        # Queries, Keys, Values projections
        self.query = nn.Linear(embed_dim, self.all_head_dim)
        self.key = nn.Linear(embed_dim, self.all_head_dim)
        self.value = nn.Linear(embed_dim, self.all_head_dim)

        # Output projection
        self.fc_out = nn.Linear(self.all_head_dim, embed_dim)

    def forward(self, query, key, value):
        N = query.shape[0]
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Split into multiple heads
        Q = Q.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (N, heads, seq_len, head_dim)
        K = K.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        energy = torch.einsum("nhqd,nhkd->nhqk", [Q, K])  # (N, heads, seq_len_q, seq_len_k)
        attention = torch.softmax(energy / (self.head_dim ** (1 / 2)), dim=-1)

        out = torch.einsum("nhqk,nhvd->nhqd", [attention, V])  # (N, heads, seq_len_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(N, -1, self.all_head_dim)  # (N, seq_len_q, all_head_dim)

        out = self.fc_out(out)  # (N, seq_len_q, embed_dim)
        return out


class CrossModalFusion(nn.Module):
    def __init__(self, channels, embed_dim=128, use_gating: bool = True, gating_mode: str = "adaptive"):
        """
        gating_mode:
            - "adaptive": 原自适应门控
            - "ones":     gate=1，输出= x（用于“无门控”对照，保留cross_att仅用于统计/可视化，不注入）
            - "average":  gate=0.5，输出= 0.5*x + 0.5*cross_att（去掉门控的自适应性，但仍做固定融合）
        """
        super().__init__()
        self.use_gating = use_gating
        self.gating_mode = gating_mode

        self.text_encoder = nn.Linear(768, embed_dim)  # 假设使用BERT嵌入
        self.multihead_attention = MultiHeadAttention(embed_dim, num_heads=8)

        # 门控网络（仅在 adaptive 时使用）
        self.gate_net = nn.Sequential(
            nn.Linear(channels + embed_dim, channels),
            nn.Sigmoid()
        )

    def forward(self, x, knowledge_embed):
        # x: (B, C, L), knowledge_embed: (B,768)
        B, C, L = x.shape

        # 编码外部知识
        knowledge = self.text_encoder(knowledge_embed)  # (B, embed_dim)
        knowledge_expanded = knowledge.unsqueeze(1).expand(-1, L, -1)  # (B, L, embed_dim)

        # 跨模态注意力：用知识向量作为 K/V，引导每个时间步
        cross_att = self.multihead_attention(
            x.transpose(1, 2),              # (B, L, C) 作为 Query
            knowledge_expanded,             # (B, L, E) 作为 Key
            knowledge_expanded              # (B, L, E) 作为 Value
        ).transpose(1, 2)                   # -> (B, C, L)

        if not self.use_gating:
            if self.gating_mode == "ones":
                # gate=1，仅返回原输入（不引入 cross_att），用来隔离“门控的自适应性”
                return x
            elif self.gating_mode == "average":
                # 固定 0.5 融合，去掉门控的自适应性但保融合通路
                return 0.5 * x + 0.5 * cross_att
            else:
                raise ValueError(f"Unsupported gating_mode when use_gating=False: {self.gating_mode}")

        # 自适应门控（原逻辑）
        gate = self.gate_net(torch.cat([x.mean(dim=2), knowledge], dim=1)).unsqueeze(2)  # (B,C,1)
        return gate * x + (1.0 - gate) * cross_att



class EnhancedSAMFormer(nn.Module):
    def __init__(self, channels, seq_len, embed_dim=128):
        super().__init__()
        self.quantum_enc = QuantumChannelEncoder(channels)
        self.st_attention = STDecoupledAttention(channels, seq_len)
        self.fusion = CrossModalFusion(channels, embed_dim)
        self.pruner = ChannelPruner(channels)

    def forward(self, x, domain_knowledge=None):
        x = self.quantum_enc(x)
        x = self.st_attention(x)

        if domain_knowledge is not None:
            x = self.fusion(x, domain_knowledge)

        return self.pruner(x)


class QuantumChannelEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Linear(channels, channels)

    def forward(self, x):
        return self.fc(x)


class STDecoupledAttention(nn.Module):
    def __init__(self, channels, seq_len):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=channels, num_heads=4)

    def forward(self, x):
        return self.attention(x, x, x)[0]


class ChannelPruner(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Linear(channels, channels)

    def forward(self, x):
        return self.fc(x)




class StandardAttention(nn.Module):
    def __init__(self, seq_len, hid_dim):
        super().__init__()
        self.compute_keys = nn.Linear(seq_len, hid_dim)
        self.compute_queries = nn.Linear(seq_len, hid_dim)
        self.compute_values = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        queries = self.compute_queries(x)
        keys = self.compute_keys(x)
        values = self.compute_values(x)
        att_score = scaled_dot_product_attention(queries, keys, values)
        return att_score


class ReverseAttention(nn.Module):
    def __init__(self, seq_len, hid_dim):
        super().__init__()
        self.compute_keys = nn.Linear(seq_len, hid_dim)
        self.compute_queries = nn.Linear(seq_len, hid_dim)
        self.compute_values = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        queries = self.compute_queries(x)
        keys = self.compute_keys(x)
        values = self.compute_values(x)
        att_score = scaled_dot_product_attention(queries, keys, values)
        reversed_att_score = -att_score  # 反向处理
        return reversed_att_score


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TriTrackNetArchitecture(nn.Module):
    def __init__(
        self,
        num_channels,
        seq_len,
        hid_dim=16,
        pred_horizon=96,
        use_revin=True,
        use_aux=True,
        # A：attention vs HFF
        use_attention: bool = True,
        use_hff: bool = True,
        aux_mode: str = "keep",
        # C：gating on/off
        use_gating: bool = True,
        gating_mode: str = "adaptive",
        # cross-modal
        cross_embed_dim: int = 128
    ):
        super().__init__()
        self.use_revin = use_revin
        self.use_aux = use_aux

        self.revin = RevIN(num_features=num_channels)

        self.dual_channel_attention = DualChannelAttention(
            seq_len=seq_len,
            hid_dim=hid_dim,
            num_channels=num_channels,
            mlp_hidden_dim=64,
            use_attention=use_attention,
            use_hff=use_hff,
            aux_mode=aux_mode
        )

        # 仅在 use_aux=True 时启用跨模态融合（可配合 gating 实验）
        self.cross_modal_fusion = CrossModalFusion(
            channels=num_channels,
            embed_dim=cross_embed_dim,
            use_gating=use_gating,
            gating_mode=gating_mode
        ) if use_aux else None

        self.linear_forecaster = nn.Linear(seq_len, pred_horizon)

    def forward(self, x, domain_knowledge=None, flatten_output=True):
        # x: (B, C, L)
        if self.use_revin:
            x_norm = self.revin(x.transpose(1, 2), mode='norm').transpose(1, 2)
        else:
            x_norm = x

        att_score = self.dual_channel_attention(x_norm)

        if self.use_aux and (self.cross_modal_fusion is not None) and (domain_knowledge is not None):
            att_score = self.cross_modal_fusion(att_score, domain_knowledge)

        out = x_norm + att_score
        out = self.linear_forecaster(out)

        if self.use_revin:
            out = self.revin(out.transpose(1, 2), mode='denorm').transpose(1, 2)

        if flatten_output:
            return out.reshape([out.shape[0], out.shape[1] * out.shape[2]])
        else:
            return out



class TriTrackNet(nn.Module):
    def __init__(
        self,
        device='cuda:0',
        num_epochs=100,
        batch_size=256,
        base_optimizer=torch.optim.AdamW,
        learning_rate=1e-3,
        weight_decay=1e-5,
        rho=0.5,
        use_revin=True,
        random_state=None,
        use_aux=True,
        # A / C 开关
        use_attention: bool = True,
        use_hff: bool = True,
        aux_mode: str = "keep",
        use_gating: bool = True,
        gating_mode: str = "adaptive"
    ):
        super().__init__()
        self.criterion = nn.SmoothL1Loss()
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.base_optimizer = base_optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rho = rho
        self.use_revin = use_revin
        self.random_state = 13 if random_state is None else random_state
        self.use_aux = use_aux

        # 记录实验配置
        self.use_attention = use_attention
        self.use_hff = use_hff
        self.aux_mode = aux_mode
        self.use_gating = use_gating
        self.gating_mode = gating_mode

    def fit(self, x, y, domain_knowledge=None):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)
            np.random.seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)

        self.network = TriTrackNetArchitecture(
            num_channels=x.shape[1],
            seq_len=x.shape[2],
            hid_dim=16,
            pred_horizon=y.shape[1] // x.shape[1],
            use_revin=self.use_revin,
            use_aux=self.use_aux,
            use_attention=self.use_attention,
            use_hff=self.use_hff,
            aux_mode=self.aux_mode,
            use_gating=self.use_gating,
            gating_mode=self.gating_mode
        )

        self.criterion = self.criterion.to(self.device)
        self.network = self.network.to(self.device)
        self.network.train()

        optimizer = perturbopt(self.network.parameters(), base_optimizer=self.base_optimizer, rho=self.rho,
                        lr=self.learning_rate, weight_decay=self.weight_decay)

        train_dataset = LabeledDataset(x, y)
        data_loader_train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        progress_bar = tqdm(range(self.num_epochs))
        for epoch in progress_bar:
            loss_list = []
            for (x_batch, y_batch) in data_loader_train:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                out_batch = self.network(x_batch, domain_knowledge=domain_knowledge if self.use_aux else None)
                loss = self.criterion(out_batch, y_batch)

                if optimizer.__class__.__name__ == 'perturbopt':
                    loss.backward()
                    optimizer.first_step(zero_grad=True)

                    out_batch = self.network(x_batch, domain_knowledge=domain_knowledge if self.use_aux else None)
                    loss = self.criterion(out_batch, y_batch)

                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_list.append(loss.item())

            train_loss = np.mean(loss_list)
            self.network.train()
            progress_bar.set_description(f"Epoch {epoch}: Train Loss {train_loss:.4f}", refresh=True)
        return

    def forecast(self, x, domain_knowledge=None, batch_size=256):
        self.network.eval()
        dataset = torch.utils.data.TensorDataset(torch.tensor(x, dtype=torch.float))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        outs = []
        for _, batch in enumerate(dataloader):
            xb = batch[0].to(self.device)
            with torch.no_grad():
                out = self.network(xb, domain_knowledge=domain_knowledge if self.use_aux else None)
            outs.append(out.cpu())
        outs = torch.cat(outs)
        return outs.cpu().numpy()

    def predict(self, x, domain_knowledge=None, batch_size=256):
        return self.forecast(x, domain_knowledge=domain_knowledge, batch_size=batch_size)



class DualChannelAttention(nn.Module):
    def __init__(
        self,
        seq_len,
        hid_dim,
        num_channels,
        mlp_hidden_dim=64,
        use_attention: bool = True,
        use_hff: bool = True,
        aux_mode: str = "keep"  # "keep" | "off"
    ):
        super().__init__()
        self.num_channels = num_channels
        self.use_attention = use_attention
        self.use_hff = use_hff
        self.aux_mode = aux_mode

        # 主通道：标准注意力（时间维长依赖）
        if self.use_attention:
            self.compute_keys_1 = nn.Linear(seq_len, hid_dim)
            self.compute_queries_1 = nn.Linear(seq_len, hid_dim)
            self.compute_values_1 = nn.Linear(seq_len, seq_len)

        # 辅助通道 1：反向注意力（可视作“抑制/对比”信号）
        if self.aux_mode == "keep":
            self.reverse_attention_2 = ReverseAttention(seq_len, hid_dim)
        else:
            self.reverse_attention_2 = None

        # 辅助通道 2：HFF（异构特征融合，这里用 MLP 表征）
        if self.use_hff:
            self.mlp_3 = MLP(seq_len, mlp_hidden_dim, seq_len)
        else:
            self.mlp_3 = None

        # 当两者都关闭时，保形占位（Identity/线性）
        self.identity_head = nn.Identity()

    def forward(self, x):
        # 将通道切 3 段，仅用于与原实现对齐；也可改成全通道并行
        c = max(1, self.num_channels // 3)
        channel_1 = x[:, :c, :]
        channel_2 = x[:, c: 2 * c, :] if (c < self.num_channels) else x[:, :c, :]
        channel_3 = x[:, 2 * c:, :]   if (2 * c < self.num_channels) else x[:, :c, :]

        outs = []

        # 主通道：标准注意力
        if self.use_attention:
            q1 = self.compute_queries_1(channel_1)
            k1 = self.compute_keys_1(channel_1)
            v1 = self.compute_values_1(channel_1)
            att_score_1 = scaled_dot_product_attention(q1, k1, v1)  # (B,c,L)
        else:
            # 不用注意力时，保形（可改为线性/恒等）
            att_score_1 = self.identity_head(channel_1)
        outs.append(att_score_1)

        # 辅助通道 1：反向注意力（是否保留由 aux_mode 决定）
        if self.reverse_attention_2 is not None:
            rev2 = self.reverse_attention_2(channel_2)
        else:
            rev2 = torch.zeros_like(channel_2)
        outs.append(rev2)

        # 辅助通道 2：HFF（MLP）
        if self.mlp_3 is not None:
            hff3 = self.mlp_3(channel_3)
        else:
            hff3 = torch.zeros_like(channel_3)
        outs.append(hff3)

        # 拼接回原通道数（如果切分造成尾部尺寸不齐，可在外层再做裁剪/填充）
        out = torch.cat(outs, dim=1)

        # 若拼接超出原通道数，裁剪至原通道；不足则 pad
        if out.size(1) > x.size(1):
            out = out[:, :x.size(1), :]
        elif out.size(1) < x.size(1):
            pad = torch.zeros(x.size(0), x.size(1) - out.size(1), x.size(2), device=x.device, dtype=x.dtype)
            out = torch.cat([out, pad], dim=1)

        return out


    def predict(self, x, batch_size=256):
        return self.forecast(x, batch_size=batch_size)
