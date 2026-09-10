"""Lightweight baselines -- the cheap end of the accuracy/compute frontier.

LargeST ships twelve baselines, but they are all mid-to-heavy: the cheapest
learned model in the repo is a 2-layer LSTM. The paper's claim is about the
shape of the frontier, so it needs models an order of magnitude below that.
Both models here follow the repo's BaseModel contract exactly -- forward takes
(b, t, n, f) and returns (b, horizon, n, 1) -- so they drop straight into
BaseEngine with no special-casing.

STID follows Shao et al., "Spatial-Temporal Identity: A Simple yet Effective
Baseline for Multivariate Time Series Forecasting" (CIKM 2022). Cite it; do
not present it as ours. Our contribution is the cost-accuracy analysis, not
the architecture.
"""

import torch
import torch.nn as nn

from src.base.model import BaseModel


# The CA history is resampled to 15 minutes in process_ca_his.ipynb, so a day
# is 96 steps, not the 288 you would get from the raw 5-minute feed. If you
# turn the resampling off, pass steps_per_day=288.
STEPS_PER_DAY_15MIN = 96


class HistoricalLast(BaseModel):
    """Repeat the most recent observation across the horizon. Zero parameters.

    The repo ships an equivalent (src/models/hl.py) but it returns all three
    input channels, so it only works when launched with --input_dim 1. Ours
    slices the value channel explicitly, which keeps input_dim=3 uniform
    across the whole lineup and removes a per-model special case from the
    comparison.
    """

    def __init__(self, **args):
        super(HistoricalLast, self).__init__(**args)
        # BaseEngine wants something to hand the optimiser; never used.
        self._unused = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, input, label=None):  # (b, t, n, f)
        return input[:, [-1], :, 0:1].expand(-1, self.horizon, -1, -1)


class _MLPResidual(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc2(self.drop(self.act(self.fc1(x))))
        return h + x


class STID(BaseModel):
    """Node/time identity embeddings + an MLP. No graph, no recurrence.

    The whole model is pointwise convolutions over a (b, d, n, 1) tensor, so
    cost scales linearly in node count -- which is the property that matters
    when a city adds sensors.
    """

    def __init__(self, embed_dim=32, node_dim=32, temp_dim=32, layers=3,
                 dropout=0.15, steps_per_day=STEPS_PER_DAY_15MIN,
                 use_time_feats=True, **args):
        super(STID, self).__init__(**args)
        self.use_time_feats = use_time_feats and self.input_dim >= 3
        self.steps_per_day = steps_per_day

        self.node_emb = nn.Parameter(torch.empty(self.node_num, node_dim))
        nn.init.xavier_uniform_(self.node_emb)

        if self.use_time_feats:
            self.tod_emb = nn.Parameter(torch.empty(steps_per_day, temp_dim))
            self.dow_emb = nn.Parameter(torch.empty(7, temp_dim))
            nn.init.xavier_uniform_(self.tod_emb)
            nn.init.xavier_uniform_(self.dow_emb)

        self.series_emb = nn.Conv2d(self.seq_len, embed_dim, kernel_size=(1, 1))

        hidden = embed_dim + node_dim + (2 * temp_dim if self.use_time_feats else 0)
        self.encoder = nn.Sequential(*[_MLPResidual(hidden, dropout) for _ in range(layers)])
        self.head = nn.Conv2d(hidden, self.horizon, kernel_size=(1, 1))

    def forward(self, input, label=None):  # (b, t, n, f)
        b, t, n, _ = input.shape

        # value channel -> (b, t, n, 1) -> conv over the time axis
        x = input[..., 0:1]
        h = self.series_emb(x)  # (b, embed_dim, n, 1)

        feats = [h, self.node_emb.T.unsqueeze(0).unsqueeze(-1).expand(b, -1, -1, 1)]

        if self.use_time_feats:
            # generate_data_for_training stores tod as a fraction of the day
            # and dow as dayofweek/7, both taken at each timestep; index off
            # the most recent one.
            tod = input[:, -1, :, 1]
            dow = input[:, -1, :, 2]
            tod_idx = torch.clamp((tod * self.steps_per_day).round().long(), 0, self.steps_per_day - 1)
            dow_idx = torch.clamp((dow * 7).round().long(), 0, 6)
            feats.append(self.tod_emb[tod_idx].permute(0, 2, 1).unsqueeze(-1))
            feats.append(self.dow_emb[dow_idx].permute(0, 2, 1).unsqueeze(-1))

        h = torch.cat(feats, dim=1)
        h = self.encoder(h)
        out = self.head(h)  # (b, horizon, n, 1)
        return out


class NLinear(BaseModel):
    """A single linear map from the input window to the horizon.

    Deliberately the floor of the frontier: if a model cannot beat this by a
    margin worth its extra joules, that is the finding. The last-value
    subtraction is the normalisation trick from Zeng et al., "Are Transformers
    Effective for Time Series Forecasting?" (AAAI 2023) -- cite it too.

    `individual=True` gives every sensor its own weights (node_num * seq_len *
    horizon params, still tiny); False shares one map across sensors.
    """

    def __init__(self, individual=False, subtract_last=True, **args):
        super(NLinear, self).__init__(**args)
        self.individual = individual
        self.subtract_last = subtract_last

        if individual:
            self.weight = nn.Parameter(torch.empty(self.node_num, self.seq_len, self.horizon))
            self.bias = nn.Parameter(torch.zeros(self.node_num, self.horizon))
            nn.init.xavier_uniform_(self.weight)
        else:
            self.proj = nn.Linear(self.seq_len, self.horizon)

    def forward(self, input, label=None):  # (b, t, n, f)
        x = input[..., 0]  # (b, t, n)

        last = x[:, -1:, :] if self.subtract_last else 0.0
        x = x - last

        x = x.permute(0, 2, 1)  # (b, n, t)
        if self.individual:
            # (b, n, t) x (n, t, h) -> (b, n, h)
            out = torch.einsum("bnt,nth->bnh", x, self.weight) + self.bias
        else:
            out = self.proj(x)

        out = out.permute(0, 2, 1)  # (b, h, n)
        if self.subtract_last:
            out = out + last
        return out.unsqueeze(-1)  # (b, h, n, 1)
