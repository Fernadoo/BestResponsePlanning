import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class MlpExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, feat_dim):
        super().__init__()
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.head = nn.Linear(hidden_dim, feat_dim)

        # self.block = nn.Sequential(
        #     self.fc1, nn.ReLU(),
        #     self.fc2, nn.ReLU(),
        #     self.fc3, nn.ReLU(),
        #     self.head, nn.ReLU(),
        # )

        self.block1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim), nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim), nn.ReLU(),
        )

    def forward(self, x):
        feat1 = self.block1(x)
        feat2 = feat1 + self.block2(feat1)
        return feat2


class ObsExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim=128,
                 N=5, loc_hidden_dim=128, belief_hidden_dim=256, feat_loc_dim=64, feat_belief_dim=32, postproc_hidden_dim=128):
        super().__init__(observation_space, features_dim)

        self.input_dim = np.product(observation_space.shape)
        self.loc_dim = 2 + N * 2
        self.belief_dim = (self.input_dim - self.loc_dim) // (N - 1)
        self.N = N

        self.loc_extractor = MlpExtractor(
            input_dim=self.loc_dim,
            hidden_dim=loc_hidden_dim,
            feat_dim=feat_loc_dim,
        )

        self.belief_extractors = nn.ModuleList([
            MlpExtractor(
                input_dim=self.belief_dim,
                hidden_dim=belief_hidden_dim,
                feat_dim=feat_belief_dim,
            )
            for _ in range(self.N - 1)
        ])

        self.loc_ln = nn.LayerNorm(feat_loc_dim)
        self.belief_ln = nn.LayerNorm(feat_belief_dim)

        self.postproc1 = nn.Sequential(
            nn.Linear(feat_loc_dim + (N - 1) * feat_belief_dim, N * postproc_hidden_dim), nn.ReLU(),
            nn.Linear(N * postproc_hidden_dim, N * postproc_hidden_dim), nn.ReLU(),
            nn.Linear(N * postproc_hidden_dim, N // 2 * postproc_hidden_dim), nn.ReLU(),
            nn.Linear(N // 2 * postproc_hidden_dim, features_dim), nn.ReLU(),
        )

        self.postproc2 = nn.Sequential(
            nn.Linear(features_dim, N * postproc_hidden_dim), nn.ReLU(),
            nn.Linear(N * postproc_hidden_dim, N // 2 * postproc_hidden_dim), nn.ReLU(),
            nn.Linear(N // 2 * postproc_hidden_dim, features_dim), nn.ReLU(),
        )

    def forward(self, obs):
        locs = obs[:, :self.loc_dim]
        feat_loc = self.loc_ln(self.loc_extractor(locs))
        feat_belief = [
            self.belief_ln(self.belief_extractors[i](
                obs[:,
                    self.loc_dim + i * self.belief_dim:
                    self.loc_dim + (i + 1) * self.belief_dim]
            ))
            for i in range(self.N - 1)
        ]
        feat_merge = torch.cat([feat_loc, *feat_belief], dim=1)
        feat1 = self.postproc1(feat_merge)
        feat2 = feat1 + self.postproc2(feat1)
        return feat2
