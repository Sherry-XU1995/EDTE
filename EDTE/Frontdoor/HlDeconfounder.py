import math
import torch.nn as nn

from .layers import GCNConv, HodgeLaguerreConv


class HLDeconfounder(nn.Module):

    def __init__(self, args):
        super(HLDeconfounder, self).__init__()
        n_node = args.num_nodes
        emb_input_dim = args.n_factors * args.delta_d
        time_delay_scaler = 3
        edge_feature_dim = 2 + math.ceil(args.length / time_delay_scaler)
        seq_len = args.length
        K = 3
        bias = True
        # Start MLP for edges
        hid_dim = emb_input_dim
        self.start_mlp_edge = nn.Linear(edge_feature_dim, hid_dim)
        # HodgeLaguerreConv for edges
        self.spatial_edge = HodgeLaguerreConv(in_channels=hid_dim, out_channels=hid_dim, K=K, bias=bias)
        # project the updated edge features to the causal score
        self.edge_causal = nn.Linear(hid_dim, K * 2)
        ###### entity ########
        # reduce the time dimension of entity
        self.t_proj_cau = nn.Linear(seq_len, 1)
        ######### message passing ###########
        self.spatial_node = GCNConv(in_channels=emb_input_dim, num_nodes=n_node, out_channels=hid_dim, K=K)

    def forward(self, h_entity, x_link, edge_index_link, edge_weight_link, edge_index):
        ############# edge feature to causal score ############
        # update the edge feature to recogenize the causal score
        h_link = self.start_mlp_edge(x_link.float())
        h_link_updated = self.spatial_edge(h_link, edge_index_link, edge_weight_link)
        norm_causal_score = self.edge_causal(h_link_updated)
        ############ entity and message passing ############
        # reduce the time dimension
        h_entity = self.t_proj_cau(h_entity.permute(1, 2, 0)).squeeze()
        # update the node representation based on the causal score
        h_entity = self.spatial_node(h_entity, edge_index, norm_causal_score)
        return h_entity
