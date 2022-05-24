# Copyright (c) 2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cudf
from cugraph.experimental import PropertyGraph
import numpy as np
import scipy.sparse as sp


def create_property_graph_from_reddit_data(reddit_graph_file_name,
                                           reddit_data_file_name):
    """
    Creates and returns a cugraph PropertyGraph using data from
    reddit_graph_file_name and reddit_data_file_name.
    """
    coo_adj = sp.load_npz(reddit_graph_file_name)
    edgelist = cudf.DataFrame()
    edgelist['src'] = cudf.Series(coo_adj.row)
    edgelist['dst'] = cudf.Series(coo_adj.col)
    edgelist['wt'] = cudf.Series(coo_adj.data)

    reddit_data = np.load(reddit_data_file_name)
    features = reddit_data["feature"]
    cu_features = cudf.DataFrame(features)
    cu_features['name'] = np.arange(cu_features.shape[0])

    pG = PropertyGraph()
    pG.add_edge_data(edgelist, vertex_col_names=("src", "dst"))
    pG.add_vertex_data(cu_features, vertex_col_name="name")

    return pG
