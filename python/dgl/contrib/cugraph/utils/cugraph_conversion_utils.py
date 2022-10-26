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

# Utils to convert b/w dgl heterograph to cugraph GraphStore
from typing import Optional
import dgl
import torch
from dgl.backend import zerocopy_to_dlpack

import numpy as np
import cupy as cp
import dask_cudf
import dask.dataframe as dd
import cudf
import pandas as pd

# Feature Tensor to DataFrame Utils
def convert_to_column_major(t: torch.Tensor):
    return t.t().contiguous().t()


def add_1d_tensor_to_dataframe(df, t, col_name):
    if t.device.type == 'cpu':
        ar = t.numpy()
    else:
        ar = cp.from_dlpack(zerocopy_to_dlpack(t))
    if isinstance(df, dask_cudf.DataFrame):
        # Need to send to GPU to cleanly distributed
        # FIXME: Can be removed if becomes a bottlenec
        if isinstance(ar, cp.ndarray):
            ar = ar.get()
        ar_ser = pd.Series(ar)
        dask_pandas_ser = dd.from_pandas(ar_ser, df.npartitions)
        dask_cudf_ser = dask_pandas_ser.map_partitions(cudf.from_pandas)
        df[col_name] = dask_cudf_ser
    elif isinstance(df, cudf.DataFrame):
        cudf_ser = cudf.Series(ar)
        df[col_name] = cudf_ser
    else:
        raise ValueError("Only adding to dask_cudf.DataFrame and cudf.DataFrame is currently supported")


def create_feature_frame(feat_t_d: dict[str, torch.Tensor], single_gpu=True):
    """
    Convert a feature_tensor_d to a dataframe
    """
    df_ls = []
    feat_name_map = {}
    for feat_key, feat_t in feat_t_d.items():
        
        if single_gpu:
            # Create single GPU cudf dataframes
            feat_t = feat_t.to("cuda")
            feat_t = convert_to_column_major(feat_t)
            ar = cp.from_dlpack(zerocopy_to_dlpack(feat_t))
            del feat_t
            df = cudf.DataFrame(ar)
            feat_columns = [f"{feat_key}_{i}" for i in range(len(df.columns))]
            df.columns = feat_columns
        else:
            # Create Multi-gpu dask_cudf dataframes
            ar = feat_t.numpy()
            df = pd.DataFrame(ar)
            feat_columns = [f"{feat_key}_{i}" for i in range(len(df.columns))]
            df.columns = feat_columns
            df = dd.from_pandas(df, npartitions=16)
            # Send to GPU only after distributing across partitions
            df = df.map_partitions(cudf.from_pandas)
        feat_name_map[feat_key] = df.columns
        df_ls.append(df)

    if isinstance(df_ls[0], cudf.DataFrame):
        df = cudf.concat(df_ls, axis=1)
    else:
        df = dask_cudf.concat(df_ls, axis=1)
    return df, feat_name_map


# Add ndata utils
def add_ndata_of_single_type(
    gs: dgl.contrib.cugraph.CuGraphStorage,
    feat_t_d: Optional[dict[torch.Tensor]],
    ntype: str,
    n_rows: int,
    idtype=torch.int64,
):
    node_ids = dgl.backend.arange(0, n_rows, idtype)
    df, feat_name_map = create_feature_frame(feat_t_d, gs.single_gpu)
    add_1d_tensor_to_dataframe(df, node_ids, "node_id")
    gs.add_node_data(
        df,
        "node_id",
        ntype=ntype,
        feat_name=feat_name_map,
        contains_vector_features=True,
    )
    return gs


def add_nodes_from_dgl_heteroGraph(
    gs: dgl.contrib.cugraph.CuGraphStorage,
    graph: dgl.DGLHeteroGraph,
):
    if len(graph.ntypes) > 1:
        ntype_feat_d = dict()
        for feat_name in graph.ndata.keys():
            for ntype in graph.ndata[feat_name]:
                if ntype not in ntype_feat_d:
                    ntype_feat_d[ntype] = {}
                ntype_feat_d[ntype][feat_name] = graph.ndata[feat_name][ntype]

        for ntype in gs.num_nodes_dict.keys():
            feat_t_d = ntype_feat_d.get(ntype, None)
            if feat_t_d is not None:
                gs = add_ndata_of_single_type(
                    gs=gs,
                    feat_t_d=feat_t_d,
                    ntype=ntype,
                    n_rows=gs.num_nodes_dict[ntype],
                    idtype=graph.idtype,
                )
    else:
        ntype = graph.ntypes[0]
        gs = add_ndata_of_single_type(
            gs,
            feat_t_d=graph.ndata,
            ntype=ntype,
            n_rows=graph.number_of_nodes(),
            idtype=graph.idtype
        )
    return gs


# Add edata utils
def add_edata_of_single_type(
    gs: dgl.contrib.cugraph.CuGraphStorage,
    feat_t_d: Optional[dict[torch.Tensor]],
    src_t: torch.Tensor,
    dst_t: torch.Tensor,
    can_etype: tuple([str, str, str]),
):
    if feat_t_d:
        feat_df, feat_name_map = create_feature_frame(feat_t_d, gs.single_gpu)
        add_1d_tensor_to_dataframe(feat_df, src_t, 'src')
        add_1d_tensor_to_dataframe(feat_df, dst_t, 'dst')
        gs.add_edge_data(
            df,
            ["src", "dst"],
            canonical_etype=can_etype,
            feat_name=feat_name_map,
            contains_vector_features=True,
        )
    else:
        # TODO: Fix this
        df = cudf.DataFrame(
        {
            "src": cudf.from_dlpack(zerocopy_to_dlpack(src_t)),
            "dst": cudf.from_dlpack(zerocopy_to_dlpack(dst_t)),
        })
        if not gs.single_gpu:
            df = dask_cudf.from_cudf(df, npartitions=16)

        gs.add_edge_data(
            df,
            ["src", "dst"],
            canonical_etype=can_etype,
            contains_vector_features=False,
        )
    return gs


def add_edges_from_dgl_heteroGraph(
    gs: dgl.contrib.cugraph.CuGraphStorage,
    graph: dgl.DGLHeteroGraph,
):
    etype_feat_d = dict()
    for feat_name in graph.edata.keys():
        for etype in graph.edata[feat_name].keys():
            if etype not in etype_feat_d:
                etype_feat_d[etype] = {}
            etype_feat_d[etype][feat_name] = graph.edata[feat_name][etype]

    for can_etype in graph.canonical_etypes:
        src_t, dst_t = graph.edges(form="uv", etype=can_etype)
        feat_t_d = etype_feat_d.get(can_etype, None)
        add_edata_of_single_type(gs, feat_t_d, src_t, dst_t, can_etype)
