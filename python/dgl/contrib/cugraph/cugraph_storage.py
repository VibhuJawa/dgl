# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

# NOTE: Requires cuGraph nightly cugraph-22.06.00a220417 or later

import dgl
import dgl.backend as F
from functools import cached_property

# from cugraph_utils import _assert_valid_canonical_etype


class CuGraphStorage:
    """
    Duck-typed version of the DGL GraphStorage class made for cuGraph
    """

    def __init__(self, g, node_id_conversion_d=None, idtype=F.int32):
        # lazy import to prevent creating cuda context
        # till later to help in multiprocessing
        from cugraph.gnn import CuGraphStore

        self.graphstore = CuGraphStore(graph=g)
        self.node_id_conversion_d = node_id_conversion_d
        self.idtype = idtype

    def get_node_storage(self, feat_name, ntype=None):
        return self.graphstore.get_node_storage(feat_name, ntype)

    def get_edge_storage(self, feat_name, etype=None):
        return self.graphstore.get_edge_storage(feat_name, etype)

    @property
    def num_nodes_dict(self):
        return self.graphstore.num_nodes_dict

    @property
    def num_edges_dict(self):
        return self.graphstore.num_edges_dict

    @property
    def ntypes(self):
        """
        Return all the node type names in the graph.

        Returns
        -------
        list[str]
            All the node type names in a list.
        """
        return self.graphstore.ntypes

    @property
    def etypes(self):
        """
        Return all the edge type names in the graph.

        Returns
        -------
        list[str]
            All the edge type names in a list.
        """

        return [can_etype[1] for can_etype in self.canonical_etypes]

    @property
    def canonical_etypes(self):
        can_etypes = self.graphstore.etypes
        if len(can_etypes) > 1:
            return [convert_can_etype_s_to_tup(s) for s in can_etypes]
        else:
            return [("#", can_etypes[0], "#")]

    def add_node_data(self, df, node_col_name, feat_name, ntype=None):
        self.graphstore.add_node_data(df, node_col_name, feat_name, ntype)

        # This will delete cached value
        self.__dict__.pop("total_number_of_nodes", None)

    def add_edge_data(
        self, df, vertex_col_names, feat_name, canonical_etype=None
    ):
        if canonical_etype:
            _assert_valid_canonical_etype(canonical_etype)

        # Convert to a string because cugraph PG does not support tuple objects
        canonical_etype = str(canonical_etype)
        self.graphstore.add_edge_data(
            df, vertex_col_names, feat_name, canonical_etype
        )

    # Index Conversion Utils
    def get_node_id_conversion_d(self):
        # dict for node_id_start
        last_st = 0
        node_ind_st_d = {}
        for ntype in self.ntypes:
            node_ind_st_d[ntype] = last_st
            last_st = last_st + self.num_nodes_dict[ntype]
        return node_ind_st_d

    @property
    def edge_id_conversion_d(self):
        # dict for edge_id_start
        last_st = 0
        edge_ind_st_d = {}
        for etype in self.canonical_etypes:
            edge_ind_st_d[etype] = last_st
            last_st = last_st + self.num_edges_dict[str(etype)]
        return edge_ind_st_d

    def dgl_n_id_to_cugraph_id(self, index_t, ntype):
        if self.node_id_conversion_d is None:
            self.node_id_conversion_d = self.get_node_id_conversion_d()

        return index_t + self.node_id_conversion_d[ntype]

    def dgl_e_id_to_cugraph_id(self, index_t, etype):
        return index_t + self.edge_id_conversion_d[etype]

    def cugraph_n_id_to_dgl_id(self, index_t, ntype):
        if self.node_id_conversion_d is None:
            self.node_id_conversion_d == self.get_node_id_conversion_d()

        return index_t - self.node_id_conversion_d[ntype]

    def cugraph_e_id_to_dgl_id(self, index_t, etype):
        return index_t - self.edge_id_conversion_d[etype]

    def sample_neighbors(
        self,
        seed_nodes,
        fanout,
        edge_dir="in",
        prob=None,
        exclude_edges=None,
        replace=False,
        output_device=None,
    ):
        """
        Return a DGLGraph which is a subgraph induced by sampling neighboring
        edges of the given nodes.
        See ``dgl.sampling.sample_neighbors`` for detailed semantics.
        Parameters
        ----------
        seed_nodes : Tensor or dict[str, Tensor]
            Node IDs to sample neighbors from.
            This argument can take a single ID tensor or a dictionary of node
            types and ID tensors. If a single tensor is given, the graph must
            only have one type of nodes.
        fanout : int or dict[etype, int]
            The number of edges to be sampled for each node on each edge type.
            This argument can take a single int or a dictionary of edge types
            and ints. If a single int is given, DGL will sample this number of
            edges for each node for every edge type.
            If -1 is given for a single edge type, all the neighboring edges
            with that edge type will be selected.
        prob : str, optional
            Feature name used as the (unnormalized) probabilities associated
            with each neighboring edge of a node.  The feature must have only
            one element for each edge.
            The features must be non-negative floats, and the sum of the
            features of inbound/outbound edges for every node must be positive
            (though they don't have to sum up to one).  Otherwise, the result
            will be undefined. If :attr:`prob` is not None, GPU sampling is
            not supported.
        exclude_edges: tensor or dict
            Edge IDs to exclude during sampling neighbors for the seed nodes.
            This argument can take a single ID tensor or a dictionary of edge
            types and ID tensors. If a single tensor is given, the graph must
            only have one type of nodes.
        replace : bool, optional
            If True, sample with replacement.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            A sampled subgraph with the same nodes as the original graph, but
            only the sampled neighboring edges.  The induced edge IDs will be
            in ``edata[dgl.EID]``.
        """

        if prob is not None:
            raise NotImplementedError(
                "prob is not currently supported",
                " for sample_neighbors in CuGraphStorage",
            )

        if exclude_edges is not None:
            raise NotImplementedError(
                "exclude_edges is not currently supported",
                " for sample_neighbors in CuGraphStorage",
            )

        if not isinstance(seed_nodes, dict):
            if len(self.ntypes) > 1:
                raise dgl.DGLError(
                    "Must specify node type when the graph is not homogeneous."
                )
            else:
                seed_nodes = F.tensor(seed_nodes)
                seed_nodes_cap = F.zerocopy_to_dlpack(seed_nodes)
        else:
            seed_nodes = {
                k: self.dgl_n_id_to_cugraph_id(F.tensor(n), k)
                for k, n in seed_nodes.items()
            }

            seed_nodes_cap = {
                k: F.zerocopy_to_dlpack(F.tensor(n))
                for k, n in seed_nodes.items()
            }

        sample_cap_obj = self.graphstore.sample_neighbors(
            seed_nodes_cap,
            fanout,
            edge_dir=edge_dir,
            prob=prob,
            replace=replace,
        )

        if len(self.etypes) > 1:
            graph_data_d, graph_eid_d = self._convert_pycap_to_dgl_tensor_d(
                sample_cap_obj
            )
            sampled_graph = dgl.heterograph(
                data_dict=graph_data_d,
                num_nodes_dict=self.num_nodes_dict,
                idtype=self.idtype,
            )
            sampled_graph.edata[dgl.EID] = graph_eid_d
        else:
            src_c, dst_c, edge_id_c = sample_cap_obj
            src_ids = F.zerocopy_from_dlpack(src_c)
            dst_ids = F.zerocopy_from_dlpack(dst_c)
            edge_id_t = F.zerocopy_from_dlpack(edge_id_c)

            sampled_graph = dgl.graph(
                (src_ids, dst_ids),
                num_nodes=self.total_number_of_nodes,
                idtype=self.idtype,
            )
            sampled_graph.edata[dgl.EID] = edge_id_t

        # to device function move the dgl graph to desired devices
        if output_device is not None:
            sampled_graph.to_device(output_device)
        return sampled_graph

    # Required in Cluster-GCN
    def subgraph(self, nodes, relabel_nodes=False, output_device=None):
        """Return a subgraph induced on given nodes.
        This has the same semantics as ``dgl.node_subgraph``.
        Parameters
        ----------
        nodes : nodes or dict[str, nodes]
            The nodes to form the subgraph. The allowed nodes formats are:
            * Int Tensor: Each element is a node ID. The tensor must have the
             same device type and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.
            * Bool Tensor: Each :math:`i^{th}` element is a bool flag
             indicating whether node :math:`i` is in the subgraph.
             If the graph is homogeneous, directly pass the above formats.
             Otherwise, the argument must be a dictionary with keys being
             node types and values being the node IDs in the above formats.
        relabel_nodes : bool, optional
            If True, the extracted subgraph will only have the nodes in the
            specified node set and it will relabel the nodes in order.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            The subgraph.
        """
        raise NotImplementedError("subgraph is not implemented")

    # Required in Link Prediction
    # relabel = F we use dgl functions,
    # relabel = T, we need to delete nodes and relabel
    def edge_subgraph(self, edges, relabel_nodes=False, output_device=None):
        """
        Return a subgraph induced on given edges.
        This has the same semantics as ``dgl.edge_subgraph``.
        Parameters
        ----------
        edges : edges or dict[(str, str, str), edges]
            The edges to form the subgraph. The allowed edges formats are:
            * Int Tensor: Each element is an edge ID. The tensor must have the
              same device type and ID data type as the graph's.
            * iterable[int]: Each element is an edge ID.
            * Bool Tensor: Each :math:`i^{th}` element is a bool flag
             indicating whether edge :math:`i` is in the subgraph.
            If the graph is homogeneous, one can directly pass the above
            formats. Otherwise, the argument must be a dictionary with keys
            being edge types and values being the edge IDs in the above formats
        relabel_nodes : bool, optional
            If True, the extracted subgraph will only have the nodes in the
            specified node set and it will relabel the nodes in order.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            The subgraph.
        """
        raise NotImplementedError("edge_subgraph is not implemented")

    # Required in Link Prediction negative sampler
    def find_edges(self, eid, etype=None, output_device=None):
        """
        Return the source and destination node ID(s) given the edge ID(s).

        Parameters
        ----------
        eid : edge ID(s)
            The edge IDs. The allowed formats are:

            * ``int``: A single ID.
            * Int Tensor: Each element is an ID.
            The tensor must have the same device type
            and ID data type as the graph's.
            * iterable[int]: Each element is an ID.

        etype : str
            The type names of the edges.
            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        Tensor
            The source node IDs of the edges.
            The i-th element is the source node ID of the i-th edge.
        Tensor
            The destination node IDs of the edges.
            The i-th element is the destination node ID of the i-th edge.
        """
        src_cap, dst_cap = self.graphstore.find_edges(eid, etype)
        # edges are a range of edge IDs, for example 0-100
        src_nodes_tensor = F.zerocopy_from_dlpack(src_cap).to(output_device)
        dst_nodes_tensor = F.zerocopy_from_dlpack(dst_cap).to(output_device)
        return src_nodes_tensor, dst_nodes_tensor

    # Required in Link Prediction negative sampler
    def num_nodes(self, ntype=None):
        """
        Return the number of nodes in the graph.
        Parameters
        ----------
        ntype : str, optional
            The node type name. If given, it returns the number of nodes of the
            type.
            If not given (default), it  returns the total number of nodes
            of all types.

        Returns
        -------
        int
            The number of nodes.
        """
        # use graphstore function
        return self.graphstore.num_nodes(ntype)

    def number_of_nodes(self, ntype):
        return self.num_nodes(ntype)

    @cached_property
    def total_number_of_nodes(self):
        return self.num_nodes()

    def num_edges(self, etype=None):
        """
        Return the number of edges in the graph.
        Parameters
        ----------
        ntype : str, optional
            The node type name. If given, it returns the number of nodes of the
            type. If not given (default), it returns the total number of nodes
            of all types.

        Returns
        -------
        int
            The number of edges
        """
        # use graphstore function
        return self.graphstore.num_edges(etype)

    def global_uniform_negative_sampling(
        self, num_samples, exclude_self_loops=True, replace=False, etype=None
    ):
        """
        Per source negative sampling as in ``dgl.dataloading.GlobalUniform``
        """
        raise NotImplementedError("canonical not implemented")

    def _convert_pycap_to_dgl_tensor_d(
        self, graph_data_cap_d, o_dtype=F.int64
    ):
        graph_data_d = {}
        graph_eid_d = {}
        for canonical_etype_s, (
            src_cap,
            dst_cap,
            edge_id_cap,
        ) in graph_data_cap_d.items():
            if isinstance(canonical_etype_s, str):
                # FIXME: REMOVE ALL string extracting
                canonical_etype = convert_can_etype_s_to_tup(canonical_etype_s)
                src_type = canonical_etype[0]
                dst_type = canonical_etype[2]
            else:
                raise AssertionError("FIXME:  To cleanly expect tuple")

            if src_cap is None:
                src_t = F.tensor(data=[])
                dst_t = F.tensor(data=[])
                edge_id_t = F.tensor(data=[])
            else:
                src_t = F.zerocopy_from_dlpack(src_cap)
                dst_t = F.zerocopy_from_dlpack(dst_cap)
                edge_id_t = F.zerocopy_from_dlpack(edge_id_cap)

                src_t = self.cugraph_n_id_to_dgl_id(src_t, src_type)
                dst_t = self.cugraph_n_id_to_dgl_id(dst_t, dst_type)
                edge_id_t = self.cugraph_e_id_to_dgl_id(
                    edge_id_t, canonical_etype
                )

            graph_data_d[canonical_etype] = (
                src_t.to(o_dtype).to("cuda"),
                dst_t.to(o_dtype).to("cuda"),
            )
            graph_eid_d[canonical_etype] = edge_id_t.to(o_dtype).to("cuda")

        return graph_data_d, graph_eid_d


def convert_can_etype_s_to_tup(canonical_etype_s):
    src_type, etype, dst_type = canonical_etype_s.split(",")
    src_type = src_type[2:-1]
    dst_type = dst_type[2:-2]
    etype = etype[2:-1]
    return (src_type, etype, dst_type)


# TODO: Move to utils
def _assert_valid_canonical_etype(canonical_etype):
    if not _is_valid_canonical_etype:
        error_message = (
            f"Invalid canonical_etype {canonical_etype} "
            + "canonical etype should be is a string triplet (str, str, str)"
            + "for source node type, edge type and destination node type"
        )
        raise dgl.DGLError(error_message)


def _is_valid_canonical_etype(canonical_etype):
    if not isinstance(canonical_etype, tuple):
        return False

    if len(canonical_etype) != 3:
        return False

    for t in canonical_etype:
        if not isinstance(t, str):
            return False
    return True