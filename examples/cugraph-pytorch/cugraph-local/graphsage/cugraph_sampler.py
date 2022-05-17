"""
cugraph sampling test on Benchmark Datasets
"""

#  Import the modules
import cugraph
import cudf
import dgl

# system and other
import time
import cupy as cp

# MTX file reader
from scipy.io import mmread
from torch.utils.dlpack import from_dlpack
import torch

# dlpack can only used for pytorch > 1.10 and cupy > 10


def read_and_create(datafile):
    # print('Reading ' + str(datafile) + '...')
    M = mmread(datafile).asfptype()

    _gdf = cudf.DataFrame()
    _gdf["src"] = M.row
    _gdf["dst"] = M.col
    _gdf["wt"] = 1.0

    _g = cugraph.Graph()
    _g.from_cudf_edgelist(
        _gdf, source="src", destination="dst", edge_attr="wt", renumber=False
    )

    # print("\t{:,} nodes, {:,} edges".format(_g.number_of_nodes(), _g.number_of_edges() ))

    return _g


def create_tensor_from_cupy_cudf_objs(obj):
    if isinstance(obj, cudf.Series):
        return from_dlpack(obj.values.toDlpack())
    elif isinstance(obj, cp.ndarray):
        return from_dlpack(obj.toDlpack())
    else:
        raise TypeError(
            "Expected type of obj to be either cudf.Series or cp.ndarray,"
            + f" got obj of {type(obj)} instead"
        )


def group_sample(df, by, n_samples):
    # first, shuffle the dataframe and reset its index,
    # so that the ordering of values within each group
    # is made random:
    df = df.sample(frac=1).reset_index(drop=True)

    # add an integer-encoded version of the "by" column,
    # since the rank aggregation seems not to work for
    # non-numeric data
    df["_"] = df[by].astype("category").cat.codes

    # now do a "rank" aggregation and filter out only
    # the first N_SAMPLES ranks.
    result = df.loc[df.groupby(by)["_"].rank("first") <= n_samples, :]
    del result["_"]
    return result


def cugraphSampler(
    g,
    nodes,
    fanouts,
    edge_dir="in",
    prob=None,
    replace=False,
    copy_ndata=True,
    copy_edata=True,
    _dist_training=False,
    exclude_edges=None,
):
    # from here get in a new for loop
    # ego_net return edge list

    # vjawa: Below fails
    # print("edge data")
    # print(g.edge_data)
    num_nodes = len(nodes)
    if torch.is_tensor(nodes):
        current_seeds = cp.asarray(nodes)
        current_seeds = cudf.Series(current_seeds)
    else:
        current_seeds = nodes.reindex(index=cp.arange(0, num_nodes))

    # blocks = []
    # seeds = cudf.Series(nodes.to_array())
    for fanout in fanouts:
        (
            ego_edge_list,
            seeds_offsets,
        ) = cugraph.community.egonet.batched_ego_graphs(
            g, current_seeds, radius=1
        )
        # filter and get a certain size neighborhood
        # Step 1
        # Get Filtered List of ego_edge_list corresposing to current_seeds
        # We filter by creating a series of destination nodes
        # corresponding to the offsets and filtering non matching vallues

        seeds_offsets_s = cudf.Series(seeds_offsets).values
        offset_lens = seeds_offsets_s[1:] - seeds_offsets_s[0:-1]
        dst_seeds = current_seeds.repeat(offset_lens)
        dst_seeds.index = ego_edge_list.index
        filtered_list = ego_edge_list[ego_edge_list["dst"] == dst_seeds]

        # Step 2
        # Sample Fan Out
        # for each dst take maximum of fanout samples
        filtered_list = group_sample(filtered_list, by="dst", n_samples=fanout)

        all_children = create_tensor_from_cupy_cudf_objs(filtered_list["src"])
        all_parents = create_tensor_from_cupy_cudf_objs(filtered_list["dst"])

        sampled_graph = dgl.graph((all_children, all_parents))

        # print(all_parents)
        # print(all_children)
        # print(sampled_graph.edges())
        # print(seeds.to_array())
        # '_ID' is EID

        num_edges = len(all_children)
        sampled_graph.edata["_ID"] = from_dlpack(
            cp.arange(num_edges).toDlpack()
        )
        # print(sampled_graph.edata)
        # block =dgl.to_block(sampled_graph,current_seeds.to_array())
        # block.edata[dgl.EID] = eid
        # current_seeds = block.srcdata[dgl.NID]
        # current_seeds = cudf.Series(current_seeds.cpu().detach().numpy())

        # blocks.insert(0, block)
        # end of for

    return sampled_graph


if __name__ == "__main__":
    data = [
        "polbooks"
    ]  # , 'as-Skitter', 'citationCiteseer', 'caidaRouterLevel', 'coAuthorsDBLP', 'coPapersDBLP']
    for file_name in data:
        G_cu = read_and_create(
            "/home/nfs/vjawa/dgl/" + file_name + ".mtx"
        )
        nodes = G_cu.nodes()  # .to_array().tolist()
        # print(nodes.index)
        num_nodes = G_cu.number_of_nodes()
        # num_seeds_ = [1000, 3000, 5000, 10000]
        # just test 1 epoch
        batch_size = 10
        num_batch = num_nodes / batch_size
        print(num_batch)
        # in each epoch shuffle the nodes
        shuffled_nodes = cp.arange(num_nodes)
        # print(len(nodes), len(shuffled_nodes))
        cp.random.shuffle(shuffled_nodes)
        print(type(nodes))
        # shuffled_nodes = cudf.Series(shuffled_nodes)
        # new_nodes = nodes.reindex(index = shuffled_nodes)
        shuffled_nodes = create_tensor_from_cupy_cudf_objs(shuffled_nodes)
        print(nodes)
        for i in range(int(num_batch) - 1):
            blocks = cugraphSampler(
                G_cu,
                shuffled_nodes[i * batch_size: (i + 1) * batch_size],
                [5, 10],
            )
