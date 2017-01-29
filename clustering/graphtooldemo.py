#!/usr/bin/python3.5
# coding: utf-8
"""
example graph-tool usage.
"""
import graph_tool
from graph_tool import inference
from graph_tool import draw
from sizeDistributions import treeFileToNumpyArray
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument("edgelistfilename", help="the edgelist to be parsed")
parser.add_argument(
    "-p",
    "--pngfilename",
    type=str,
    help="the output png name",
    default="test.png")
args = parser.parse_args()

# if args.positioned:
# push the ATOM positions to a file
# with open(pdbFilename) as pdbFile:
# for line in pdbFile:
# comArray = treeFileToNumpyArray("edgelist3.tree")

g = graph_tool.load_graph_from_csv(
    args.edgelistfilename,
    directed=False,
    skip_first=True,
    csv_options={"delimiter": " "})

# pos = graph_tool.draw.sfdp_layout(g, C=1000000)
# pos = graph_tool.draw.sfdp_layout(g)
# pos = graph_tool.draw.fruchterman_reingold_layout(g, n_iter=1000)
# print(pos)
comps = graph_tool.topology.label_components(g)
# state = inference.minimize_blockmodel_dl(g, deg_corr=False)
# print(comps[1])
# print([x for x in comps])

numpy.savetxt("comps.txt", comps[1])# draw.graph_draw(
#     g,
#     pos=pos,
#     output=args.pngfilename,
#     output_size=(1200, 1200),
#     vertex_fill_color=comps)
# dS, nmoves = state.mcmc_sweep(niter=1000)
# print(nmoves)

# state.draw(pos=pos, output=args.pngfilename, output_size=(1200, 1200))
