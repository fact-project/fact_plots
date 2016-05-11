#!/usr/bin/env python2
# -*- coding:utf-8 -*-
"""Plot Extractor Resolution and Bias of facttools for different data sets

Usage:
    plotExtractorPerformance.py <infile> [options]

Options:
    -o <outfolder>  outputpath [default: build/]
    -l <label>      label that shall be given in the legend [default: Crosstalk]
"""


from __future__ import division, print_function
import numpy as np
import h5py as h5
from matplotlib.colors import LogNorm
from docopt import docopt
from os import path
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import ROOT as rt
rt.gROOT.SetBatch()
import re
from commons import leg_coord as l
from commons import canv_res as cr

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

args = docopt(__doc__)


infile = args["<infile>"]
outfolder = str(args["-o"])
leg_label = str(args["-l"])

outfile_prefix = os.path.basename(infile).split(".")[0]

f = h5.File(infile)
keys = f.keys()
keys.sort(key=natural_keys)

df_dict = dict()

reso_canv = rt.TCanvas("reso", "reso", cr[0], cr[1])
bias_canv = rt.TCanvas("bias", "bias", cr[0], cr[1])

reso_graph = dict()
bias_graph = dict()

reso_leg = rt.TLegend(l[0], l[1], l[2], l[3])
bias_leg = rt.TLegend(l[0], l[1], l[2], l[3])

y_axis_title_offset = 1.2
x_axis_title_offset = 1.2

x_title = "True photon charge / p.e."

color = ([
    rt.kBlue,
    rt.kBlack,
    rt.kRed,
    rt.kGreen+2,
    rt.kMagenta,
])

for k, key in enumerate(keys):
    if "xTalk_10_100k" in key:
        keys[k] = "DCrate_4_100k"

num_of_plots_so_far = 0
keys.sort(key=natural_keys)
print(keys)

for key in keys:
    if "DCrate_4_100k" in key:
        key= "xTalk_10_100k"
    df_dict[key] = pd.read_hdf(infile,key)

    bin_width   = df_dict[key].bin_width.values
    bin_middles = df_dict[key].bin_middles.values

    # bin_middles = np.array(bin_middles, dtype=int)

    res         = df_dict[key].res.values
    err_res     = df_dict[key].err_res.values
    bias        = df_dict[key]["mean"].values
    err_bias    = df_dict[key].err_mean.values

    unit            = df_dict[key].unit_variated_parameter_key[0]
    variated_par    = df_dict[key].variated_par[0]
    variated_val    = df_dict[key][variated_par][0]


    # plot_charge_resolution(res, err_res, bias, err_bias, bin_middles, bin_width, name="")
    # plt.show()

    reso_canv.cd()
    reso_graph[key] = rt.TGraphErrors(len(res),bin_middles, res, bin_width/2, err_res)

    reso_graph[key].SetTitle("")
    reso_graph[key].GetXaxis().SetTitle(x_title)
    reso_graph[key].GetYaxis().SetTitle("Resolution")

    reso_graph[key].GetYaxis().SetRangeUser(0,0.35)
    reso_graph[key].GetXaxis().SetRangeUser(1,200.)

    reso_graph[key].GetYaxis().SetTitleOffset(y_axis_title_offset)
    reso_graph[key].GetXaxis().SetTitleOffset(x_axis_title_offset)

    reso_graph[key].SetMarkerColor(color[num_of_plots_so_far])
    reso_graph[key].SetLineColor(color[num_of_plots_so_far])
    reso_graph[key].SetMarkerStyle(20+num_of_plots_so_far)

    if not num_of_plots_so_far:
        reso_graph[key].Draw("AP")
        reso_canv.SetGridx()
        reso_canv.SetGridy()
    else:
        reso_graph[key].Draw("PSames")

    label = "{}{: 6.1f} {}".format(leg_label, variated_val, unit)
    reso_leg.AddEntry(reso_graph[key], label, "lpe" )

    reso_canv.Modified()
    reso_canv.Update()



    bias_canv.cd()
    bias_graph[key] = rt.TGraphErrors(len(bias),bin_middles, bias, bin_width/2, err_bias)

    bias_graph[key].SetTitle("")
    bias_graph[key].GetXaxis().SetTitle(x_title)
    bias_graph[key].GetYaxis().SetTitle("Bias")

    bias_graph[key].GetYaxis().SetRangeUser(0.08,0.2)
    bias_graph[key].GetXaxis().SetRangeUser(1,200.)

    bias_graph[key].GetYaxis().SetTitleOffset(y_axis_title_offset)
    bias_graph[key].GetXaxis().SetTitleOffset(x_axis_title_offset)

    bias_graph[key].GetXaxis().SetDecimals(False)


    bias_graph[key].SetMarkerColor(color[num_of_plots_so_far])
    bias_graph[key].SetLineColor(color[num_of_plots_so_far])
    bias_graph[key].SetMarkerStyle(20+num_of_plots_so_far)

    if not num_of_plots_so_far:
        bias_graph[key].Draw("AP")
        bias_canv.SetGridx()
        bias_canv.SetGridy()
    else:
        bias_graph[key].Draw("PSames")

    label = "{}{: 6.1f} {}".format(leg_label, variated_val, unit)
    bias_leg.AddEntry(reso_graph[key], label, "lpe" )

    bias_canv.Modified()
    bias_canv.Update()
    num_of_plots_so_far += 1

reso_canv.cd()
reso_canv.Draw()
reso_leg.Draw()
reso_canv.SaveAs(os.path.join(outfolder,outfile_prefix+"_Resolution.pdf"))
#
bias_canv.cd()
bias_canv.Draw()
bias_leg.Draw()
bias_canv.SaveAs(os.path.join(outfolder,outfile_prefix+"_Bias.pdf"))
