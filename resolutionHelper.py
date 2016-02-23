from __future__ import division, print_function
import numpy as np
import h5py
from matplotlib import style
style.use("ggplot")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import moment
from scipy.optimize import curve_fit
from docopt import docopt
from os import path
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

def gauss(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5*(x-mu)**2/sigma**2)

def calcErrorOfEmpStdDev(X):
    mu = np.mean(X)
    mu2 = moment(X,2)
    mu3 = moment(X,3)
    mu4 = moment(X,4)
    return np.sqrt((mu4 - mu2**2 - 4*mu*mu3 + 4 * mu2* mu**2) / len(X))

def calc_resolution(e_mc, e_rek, bin_width=5, plot_hists=False, min_e=None, max_e=None):

    if min_e is None:
        min_e = np.min(e_mc)
    if max_e is None:
        max_e = np.max(e_mc)

    bins = int((max_e - min_e)/bin_width)
    bins_rek = []
    bins_mc  = []
    bin_middles = np.empty(bins)
    bin_middles[:] = np.nan
    res = np.empty(bins)
    res[:] = np.nan
    err_res = np.empty(bins)
    err_res[:] = np.nan
    means = np.empty(bins)
    means[:] = np.nan
    err_means = np.empty(bins)
    err_means[:] = np.nan

    # diff = var("diff")

    for i in range(bins):

        mask1 = e_mc < bin_width*(i+1) + min_e
        mask2 = e_mc >= bin_width*i + min_e

        mask = np.logical_and(mask1, mask2)

        if np.sum(mask) <= 1:
            continue

        bins_rek.append(e_rek[mask])
        bins_mc.append(e_mc[mask])

        bin_middles[i] = bin_width * (i+0.5) + min_e

    # mu1 = par("mu1")
    # sigma1 = par("sigma1")
    # model = Normal(diff, mu1, sigma1)

    for i, bin_mc, bin_rek, bin_middle in zip(np.arange(bins),bins_mc, bins_rek, bin_middles):
        rel_diff = (bin_rek-bin_mc)/bin_mc
        rel_diff = rel_diff[abs(rel_diff)<8]
        rel_diff = np.array(rel_diff)
        #print(len(rel_diff))
        fit_entries, fit_edges = np.histogram(rel_diff,
                                              bins=30,
                                              range=[np.min(rel_diff), np.max(rel_diff)],
                                              density=True,
                                              )
        fit_middles = 0.5*(fit_edges[1:] + fit_edges[:-1])
        params, cov = curve_fit(gauss, fit_middles, fit_entries)

        # result = model.fit({"diff": np.array(rel_diff)}, init={"mu1":1, "sigma1":0.5}, method="Powell")

        # sigma = result.x["sigma1"]

        sigma, f_sigma = params[1], np.sqrt(cov[1,1])
        mean, f_mean = params[0], np.sqrt(cov[0,0])

        if f_sigma > 1 or f_mean > 1:
            continue

        res[i]      = sigma
        err_res[i]  = f_sigma
        means[i]    = mean
        err_means[i]= f_mean

        print("bin {:3d} : #={:6d} $/sigma$={:1.2f} +/- {:1.2f} bin_middle={:1.2f}".format(i, len(bin_mc), sigma,f_sigma, bin_middle))
        if plot_hists is True:
            x_plot = np.linspace(-1, 8, 1000)
            plt.figure()
            plt.hist(np.array(rel_diff), 50, [-1, 8], normed=True, histtype='step')
            plt.plot(x_plot, gauss(x_plot, *params), 'r-')
            # plt.savefig("resolution_plots/bin_{}.pdf".format(i))
            # plt.close("all")
            pdf.savefig()
            #plt.show()

        return_dict = dict(
            res         = np.array(res),
            err_res     = np.array(err_res),
            means       = np.array(means),
            err_means   = np.array(err_means),
            bin_middles = np.array(bin_middles),
            bin_width   = np.array(bin_width)
        )

    return return_dict

def calc_binned_charge_dist(true_charge, extracted_charge, mask, width=1):
    max_num_photons = np.max(true_charge)
    rel_errors = list()
    for i in range(0,max_num_photons,width):
        #curr_mask = np.logical_and(true_charge >= i, true_charge<i+width)
        curr_mask = np.logical_and(true_charge > 0, true_charge >= i)
        curr_mask = np.logical_and(curr_mask, true_charge < i+width)
        curr_mask = np.logical_and(curr_mask, mask)
        curr_charge_relError = (extracted_charge[curr_mask] - true_charge[curr_mask])/true_charge[curr_mask]
        rel_errors.append([curr_charge_relError,i,i+width,np.nanmean(curr_charge_relError),np.nanstd(curr_charge_relError) ])
    return rel_errors

def calc_arrival_performace(at_mean_true, at_extracted, true_charge, shower):
    true_size       = list()
    true_num_pix    = list()
    extr_num_pix    = list()
    std_extracted   = list()
    std_true        = list()

    for evtNr in range(len(at_extracted)):
        at_extracted[evtNr] = at_extracted[evtNr] - np.nanmean(at_extracted[evtNr])
        at_mean_true[evtNr] = at_mean_true[evtNr] - np.nanmean(at_mean_true[evtNr])

        curr_at_extracted = at_extracted[evtNr]
        curr_at_mean_true = at_mean_true[evtNr]
        curr_shower       = shower[evtNr]
        curr_charge_true  = true_charge[evtNr]

        curr_true_size    = np.sum(curr_charge_true)
        curr_mask = np.logical_and(curr_charge_true > 3, curr_shower)
        curr_true_num_pix = len(curr_charge_true[curr_mask])
        curr_extr_num_pix = len(curr_shower[curr_shower==True])
        curr_std_extracted = np.std(curr_at_extracted[curr_shower])
        curr_std_true = np.std(curr_at_mean_true[curr_shower])

        true_size.append(curr_true_size)
        true_num_pix.append(curr_true_num_pix)
        extr_num_pix.append(curr_extr_num_pix)
        std_extracted.append(curr_std_extracted)
        std_true.append(curr_std_true)

    return np.array(true_size), np.array(true_num_pix), np.array(extr_num_pix), \
           np.array(std_extracted), np.array(std_true),at_mean_true, at_extracted

def calc_true_num_photons(true_charge, shower):
    true_num_pix    = list()
    extr_num_pix    = list()

    for evtNr in range(len(true_charge)):
        curr_shower       = shower[evtNr]
        curr_charge_true  = true_charge[evtNr]
        curr_mask = np.logical_and(curr_charge_true > 3, curr_shower)
        curr_true_num_pix = len(curr_charge_true[curr_mask])
        curr_extr_num_pix = len(curr_shower[curr_shower==True])

        true_num_pix.append(curr_true_num_pix)
        extr_num_pix.append(curr_extr_num_pix)

    return np.array(true_num_pix), np.array(extr_num_pix)
