
import os, sys
import subprocess
import glob

import numpy as np
from scipy.optimize import curve_fit
from astropy import units as u
from matplotlib import pyplot as plt
from matplotlib import colors
import pickle
import math

from ctapipe.utils import Histogram
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter


ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")

fig, ax = plt.subplots()
figsize_x = 8.6
figsize_y = 6.4
fig.set_figheight(figsize_y)
fig.set_figwidth(figsize_x)

ana_tag = 'veritas'

#telescope_type = 'MST_SCT_SCTCam'
telescope_type = 'MST_MST_NectarCam'
#telescope_type = 'MST_MST_FlashCam'
#telescope_type = 'SST_1M_DigiCam'
#telescope_type = 'SST_ASTRI_ASTRICam'
#telescope_type = 'SST_GCT_CHEC'
#telescope_type = 'LST_LST_LSTCam'

#sim_files = 'sct_onaxis_train.txt'
#sim_files = 'sct_onaxis_test.txt'
#sim_files = 'sct_diffuse_all.txt'
#sim_files = 'mst_onaxis_train.txt'
sim_files = 'mst_onaxis_test.txt'
#sim_files = 'mst_diffuse_all.txt'

font = {'family': 'serif', 'color':  'black', 'weight': 'normal', 'size': 10, 'rotation': 0.,}

training_sample_path = []
max_nfiles = 1e10
nfiles = 0

unc_cut = 0.2
#unc_cut = 1e10
hist_edge = 0.04
#list_edge = 1e10
list_edge = 0.04

energy_cut = 0.001
#energy_cut = 1.0

ref_name = []
ref_name += ['template']
ref_name += ['least square']
ref_name += ['least square + template']

list_ref_chi2 = []
list_ref_unc = []
list_ref_off_angle = []
list_ref_off_angle_pass = []
list_ref_off_angle_fail = []
for ref in range(0,len(ref_name)):
    list_ref_chi2 += [None]
    list_ref_unc += [None]
    list_ref_off_angle += [None]
    list_ref_off_angle_pass += [None]
    list_ref_off_angle_fail += [None]

hist_template_norm = Histogram(nbins=(4), ranges=[[-1,1]])
hist_template_off_angle = Histogram(nbins=(4), ranges=[[-1,1]])
hist_template_off_angle_err = Histogram(nbins=(4), ranges=[[-1,1]])
hist_template_energy_resolution = Histogram(nbins=(4), ranges=[[-1,1]])
hist_template_energy_resolution_err = Histogram(nbins=(4), ranges=[[-1,1]])
hist_template_reco_time = Histogram(nbins=(4), ranges=[[-1,1]])
hist_hillas_norm = Histogram(nbins=(4), ranges=[[-1,1]])
hist_hillas_off_angle = Histogram(nbins=(4), ranges=[[-1,1]])
hist_hillas_off_angle_err = Histogram(nbins=(4), ranges=[[-1,1]])
hist_xing_norm = Histogram(nbins=(4), ranges=[[-1,1]])
hist_xing_off_angle = Histogram(nbins=(4), ranges=[[-1,1]])
hist_xing_off_angle_err = Histogram(nbins=(4), ranges=[[-1,1]])
hist_combine_norm = Histogram(nbins=(4), ranges=[[-1,1]])
hist_combine_off_angle = Histogram(nbins=(4), ranges=[[-1,1]])
hist_combine_off_angle_err = Histogram(nbins=(4), ranges=[[-1,1]])

list_hist_norm = []
list_hist_off_angle = []
list_hist_off_angle_err = []
for ref in range(0,len(ref_name)):
    if ref_name[ref] == 'least square':
        list_hist_norm += [hist_xing_norm]
        list_hist_off_angle += [hist_xing_off_angle]
        list_hist_off_angle_err += [hist_xing_off_angle_err]
    if ref_name[ref] == 'least square + template':
        list_hist_norm += [hist_combine_norm]
        list_hist_off_angle += [hist_combine_off_angle]
        list_hist_off_angle_err += [hist_combine_off_angle_err]
    if ref_name[ref] == 'template':
        list_hist_norm += [hist_template_norm]
        list_hist_off_angle += [hist_template_off_angle]
        list_hist_off_angle_err += [hist_template_off_angle_err]

with open(f'{ctapipe_input}/{sim_files}', 'r') as file:
    for line in file:
        #training_sample_path += [get_dataset_path(line.strip('\n'))]
        training_sample_path += [line.strip('\n')]
        nfiles += 1
        if nfiles >= max_nfiles: break

def gauss_func(x,A,sigma):
    return A * np.exp(-((x-0.)**2)/(2*sigma*sigma))

def plot_analysis_result():

    list_hillas_off_angle = []
    list_hillas_off_angle_pass = []
    list_hillas_off_angle_fail = []
    list_hillas_unc = []
    list_hillas_unc_pass = []
    list_hillas_unc_fail = []
    list_xing_off_angle = []
    list_xing_off_angle_pass = []
    list_xing_off_angle_fail = []
    list_xing_unc = []
    list_xing_unc_pass = []
    list_xing_unc_fail = []
    list_template_energy_resolution = []
    list_template_energy_resolution_pass = []
    list_template_energy_resolution_fail = []
    list_template_off_angle = []
    list_template_off_angle_pass = []
    list_template_off_angle_fail = []
    list_template_unc = []
    list_template_unc_pass = []
    list_template_unc_fail = []
    list_combine_off_angle = []
    list_combine_off_angle_pass = []
    list_combine_off_angle_fail = []
    list_combine_unc = []
    list_combine_unc_pass = []
    list_combine_unc_fail = []
    list_tmp_alt = []
    list_tmp_az = []
    list_hillas_truth_log_energy = []
    list_xing_truth_log_energy = []
    list_combine_truth_log_energy = []
    list_template_truth_log_energy = []
    list_template_reco_log_energy = []
    list_template_reco_time = []
    list_hillas_truth_log_energy_pass = []
    list_xing_truth_log_energy_pass = []
    list_combine_truth_log_energy_pass = []
    list_template_truth_log_energy_pass = []
    list_hillas_truth_log_energy_fail = []
    list_xing_truth_log_energy_fail = []
    list_combine_truth_log_energy_fail = []
    list_template_truth_log_energy_fail = []

    for path in range(0,len(training_sample_path)):
    
        #source = SimTelEventSource(training_sample_path[path], focal_length_choice='EQUIVALENT')
        #subarray = source.subarray
        #ob_keys = source.observation_blocks.keys()
        #run_id = list(ob_keys)[0]

        run_id = training_sample_path[path].split("_")[3].strip("run")
        print (f"run_id = {run_id}")
    
        input_filename = f'{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}_{telescope_type}.pkl'
        print (f'loading pickle analysis data: {input_filename}')
        if not os.path.exists(input_filename):
            print (f'file does not exist.')
            continue
        analysis_result = pickle.load(open(input_filename, "rb"))

        hillas_result = analysis_result[0]
        xing_result = analysis_result[1]
        template_result = analysis_result[2]
        combine_result = analysis_result[3]

        for evt in range(0,len(hillas_result)):
            truth_energy = hillas_result[evt][0].value
            if truth_energy<energy_cut: continue
            off_angle = hillas_result[evt][2]
            if math.isnan(off_angle): continue
            unc = hillas_result[evt][3]
            if unc==0.:
                unc = 0.00001
            list_hillas_off_angle += [min(off_angle*off_angle,list_edge-0.001)]
            list_hillas_unc += [unc]
            list_hillas_truth_log_energy += [np.log10(truth_energy)]
            if unc<unc_cut:
                list_hillas_off_angle_pass += [min(off_angle*off_angle,list_edge-0.001)]
                list_hillas_unc_pass += [unc]
                list_hillas_truth_log_energy_pass += [np.log10(truth_energy)]
            else:
                list_hillas_off_angle_fail += [min(off_angle*off_angle,list_edge-0.001)]
                list_hillas_unc_fail += [unc]
                list_hillas_truth_log_energy_fail += [np.log10(truth_energy)]
        for evt in range(0,len(xing_result)):
            truth_energy = xing_result[evt][0].value
            if truth_energy<energy_cut: continue
            off_angle = xing_result[evt][2]
            if math.isnan(off_angle): continue
            unc = xing_result[evt][3]
            list_xing_off_angle += [min(off_angle*off_angle,list_edge-0.001)]
            list_xing_unc += [unc]
            list_xing_truth_log_energy += [np.log10(truth_energy)]
            if unc<unc_cut:
                list_xing_off_angle_pass += [min(off_angle*off_angle,list_edge-0.001)]
                list_xing_unc_pass += [unc]
                list_xing_truth_log_energy_pass += [np.log10(truth_energy)]
            else:
                list_xing_off_angle_fail += [min(off_angle*off_angle,list_edge-0.001)]
                list_xing_unc_fail += [unc]
                list_xing_truth_log_energy_fail += [np.log10(truth_energy)]
        for evt in range(0,len(template_result)):
            truth_energy = template_result[evt][0].value
            if truth_energy<energy_cut: continue
            for img in range(0,len(template_result[evt][7])):
                list_tmp_alt += [template_result[evt][7][img]]
                list_tmp_az += [template_result[evt][8][img]]
            off_angle = template_result[evt][1]
            if math.isnan(off_angle): continue
            unc = template_result[evt][2]
            truth_log_energy = template_result[evt][5].value
            tmp_log_energy = template_result[evt][6].value
            tmp_reco_time = template_result[evt][10]
            list_template_energy_resolution += [abs(pow(10.,tmp_log_energy)-pow(10.,truth_log_energy))/pow(10.,truth_log_energy)]
            list_template_off_angle += [min(off_angle*off_angle,list_edge-0.001)]
            list_template_unc += [unc]
            list_template_truth_log_energy += [np.log10(truth_energy)]
            list_template_reco_log_energy += [tmp_log_energy]
            list_template_reco_time += [tmp_reco_time]
            if unc<unc_cut:
                list_template_off_angle_pass += [min(off_angle*off_angle,list_edge-0.001)]
                list_template_unc_pass += [unc]
                list_template_truth_log_energy_pass += [np.log10(truth_energy)]
                list_template_energy_resolution_pass += [abs(pow(10.,tmp_log_energy)-pow(10.,truth_log_energy))/pow(10.,truth_log_energy)]
            else:
                list_template_off_angle_fail += [min(off_angle*off_angle,list_edge-0.001)]
                list_template_unc_fail += [unc]
                list_template_truth_log_energy_fail += [np.log10(truth_energy)]
                list_template_energy_resolution_fail += [abs(pow(10.,tmp_log_energy)-pow(10.,truth_log_energy))/pow(10.,truth_log_energy)]
        for evt in range(0,len(combine_result)):
            truth_energy = combine_result[evt][0].value
            off_angle = combine_result[evt][1]
            if math.isnan(off_angle): continue
            unc = combine_result[evt][2]
            list_combine_off_angle += [min(off_angle*off_angle,list_edge-0.001)]
            list_combine_unc += [unc]
            list_combine_truth_log_energy += [np.log10(truth_energy)]
            if unc<unc_cut:
                list_combine_off_angle_pass += [min(off_angle*off_angle,list_edge-0.001)]
                list_combine_unc_pass += [unc]
                list_combine_truth_log_energy_pass += [np.log10(truth_energy)]
            else:
                list_combine_off_angle_fail += [min(off_angle*off_angle,list_edge-0.001)]
                list_combine_unc_fail += [unc]
                list_combine_truth_log_energy_fail += [np.log10(truth_energy)]
    
    list_hillas_chi2 = np.array(list_hillas_off_angle_pass)/np.square(np.array(list_hillas_unc_pass))

    for ref in range(0,len(ref_name)):
        if ref_name[ref] == 'least square':
            list_ref_chi2[ref]= np.array(list_xing_off_angle_pass)/np.square(np.array(list_xing_unc_pass))
            list_ref_off_angle[ref] = list_xing_off_angle
            list_ref_unc[ref] = list_xing_unc
            list_ref_off_angle_pass[ref] = list_xing_off_angle_pass
            list_ref_off_angle_fail[ref] = list_xing_off_angle_fail
        if ref_name[ref] == 'least square + template':
            list_ref_chi2[ref] = np.array(list_combine_off_angle_pass)/np.square(np.array(list_combine_unc_pass))
            list_ref_off_angle[ref] = list_combine_off_angle
            list_ref_unc[ref] = list_combine_unc
            list_ref_off_angle_pass[ref] = list_combine_off_angle_pass
            list_ref_off_angle_fail[ref] = list_combine_off_angle_fail
        if ref_name[ref] == 'template':
            list_ref_chi2[ref] = np.array(list_template_off_angle_pass)/np.square(np.array(list_template_unc_pass))
            list_ref_off_angle[ref] = list_template_off_angle
            list_ref_unc[ref] = list_template_unc
            list_ref_off_angle_pass[ref] = list_template_off_angle_pass
            list_ref_off_angle_fail[ref] = list_template_off_angle_fail

    hist_hillas_norm.fill(list_hillas_truth_log_energy_pass)
    hist_hillas_off_angle.fill(list_hillas_truth_log_energy_pass,weights=list_hillas_off_angle_pass)
    hist_hillas_off_angle.data = np.sqrt(hist_hillas_off_angle.data / hist_hillas_norm.data)
    hist_hillas_off_angle_err.data = hist_hillas_off_angle.data / np.sqrt(hist_hillas_norm.data)

    hist_xing_norm.fill(list_xing_truth_log_energy_pass)
    hist_xing_off_angle.fill(list_xing_truth_log_energy_pass,weights=list_xing_off_angle_pass)
    hist_xing_off_angle.data = np.sqrt(hist_xing_off_angle.data / hist_xing_norm.data)
    hist_xing_off_angle_err.data = hist_xing_off_angle.data / np.sqrt(hist_xing_norm.data)

    hist_combine_norm.fill(list_combine_truth_log_energy_pass)
    hist_combine_off_angle.fill(list_combine_truth_log_energy_pass,weights=list_combine_off_angle_pass)
    hist_combine_off_angle.data = np.sqrt(hist_combine_off_angle.data / hist_combine_norm.data)
    hist_combine_off_angle_err.data = hist_combine_off_angle.data / np.sqrt(hist_combine_norm.data)

    hist_template_norm.fill(list_template_truth_log_energy_pass)
    hist_template_off_angle.fill(list_template_truth_log_energy_pass,weights=list_template_off_angle_pass)
    hist_template_off_angle.data = np.sqrt(hist_template_off_angle.data / hist_template_norm.data)
    hist_template_off_angle_err.data = hist_template_off_angle.data / np.sqrt(hist_template_norm.data)
    hist_template_energy_resolution.fill(list_template_truth_log_energy_pass,weights=list_template_energy_resolution_pass)
    hist_template_energy_resolution.data = hist_template_energy_resolution.data / hist_template_norm.data
    hist_template_energy_resolution_err.data = hist_template_energy_resolution.data / np.sqrt(hist_template_norm.data)

    hist_template_reco_time.fill(list_template_truth_log_energy,weights=list_template_reco_time)
    hist_template_reco_time.data = hist_template_reco_time.data / hist_template_norm.data

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "log10 energy [TeV]"
    label_y = "reconstruction time [sec]"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.plot(hist_template_norm.bin_centers(0),hist_template_reco_time.data)
    fig.savefig(
        f"{ctapipe_output}/output_plots/template_reco_time_{telescope_type}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "log10 energy [TeV]"
    label_y = "68% containment [deg]"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.errorbar(hist_hillas_norm.bin_centers(0),hist_hillas_off_angle.data,hist_hillas_off_angle_err.data,label='Hillas')
    for ref in range(0,len(ref_name)):
        ax.errorbar(list_hist_norm[ref].bin_centers(0),list_hist_off_angle[ref].data,list_hist_off_angle_err[ref].data,label=ref_name[ref])
    ax.legend(loc='best')
    ax.set_yscale('log')
    fig.savefig(
        f"{ctapipe_output}/output_plots/analysis_off_angle_{telescope_type}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "log10 energy [TeV]"
    label_y = "68% containment [deg]"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.errorbar(hist_template_norm.bin_centers(0),hist_template_energy_resolution.data,hist_template_energy_resolution_err.data,label='template')
    ax.legend(loc='best')
    fig.savefig(
        f"{ctapipe_output}/output_plots/analysis_energy_resolution_{telescope_type}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

    hillas_mean = np.sqrt(np.mean(np.array(list_hillas_off_angle_pass)))
    hillas_chi2_mean = np.mean(np.array(list_hillas_chi2))
    print (f"hillas_mean = {hillas_mean}")
    print (f"hillas_chi2_mean = {hillas_chi2_mean}")
    for ref in range(0,len(ref_name)):
        ref_mean = np.sqrt(np.mean(np.array(list_ref_off_angle_pass[ref])))
        ref_chi2_mean = np.mean(np.array(list_ref_chi2[ref]))
        print ("============================================")
        print (f"ref_name = {ref_name[ref]}")
        print (f"ref_mean = {ref_mean}")
        print (f"ref_chi2_mean = {ref_chi2_mean}")


    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "sqaured angular distance [$\mathrm{degree}^{2}$]"
    label_y = "count"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.hist(list_hillas_off_angle,histtype='step',bins=25,range=(0.,hist_edge),label='Hillas')
    for ref in range(0,len(ref_name)):
        ax.hist(list_ref_off_angle[ref],histtype='step',bins=25,range=(0.,hist_edge),label=ref_name[ref])
    ax.legend(loc='best')
    fig.savefig(
        f"{ctapipe_output}/output_plots/reconstruction_off_angle_{telescope_type}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "sqaured angular distance [$\mathrm{degree}^{2}$]"
    label_y = "count"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.hist(list_hillas_off_angle_pass,histtype='step',bins=25,range=(0.,hist_edge),label='Hillas')
    for ref in range(0,len(ref_name)):
        ax.hist(list_ref_off_angle_pass[ref],histtype='step',bins=25,range=(0.,hist_edge),label=ref_name[ref])
    ax.legend(loc='best')
    fig.savefig(
        f"{ctapipe_output}/output_plots/reconstruction_off_angle_pass_{telescope_type}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "sqaured angular distance [$\mathrm{degree}^{2}$]"
    label_y = "count"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.hist(list_hillas_off_angle_fail,histtype='step',bins=25,range=(0.,hist_edge),label='Hillas')
    for ref in range(0,len(ref_name)):
        ax.hist(list_ref_off_angle_fail[ref],histtype='step',bins=25,range=(0.,hist_edge),label=ref_name[ref])
    ax.legend(loc='best')
    fig.savefig(
        f"{ctapipe_output}/output_plots/reconstruction_off_angle_fail_{telescope_type}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "chi square"
    label_y = "count"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.hist(list_hillas_chi2,histtype='step',bins=20,range=(0.,5.),label='Hillas')
    for ref in range(0,len(ref_name)):
        ax.hist(list_ref_chi2[ref],histtype='step',bins=20,range=(0.,5.),label=ref_name[ref])
    ax.legend(loc='best')
    fig.savefig(
        f"{ctapipe_output}/output_plots/reconstruction_chi2_{telescope_type}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "X"
    label_y = "Y"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.scatter(
        list_tmp_alt,
        list_tmp_az,
        s=90,
        facecolors="none",
        c="r",
        alpha=0.3,
        marker="+",
    )
    #ax.set_xlim(70.-0.25, 70.+0.25)
    #ax.set_ylim(0.-0.25, 0.+0.25)
    fig.savefig(
        f"{ctapipe_output}/output_plots/template_altaz.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "truth log10 energy (TeV)"
    label_y = "reconstructed log10 energy (TeV)"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.scatter(
        list_template_truth_log_energy,
        list_template_reco_log_energy,
        s=90,
        facecolors="none",
        c="r",
        alpha=0.3,
        marker="+",
    )
    fig.savefig(
        f"{ctapipe_output}/output_plots/template_log_energy.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

plot_analysis_result()
