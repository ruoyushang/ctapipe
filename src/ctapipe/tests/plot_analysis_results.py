import os
import pickle

import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")
print(f"ctapipe_output = {ctapipe_output}")


list_denoise_tag = []
list_denoise_tag += ['test1']
#list_denoise_tag += ['test2']
#list_denoise_tag += ['test3']

array_type = 'SCT'
#array_type = "Nectar"
#array_type = 'Flash'
#array_type = 'LST'
#array_type = 'ASTRI'
#array_type = 'CHEC'
#array_type = 'Digi'
#array_type = 'LST_Nectar_ASTRI'

pointing = "onaxis"
#pointing = 'diffuse'

template = "yes"
# template = 'no'

ana_tag = f"psi_unc_{array_type}_{pointing}"

image_size_bins = []
image_size_bins += [0.0]
#image_size_bins += [50.]
#image_size_bins += [250.]
#image_size_bins += [1250.]
#image_size_bins += [6250.]
image_size_bins += [1e10]

truth_energy_bins = []
truth_energy_bins += [0.1]
truth_energy_bins += [1.0]
truth_energy_bins += [100.]

error_limit = 1.0


sim_files = None
bkg_files = None
if "SCT" in ana_tag:
    bkg_files = "sct_proton.txt"
    if "onaxis" in ana_tag:
        # sim_files = 'sct_onaxis_train.txt'
        # sim_files = 'sct_onaxis_test.txt'
        sim_files = "sct_onaxis_all.txt"
    else:
        sim_files = "sct_diffuse_all.txt"
else:
    bkg_files = "mst_proton.txt"
    if "onaxis" in ana_tag:
        # sim_files = 'mst_onaxis_train.txt'
        # sim_files = 'mst_onaxis_test.txt'
        sim_files = "mst_onaxis_all.txt"
    else:
        # sim_files = 'mst_diffuse_test.txt'
        sim_files = "mst_diffuse_all.txt"


training_sample_path = []
with open(f"{ctapipe_input}/{sim_files}", "r") as file:
    for line in file:
        training_sample_path += [line.strip("\n")]

training_bkg_sample_path = []
# with open(f'{ctapipe_input}/{bkg_files}', 'r') as file:
#    for line in file:
#        training_bkg_sample_path += [line.strip('\n')]

combine_sample_path = [training_sample_path]
# combine_sample_path = [training_sample_path,training_bkg_sample_path]

for denoise_tag in list_denoise_tag:

    print("==========================================================================")
    print (denoise_tag)

    for b in range(0, len(image_size_bins) - 1):
        list_psi_err = []
        list_psi_unc = []
        list_psi_significance = []
        list_image_size = []
        list_image_snr = []
        list_length = []
        list_width = []
        list_length_width_ratio = []
        list_kurtosis = []
        for path in range(0, len(training_sample_path)):
            run_id = training_sample_path[path].split("_")[3].strip("run")
    
            input_filename = (
                f"{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}_{denoise_tag}_mono.pkl"
            )
            if not os.path.exists(input_filename):
                continue
            analysis_result = pickle.load(open(input_filename, "rb"))
            for evt in range(0, len(analysis_result)):
                psi_error = analysis_result[evt][0]
                psi_uncertainty = analysis_result[evt][1]
                run_id = analysis_result[evt][2]
                event_id = analysis_result[evt][3]
                tel_id = analysis_result[evt][4]
                image_size = analysis_result[evt][5]
                image_snr = analysis_result[evt][6]
                frac_leakage_intensity = analysis_result[evt][7]
                length = analysis_result[evt][8]
                width = analysis_result[evt][9]
                kurtosis = analysis_result[evt][10]
                # print (f"psi_error = {psi_error:0.4f}, psi_uncertainty = {psi_uncertainty:0.4f}, image_size = {image_size:0.1f}")
                if image_size <= image_size_bins[b]:
                    continue
                if image_size > image_size_bins[b + 1]:
                    continue
                if width==0.: 
                    continue
                # if frac_leakage_intensity>0.: continue
                # if frac_leakage_intensity==0.: continue
                # if frac_leakage_intensity>0.01: continue
                # if frac_leakage_intensity<0.06: continue
                # if frac_leakage_intensity>0.07: continue
    
                list_psi_err += [psi_error * 180.0 / np.pi]
                list_psi_unc += [psi_uncertainty * 180.0 / np.pi]
                list_psi_significance += [psi_error/psi_uncertainty]
                list_image_size += [image_size]
                list_image_snr += [image_snr]
                list_length += [length]
                list_width += [width]
                list_length_width_ratio += [width/length]
                list_kurtosis += [kurtosis]
    
        mean_psi_unc = np.mean(list_psi_unc)
        rms_psi_unc = np.sqrt(np.mean(np.square(list_psi_unc - mean_psi_unc)))
        mean_psi_err = np.mean(list_psi_err)
        rms_psi_err = np.sqrt(np.mean(np.square(list_psi_err - mean_psi_err)))
    
        #list_matrics = list_psi_err
        #matric_range = [0,40]
        #matric_title = "observed error [deg]"
        list_matrics = list_psi_significance
        matric_range = [0,10]
        matric_title = "observed error significance"
    
        variables = []
        titles = []
        names = []
        ranges = []
        variables += [list_psi_unc]
        titles += ["estimated uncertainty [deg]"]
        names += ["psi_unc"]
        ranges += [[0.,20.]]
        variables += [list_image_size]
        titles += ["image size [p.e.]"]
        names += ["image_size"]
        ranges += [[0.,100.]]
        variables += [list_image_snr]
        titles += ["image signal-to-noise ratio"]
        names += ["image_snr"]
        ranges += [[3.,6.]]
        variables += [list_length_width_ratio]
        titles += ["image width-to-length ratio"]
        names += ["length_width_ratio"]
        ranges += [[0.,1.]]
        variables += [list_kurtosis]
        titles += ["image kurtosis"]
        names += ["kurtosis"]
        ranges += [[1.,11.]]
    
        for variable, title, name, rng in zip(variables, titles, names, ranges):
            fig, ax = plt.subplots()
            figsize_x = 6.4
            figsize_y = 6.4
            fig.set_figheight(figsize_y)
            fig.set_figwidth(figsize_x)
            label_x = matric_title
            label_y = title
            ax.set_xlabel(label_x)
            ax.set_ylabel(label_y)
            max_var = rng[1]
            min_var = rng[0]
            max_err = matric_range[1]
            min_err = matric_range[0]
            ax.scatter(
                list_matrics,
                variable,
                s=90,
                facecolors="none",
                c="r",
                alpha=0.2,
                marker="+",
            )
            n_intervals = 10
            for u in range(0, n_intervals):
                var_axis = []
                err_mean = []
                new_list_psi_err = []
                delta_var = (max_var - min_var) / float(n_intervals)
                lower_var = float(u) * delta_var + min_var
                upper_var = float(u + 1) * delta_var + min_var
                for evt in range(0, len(list_matrics)):
                    var = variable[evt]
                    err = list_matrics[evt]
                    if var < lower_var or var > upper_var:
                        continue
                    new_list_psi_err += [err]
                mean = np.sqrt(np.mean(np.square(new_list_psi_err)))
                var_axis += [0.5 * (lower_var + upper_var)]
                err_mean += [0.0]
                var_axis += [0.5 * (lower_var + upper_var)]
                err_mean += [1.0 * mean]
                ax.plot(err_mean, var_axis, color="k", marker="|")
            ax.set_xlim(0.0, max_err)
            ax.set_ylim(0.0, max_var)
            fig.savefig(
                f"{ctapipe_output}/output_plots/psi_err_vs_{name}_{ana_tag}_{denoise_tag}_size{b}.png",
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
        label_x = "$\chi^{2}$ (error / uncertainty)"
        label_y = "count"
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        list_chi2 = []
        for evt in range(0, len(list_psi_err)):
            err = list_psi_err[evt]
            unc = list_psi_unc[evt]
            list_chi2 += [pow(err / unc, 2)]
        list_chi2 = np.array(list_chi2)
        hist = ax.hist(list_chi2, bins=100, range=[0.0, 10.0])
        Sum = float(len(list_chi2))
        Mean = 0.0
        Variance = 0.0
        for entry in range(0, len(hist[0])):
            X = hist[1][entry]
            f = hist[0][entry] / Sum
            Mean += X * f
            Variance += X * X * f
        Variance = Variance - Mean * Mean
        print(f"Sum = {Sum}")
        print(f"Mean = {Mean:0.4f}")
        print(f"Variance = {Variance:0.4f}")
        ax.set_yscale("log")
        txt_height = np.max(hist[0])
        plt.text(
            0.4, txt_height, f"mean = {Mean:0.4f}, variance = {Variance:0.4f}", fontsize=12
        )
        fig.savefig(
            f"{ctapipe_output}/output_plots/psi_chi2_dist_{ana_tag}_{denoise_tag}_size{b}.png",
            bbox_inches="tight",
        )
        del fig
        del ax
        plt.close()


def find_best_unc_cut(combine_sample_path,denoise_tag):

    cut_limit_lower = 0.1
    cut_limit_upper = 10.*error_limit
    cut_interval = 0.1*error_limit

    default_count = 0.0
    for path in range(0, len(combine_sample_path)):
        for sample in range(0, len(combine_sample_path[path])):
            run_id = combine_sample_path[path][sample].split("_")[3].strip("run")
            input_filename = f"{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}_{denoise_tag}_array_default.pkl"
            if not os.path.exists(input_filename):
                continue
            analysis_result = pickle.load(open(input_filename, "rb"))
            for evt in range(0, len(analysis_result)):
                default_src_loc_err = analysis_result[evt][0]
                default_src_loc_unc = analysis_result[evt][1]
                truth_energy_tev = analysis_result[evt][2]
                if truth_energy_tev <= truth_energy_bins[b]:
                    continue
                if truth_energy_tev > truth_energy_bins[b + 1]:
                    continue
                if default_src_loc_err > error_limit:
                    continue
                default_count += 1.0

    min_diff_count = 1e10
    best_cut = 1e10
    cut = cut_limit_lower
    while cut < cut_limit_upper:
        new_count = 0.0
        for path in range(0, len(combine_sample_path)):
            for sample in range(0, len(combine_sample_path[path])):
                run_id = combine_sample_path[path][sample].split("_")[3].strip("run")
                input_filename = f"{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}_{denoise_tag}_array_new.pkl"
                if not os.path.exists(input_filename):
                    continue
                analysis_result = pickle.load(open(input_filename, "rb"))
                for evt in range(0, len(analysis_result)):
                    new_src_loc_err = analysis_result[evt][0]
                    new_src_loc_unc = analysis_result[evt][1]
                    truth_energy_tev = analysis_result[evt][2]
                    if truth_energy_tev <= truth_energy_bins[b]:
                        continue
                    if truth_energy_tev > truth_energy_bins[b + 1]:
                        continue
                    if new_src_loc_err > error_limit:
                        continue
                    if new_src_loc_unc > cut:
                        continue
                    new_count += 1.0

        diff_count = abs(new_count - default_count)
        if diff_count < min_diff_count:
            min_diff_count = diff_count
            best_cut = cut

        cut += cut_interval

    return best_cut

def compute_angular_resolution(list_events):

    total_events = float(len(list_events))
    PSF_default = 0.0
    for entry in range(0, len(list_events)):
        X = list_events[entry]
        PSF_default += X * X
    PSF_default = pow(PSF_default/total_events, 0.5)
    return PSF_default


for denoise_tag in list_denoise_tag:

    print("==========================================================================")
    print (denoise_tag)

    for b in range(0, len(truth_energy_bins) - 1):
        best_unc_cut = find_best_unc_cut(combine_sample_path,denoise_tag)
    
        list_default_src_loc_err = []
        list_default_src_loc_unc = []
        list_new_src_loc_err = []
        list_new_src_loc_unc = []
    
        for path in range(0, len(combine_sample_path)):
            for sample in range(0, len(combine_sample_path[path])):
                run_id = combine_sample_path[path][sample].split("_")[3].strip("run")
    
                input_filename = f"{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}_{denoise_tag}_array_default.pkl"
                if not os.path.exists(input_filename):
                    continue
                analysis_result = pickle.load(open(input_filename, "rb"))
                for evt in range(0, len(analysis_result)):
                    default_src_loc_err = analysis_result[evt][0]
                    default_src_loc_unc = analysis_result[evt][1]
                    truth_energy_tev = analysis_result[evt][2]
                    if truth_energy_tev <= truth_energy_bins[b]:
                        continue
                    if truth_energy_tev > truth_energy_bins[b + 1]:
                        continue
                    if default_src_loc_err>max(1.0,error_limit):
                        continue
                    if default_src_loc_unc == 0.0:
                        default_src_loc_unc = 0.3
                    list_default_src_loc_err += [default_src_loc_err]
                    list_default_src_loc_unc += [default_src_loc_unc]
    
                input_filename = (
                    f"{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}_{denoise_tag}_array_new.pkl"
                )
                if not os.path.exists(input_filename):
                    continue
                analysis_result = pickle.load(open(input_filename, "rb"))
                for evt in range(0, len(analysis_result)):
                    new_src_loc_err = analysis_result[evt][0]
                    new_src_loc_unc = analysis_result[evt][1]
                    truth_energy_tev = analysis_result[evt][2]
                    if truth_energy_tev <= truth_energy_bins[b]:
                        continue
                    if truth_energy_tev > truth_energy_bins[b + 1]:
                        continue
                    if new_src_loc_err>max(1.0,error_limit):
                        continue
                    if new_src_loc_unc > best_unc_cut:
                        continue
                    list_new_src_loc_err += [new_src_loc_err]
                    list_new_src_loc_unc += [new_src_loc_unc]
    
        default_weights = []
        default_statistics = []
        total_default_weight = 0.0
        for evt in range(0, len(list_default_src_loc_unc)):
            total_default_weight += 1.0
            default_weights += [1.0]
            default_statistics += [1.0]
        default_weights = np.array(default_weights) / total_default_weight
        default_statistics = np.array(default_statistics) / (
            total_default_weight * total_default_weight
        )
    
        new_weights = []
        new_statistics = []
        total_new_weight = 0.0
        for evt in range(0, len(list_new_src_loc_unc)):
            total_new_weight += 1.0
            new_weights += [1.0]
            new_statistics += [1.0]
        new_weights = np.array(new_weights) / total_new_weight
        new_statistics = np.array(new_statistics) / (total_new_weight * total_new_weight)
    
        fig, ax = plt.subplots()
        figsize_x = 6.4
        figsize_y = 4.6
        fig.set_figheight(figsize_y)
        fig.set_figwidth(figsize_x)
        label_x = "angular distance from source [deg]"
        label_y = "count"
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        PSF_default = compute_angular_resolution(list_default_src_loc_err)
        PSF_new = compute_angular_resolution(list_new_src_loc_err)
        print(f"PSF_default = {PSF_default:0.4f} deg")
        print(f"PSF_new = {PSF_new:0.4f} deg")
        hist_default = ax.hist(
            list_default_src_loc_err,
            bins=50,
            range=[0.0, error_limit],
            alpha=1.0,
            histtype='step',
            label=f"default tailcut (PSF = {PSF_default:0.2f} deg)",
        )
        hist_new = ax.hist(
            list_new_src_loc_err,
            bins=50,
            range=[0.0, error_limit],
            alpha=1.0,
            histtype='step',
            label=f"denoising + tailcut (PSF = {PSF_new:0.2f} deg)",
        )
        ax.legend(loc="best")
        ax.set_yscale("log")
        ax.set_title(f"{truth_energy_bins[b]}-{truth_energy_bins[b+1]} TeV")
        fig.savefig(
            f"{ctapipe_output}/output_plots/angular_error_dist_{ana_tag}_{denoise_tag}_energy{b}.png",
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
        label_x = "log10 angular uncertainty [deg]"
        label_y = "count"
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        hist = ax.hist(
            np.log10(list_default_src_loc_unc),
            bins=20,
            range=[-2.0, 2.0],
            alpha=0.5,
            label="default method",
        )
        hist = ax.hist(
            np.log10(list_new_src_loc_unc),
            bins=20,
            range=[-2.0, 2.0],
            alpha=0.5,
            label="least-square method",
        )
        ax.legend(loc="best")
        fig.savefig(
            f"{ctapipe_output}/output_plots/angular_uncertainty_dist_{ana_tag}_{denoise_tag}_energy{b}.png",
            bbox_inches="tight",
        )
        del fig
        del ax
        plt.close()
