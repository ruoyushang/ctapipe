import os
import pickle
import time

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from common_tools import DenoisingCNN, cleaning_image, denoising_image, univ_inv_sol, dynamic_cleaning, cleaning_level_significance
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from ctapipe.calib import CameraCalibrator
from ctapipe.image import (
    ImageProcessor,
    number_of_islands,
    tailcuts_clean,
    hillas_parameters,
)
from ctapipe.image.toymodel import Gaussian
from ctapipe.instrument import CameraGeometry
from ctapipe.io import SimTelEventSource, EventSource
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.visualization import CameraDisplay
from ctapipe.reco import ShowerProcessor

ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")
ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")

#train_model = True
train_model = False
#toy_test = True
toy_test = False
sim_test = True
#sim_test = False

#array_type = 'LSTCam'
#array_type = "NectarCam"
#array_type = 'FlashCam'
#array_type = 'DigiCam'
array_type = 'ASTRICam'
#array_type = 'CHEC'
#array_type = 'SCTCam'

#pointing = "proton"
pointing = "onaxis"
#pointing = 'diffuse'

select_evt = None
# run_id, event_id = 402, 21104 # good example
#run_id = 402
#event_id = 24606
#select_evt = [run_id, event_id]

ana_tag = f"psi_unc_{array_type}_{pointing}"

telescope_type = []
if "SCT" in ana_tag:
    telescope_type += ["MST_SCT_SCTCam"]
if "Nectar" in ana_tag:
    telescope_type += ["MST_MST_NectarCam"]
if "Flash" in ana_tag:
    telescope_type += ["MST_MST_FlashCam"]
if "LST" in ana_tag:
    telescope_type += ["LST_LST_LSTCam"]
if "ASTRI" in ana_tag:
    telescope_type += ["SST_ASTRI_ASTRICam"]
if "CHEC" in ana_tag:
    telescope_type += ["SST_GCT_CHEC"]
if "DigiCam" in ana_tag:
    telescope_type += ["SST_1M_DigiCam"]

rng = np.random.default_rng(0)
cam = CameraGeometry.from_name(array_type)
pix_x_max = np.max(cam.pix_x.to_value(u.m))
pix_y_max = np.max(cam.pix_y.to_value(u.m))

true_width = 0.05 * u.m
true_length = 0.3 * u.m
true_psi = 45 * u.deg
true_x = 0.5 * u.m
true_y = -0.2 * u.m

image_intensity = 1000
#nsb_level_pe = 3
nsb_level_pe = 5
#nsb_level_pe = 10

# n_sample = 10
n_sample = 100000

font = {
    "family": "serif",
    "color": "black",
    "weight": "normal",
    "size": 10,
    # "rotation": 0.0,
}


def loop_all_events(
    ana_tag,
    training_sample_path,
    ctapipe_output,
    list_telescope_type,
    denoiser_model,
    select_evt=None,
):
    print(f"loading file: {training_sample_path}")
    source = SimTelEventSource(training_sample_path, focal_length_choice="EQUIVALENT")
    #source = EventSource(training_sample_path, focal_length_choice="EQUIVALENT")

    list_tel_id = []
    for tel_idx in range(0, source.subarray.n_tels):
        tel_id = source.subarray.tel_ids[tel_idx]
        telescope_type = str(source.subarray.tel[tel_id])
        if telescope_type not in list_telescope_type:
            continue
        list_tel_id += [tel_id]
    new_subarray = source.subarray.select_subarray(list_tel_id)

    # Explore the instrument description
    subarray_table = new_subarray.to_table()
    nlines = len(subarray_table) + 10
    subarray_table.pprint(nlines)
    # print(new_subarray.to_table())

    calib = CameraCalibrator(subarray=source.subarray)
    image_processor = ImageProcessor(subarray=source.subarray)
    shower_processor = ShowerProcessor(subarray=source.subarray)

    ob_keys = source.observation_blocks.keys()
    run_id = list(ob_keys)[0]

    if not select_evt == None:
        select_run_id = select_evt[0]
        if run_id != select_run_id:
            return

    tel_pointing_alt = float(
        source.observation_blocks[run_id].subarray_pointing_lat / u.rad
    )
    tel_pointing_az = float(
        source.observation_blocks[run_id].subarray_pointing_lon / u.rad
    )
    print(f"tel_pointing_alt = {tel_pointing_alt}")
    print(f"tel_pointing_az = {tel_pointing_az}")

    for event in source:
        event_id = event.index["event_id"]

        if not select_evt == None:
            select_event_id = select_evt[1]
            if event_id != select_event_id:
                continue

        truth_alt = float(event.simulation.shower.alt / u.rad)
        truth_az = float(event.simulation.shower.az / u.rad)
        truth_energy = float(event.simulation.shower.energy / u.TeV)

        #if truth_energy<1.0: continue

        calib(event)  # fills in r1, dl0, and dl1
        image_processor(event)
        shower_processor(event)

        #print (f"event.trigger.tels_with_trigger = {event.trigger.tels_with_trigger}") 

        for tel_idx in range(0, len(list(event.dl0.tel.keys()))):
            tel_id = list(event.dl0.tel.keys())[tel_idx]
            telescope_type = str(source.subarray.tel[tel_id])
            if telescope_type not in list_telescope_type:
                continue
            print(f"analyzing run_id = {run_id}, event_id = {event_id}, tel_id = {tel_id}")

            geometry = source.subarray.tel[tel_id].camera.geometry
            focal_length = float(
                source.subarray.tel[tel_id].optics.equivalent_focal_length / u.m
            )

            truth_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
            #print (f"event.simulation.tel[tel_id] = {event.simulation.tel[tel_id]}")
            if array_type == 'SCTCam':
                for pix in range(0, len(truth_image_1d)):
                    truth_image_1d[pix] = event.simulation.tel[tel_id].true_image[pix]

            noisy_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
            for pix in range(0, len(noisy_image_1d)):
                noisy_image_1d[pix] = event.dl1.tel[tel_id].image[pix]

            (
                boundary_significance,
                picture_significance,
                min_neighbors_significance,
            ) = cleaning_level_significance[geometry.name]
            init_mask, night_sky_mean, night_sky_rms, init_image_mean = dynamic_cleaning(
                geometry,
                noisy_image_1d,
                boundary_significance,
                picture_significance,
                min_neighbors_significance,
            )

            #if init_image_mean/night_sky_rms<1.: continue
            #if init_image_mean/night_sky_rms>10.: continue


            tailcut_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
            tailcut_image_mask, noisy_image_mean, noisy_background_mean, noisy_background_rms = cleaning_image(
                geometry,
                noisy_image_1d,
                tailcut_image_1d, 
                original_count=False,
            )

            denoiser_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
            denoiser_image_mask = denoising_image(
                denoiser_model_pkl,
                geometry,
                noisy_image_1d,
                denoiser_image_1d,
                apply_tailcut=False,
            )

            clean_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
            universal_image_mask, interm_Ys, noisy_image_mean, noisy_background_mean, noisy_background_rms = univ_inv_sol(
                denoiser_model_pkl,
                geometry,
                noisy_image_1d,
                clean_image_1d,
                freq = 1,
            )
            image_snr = (noisy_image_mean-noisy_background_mean)/noisy_background_rms

            if image_snr<1.5: continue

            pix_width = float(geometry.pixel_width.to_value(u.m)[0])

            hillas_results = hillas_parameters(geometry, clean_image_1d)
            intensity = hillas_results["intensity"]
            length = hillas_results["length"].to_value(u.m)
            width = hillas_results["width"].to_value(u.m)
            cog_x = hillas_results["x"].to_value(u.m)
            cog_y = hillas_results["y"].to_value(u.m)
            psi = hillas_results["psi"].to_value(u.rad)
            psi_uncertainty = hillas_results["psi_uncertainty"].to_value(u.rad)
            transverse_cog_uncertainty = hillas_results[
                "transverse_cog_uncertainty"
            ].to_value(u.m)

            if np.isnan(psi):
                continue
            if np.isnan(psi_uncertainty):
                continue

            #if length / pix_width < 2.:
            #    continue
            #if len(interm_Ys)<2:
            #    continue

            print ("making plot...")

            middle_step_1 = min(int(0.2*float(len(interm_Ys))),len(interm_Ys)-1)
            middle_step_2 = min(int(0.4*float(len(interm_Ys))),len(interm_Ys)-1)
            middle_step_3 = min(int(0.6*float(len(interm_Ys))),len(interm_Ys)-1)
            middle_step_4 = min(int(0.8*float(len(interm_Ys))),len(interm_Ys)-1)
            values = []
            titles = []
            values += [[noisy_image_1d, interm_Ys[middle_step_1], interm_Ys[middle_step_2]]]
            titles += [["noisy image", f"t={middle_step_1} step", f"t={middle_step_2} step"]]
            values += [[interm_Ys[middle_step_3], interm_Ys[middle_step_4], interm_Ys[len(interm_Ys)-1]]]
            titles += [[f"t={middle_step_3} step", f"t={middle_step_4} step", f"t={len(interm_Ys)-1} step"]]
            fig, axs = plt.subplots(2, 3, figsize=(3.0 * 6.4, 2.0 * 4.8))
            for ax1, trials1, title1 in zip(axs, values, titles):
                for ax2, trials2, title2 in zip(ax1, trials1, title1):
                    if len(trials2)==0: continue
                    display = CameraDisplay(geometry, ax=ax2)
                    display.image = trials2
                    display.cmap = "Reds"
                    ax2.set_title(title2)
            fig.savefig(
                f"{ctapipe_output}/output_plots/image_{array_type}_run{run_id}_evt{event_id}_tel{tel_id}_iteration.png", 
                dpi=300,
                bbox_inches="tight",
            )
            del fig
            del axs
            plt.close()

            values = []
            titles = []
            values += [noisy_image_1d, denoiser_image_1d]
            titles += ["noisy image", "denoised image"]
            fig, axs = plt.subplots(1, 2, figsize=(2.0 * 6.4, 1.0 * 4.8))
            for ax1, trials1, title1 in zip(axs, values, titles):
                display = CameraDisplay(geometry, ax=ax1)
                display.image = trials1
                display.cmap = "Reds"
                ax1.set_title(title1)
            fig.savefig(
                f"{ctapipe_output}/output_plots/image_{array_type}_run{run_id}_evt{event_id}_tel{tel_id}_original.png", 
                dpi=300,
                bbox_inches="tight",
            )
            del fig
            del axs
            plt.close()

            values = []
            titles = []
            masks = []
            if array_type == 'SCTCam':
                values += [truth_image_1d, denoiser_image_1d, interm_Ys[len(interm_Ys)-1]]
                titles += ["truth image", "original denoiser", "universal denoiser"]
                masks += [tailcut_image_mask, universal_image_mask, universal_image_mask]
            else:
                values += [noisy_image_1d, denoiser_image_1d, interm_Ys[len(interm_Ys)-1]]
                titles += ["noisy image", "original denoiser", "universal denoiser"]
                masks += [tailcut_image_mask, denoiser_image_mask, universal_image_mask]
            fig, axs = plt.subplots(1, 3, figsize=(3.0 * 6.4, 1.0 * 4.8))
            for ax1, trials1, masks1, title1 in zip(axs, values, masks, titles):
                display = CameraDisplay(geometry, ax=ax1)
                display.image = trials1
                display.cmap = "Reds"
                ax1.set_title(title1)
                display.highlight_pixels(
                    masks1, color="xkcd:green", linewidth=0.5, alpha=1.0
                )
            fig.savefig(
                f"{ctapipe_output}/output_plots/image_{array_type}_run{run_id}_evt{event_id}_tel{tel_id}_universal.png", 
                dpi=300,
                bbox_inches="tight",
            )
            del fig
            del axs
            plt.close()



class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, predictions, targets):
        loss = torch.mean((predictions - targets) ** 2)
        return loss


def to_log_scale_image(image):
    new_image = np.zeros_like(image)
    for pix in range(0, len(image)):
        data = image[pix]
        log_data = 0.0
        if data > 0.0:
            log_data = np.log10(data)
        new_image[pix] = log_data
    return new_image


def reverse_log_scale_image(image):
    new_image = np.zeros_like(image)
    for pix in range(0, len(image)):
        log_data = image[pix]
        data = pow(10.0, log_data)
        new_image[pix] = data
    return new_image


# Convert the training and testing data from NumPy arrays to PyTorch tensors
class ConvertData(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.Y = torch.from_numpy(Y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


### While PyTorch doesn't have a built-in 1D denoiser, you can easily build one using a few different approaches:
### 1. Autoencoder:
### Encoder: This part of the network compresses the noisy 1D signal into a lower-dimensional representation (latent space).
### Decoder: This part reconstructs the original signal from the latent representation.
### By training the autoencoder to reconstruct the clean signal from noisy inputs, it learns to remove noise.
###
### 2. Denoising Diffusion Probabilistic Models (DDPMs):
### DDPMs gradually add noise to a clean signal over multiple steps.
### A neural network is then trained to reverse this process by predicting the noise at each step.
### This allows the model to generate clean signals from pure noise.
### While DDPMs are typically used for 2D images, they can be adapted for 1D signals as well.
###
### 3. Convolutional Neural Networks (CNNs):
### Even though your data is 1D, you can still use 1D CNNs for denoising.
### CNNs are good at extracting local features and patterns, which can be helpful for removing noise.


class DenoisingAutoencoder(nn.Module):
    def __init__(self, image_dim):
        super().__init__()
        hidden_dim = int(np.sqrt(float(image_dim)))
        self.encoder = nn.Sequential(
            nn.Linear(image_dim, image_dim), nn.ReLU(), nn.Linear(image_dim, hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, image_dim), nn.ReLU(), nn.Linear(image_dim, image_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# ... Define your model, dataset, and dataloaders ...


def sample_with_noise(nsb_level_pe=5, length=0.0, width=0.0, image_intensity_sigma=0.0):

    pix_area = float(cam.pix_area.to_value(u.m**2)[0])
    pix_width = 0.5 * float(cam.pixel_width.to_value(u.m)[0])

    if length > 0.0:
        true_length = length * pix_width * u.m
    else:
        true_length = 20.0 * pix_width * u.m
    if width > 0.0:
        true_width = width * pix_width * u.m
    else:
        true_width = 1.0 * pix_width * u.m
    if image_intensity_sigma == 0.0:
        image_intensity = (
            np.random.uniform(low=10.0, high=20.0, size=None)
            * nsb_level_pe
            * (true_length.to_value(u.m) * true_width.to_value(u.m) / pix_area)
        )
    else:
        image_intensity = (
            image_intensity_sigma
            * nsb_level_pe
            * (true_length.to_value(u.m) * true_width.to_value(u.m) / pix_area)
        )
    true_psi = np.random.uniform(low=0.0, high=360.0, size=None) * u.deg
    true_x = (
        np.random.uniform(low=-0.7 * pix_x_max, high=0.7 * pix_x_max, size=None) * u.m
    )
    true_y = (
        np.random.uniform(low=-0.7 * pix_y_max, high=0.7 * pix_y_max, size=None) * u.m
    )

    model = Gaussian(true_x, true_y, true_length, true_width, true_psi)
    noisy_image, clean_image, _ = model.generate_image(
        cam, intensity=image_intensity, nsb_level_pe=nsb_level_pe, rng=rng
    )

    return [noisy_image, clean_image]


def keep_main_island(geometry, image_mask):
    n_islands = number_of_islands(geometry, image_mask)
    for pix in range(0, len(n_islands[1])):
        if n_islands[1][pix] == 1:
            image_mask[pix] = True
        else:
            image_mask[pix] = False


if train_model:
    training_images = [
        sample_with_noise(nsb_level_pe=nsb_level_pe) for _ in tqdm(range(n_sample))
    ]
    noisy_images = []
    clean_images = []
    for trial in range(0, len(training_images)):
        noisy_signal = training_images[trial][0]
        clean_signal = training_images[trial][1]
        noisy_signal_2d = np.array(
            [cam.image_to_cartesian_representation(noisy_signal)]
        )
        clean_signal_2d = np.array(
            [cam.image_to_cartesian_representation(clean_signal)]
        )
        for ch in range(0, len(noisy_signal_2d)):
            for pix_x in range(0, len(noisy_signal_2d[ch])):
                for pix_y in range(0, len(noisy_signal_2d[ch][pix_x])):
                    if np.isnan(noisy_signal_2d[ch][pix_x][pix_y]):
                        noisy_signal_2d[ch][pix_x][pix_y] = 0.0
                    if np.isnan(clean_signal_2d[ch][pix_x][pix_y]):
                        clean_signal_2d[ch][pix_x][pix_y] = 0.0
        noisy_images += [noisy_signal_2d]
        clean_images += [clean_signal_2d]
    noisy_images = np.array(noisy_images)
    clean_images = np.array(clean_images)

    train_data = ConvertData(noisy_images, clean_images)
    batch_size = 100
    train_dataloader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True
    )

    n_pix = len(cam.pix_x)
    # denosing_model = DenoisingAutoencoder(n_pix)
    denosing_model = DenoisingCNN()
    # optimizer = optim.Adam(denosing_model.parameters(), lr=0.001)
    optimizer = optim.SGD(denosing_model.parameters(), lr=0.001)
    # criterion = nn.MSELoss()
    criterion = CustomMSELoss()

    loss_values = []
    epochs = 20
    for epoch in range(epochs):
        for X, Y in train_dataloader:
            noisy_signal = X
            clean_signal = Y
            optimizer.zero_grad()
            output = denosing_model(noisy_signal)
            loss = criterion(output, clean_signal)
            loss.backward()
            optimizer.step()
            loss_values.append(loss.item())
            print(f"epoch = {epoch}, loss.item() = {loss.item()}")
    print("Training Complete")
    output_filename = (
        f"{ctapipe_output}/output_machines/denoiser_model_{array_type}.pkl"
    )
    with open(output_filename, "wb") as file:
        pickle.dump(denosing_model, file)

    fig, ax = plt.subplots()
    figsize_x = 6.4
    figsize_y = 4.6
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "steps"
    label_y = "loss"
    ax.set_xlabel(label_x, fontdict=font)
    ax.set_ylabel(label_y, fontdict=font)
    ax.plot(loss_values)
    ax.set_yscale("log")
    fig.savefig(
        f"{ctapipe_output}/output_plots/training_loss.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

    exit()

#output_filename = f"{ctapipe_output}/output_machines/denoiser_model_{array_type}.pkl"
#output_filename = f"{ctapipe_output}/output_machines/denoiser_model_NectarCam.pkl"
output_filename = f"{ctapipe_output}/output_machines/denoiser_model_LSTCam.pkl"
if not os.path.exists(output_filename):
    print(f"{output_filename} does not exist.")
    exit()
else:
    print (f"read {output_filename}...")
    denoiser_model_pkl = pickle.load(open(output_filename, "rb"))


if toy_test:
    test_images = [
        sample_with_noise(nsb_level_pe=nsb_level_pe)
        #sample_with_noise(nsb_level_pe=0.5*nsb_level_pe,image_intensity_sigma=10.)
        #sample_with_noise(nsb_level_pe=nsb_level_pe,image_intensity_sigma=15.)
        #sample_with_noise(nsb_level_pe=nsb_level_pe,length=12.,width=2.,image_intensity_sigma=20.)
        for _ in tqdm(range(10))
    ]

    for trial in range(0, len(test_images)):
        noisy_signal = test_images[trial][0]
        truth_signal = test_images[trial][1]

        tailcut_signal = np.zeros_like(noisy_signal)
        tailcut_mask, noisy_image_mean, noisy_background_mean, noisy_background_rms = cleaning_image(cam, noisy_signal, tailcut_signal, original_count=False)

        denoising_signal = np.zeros_like(noisy_signal)
        denoising_mask = denoising_image(
            denoiser_model_pkl, 
            cam, 
            noisy_signal, 
            denoising_signal, 
            apply_tailcut=False,
        )

        mask_correlation = 0.0
        mask_correlation_norm = 0.0
        for pix in range(0, len(tailcut_mask)):
            if denoising_mask[pix] and tailcut_mask[pix]:
                mask_correlation += 1.0
            if denoising_mask[pix] or tailcut_mask[pix]:
                mask_correlation_norm += 1.0
        print(f"mask_correlation = {mask_correlation}")

        univ_inv_sol_image = np.zeros_like(noisy_signal)
        univ_inv_sol_mask, interm_Ys, noisy_image_mean, noisy_background_mean, noisy_background_rms = univ_inv_sol(
               denoiser_model_pkl,
               cam,
               noisy_signal,
               univ_inv_sol_image,
               freq = 1
        )
        image_snr = (noisy_image_mean-noisy_background_mean)/noisy_background_rms

        middle_step = int(0.5*float(len(interm_Ys)))+1
        values = []
        titles = []
        values += [[noisy_signal, truth_signal, tailcut_signal]]
        titles += [["noisy image", "truth image", "tailcut on original image"]]
        values += [[interm_Ys[1], interm_Ys[len(interm_Ys)-1], univ_inv_sol_image]]
        titles += [["t=1 step", f"t={len(interm_Ys)-1} step", "tailcut on denoised image"]]
        fig, axs = plt.subplots(2, 3, figsize=(3.0 * 6.4, 2.0 * 4.8))
        for ax1, trials1, title1 in zip(axs, values, titles):
            for ax2, trials2, title2 in zip(ax1, trials1, title1):
                display = CameraDisplay(cam, ax=ax2)
                display.image = trials2
                display.cmap = "Reds"
                ax2.set_title(title2)
        fig.savefig(
            f"{ctapipe_output}/output_plots/image_trial_{trial}_universal_vs_tailcut.png", 
            dpi=300,
            bbox_inches="tight",
        )
        del fig
        del axs
        plt.close()

        middle_step = int(0.5*float(len(interm_Ys)))+1
        values = []
        titles = []
        values += [noisy_signal, truth_signal, denoising_signal]
        titles += ["noisy image", "truth image", "denoised image"]
        fig, axs = plt.subplots(1, 3, figsize=(3.0 * 6.4, 1.0 * 4.8))
        for ax1, trials1, title1 in zip(axs, values, titles):
            display = CameraDisplay(cam, ax=ax1)
            display.image = trials1
            display.cmap = "Reds"
            ax1.set_title(title1)
        fig.savefig(
            f"{ctapipe_output}/output_plots/image_trial_{trial}_universal_vs_original.png", 
            dpi=300,
            bbox_inches="tight",
        )
        del fig
        del axs
        plt.close()


if sim_test:
    sim_files = None
    if "SCT" in ana_tag:
        if "proton" in ana_tag:
            sim_files = "sct_proton.txt"
        elif "onaxis" in ana_tag:
            # sim_files = 'sct_onaxis_train.txt'
            # sim_files = 'sct_onaxis_test.txt'
            sim_files = "sct_onaxis_all.txt"
        else:
            sim_files = "sct_diffuse_all.txt"
    else:
        if "proton" in ana_tag:
            sim_files = "mst_proton.txt"
        elif "onaxis" in ana_tag:
            # sim_files = 'mst_onaxis_train.txt'
            sim_files = "mst_onaxis_test.txt"
            # sim_files = 'mst_onaxis_all.txt'
        else:
            sim_files = "mst_diffuse_test.txt"

    with open(f"{ctapipe_input}/{sim_files}", "r") as file:
        for line in file:
            training_sample_path = get_dataset_path(line.strip("\n"))

            run_id = line.split("_")[3].strip("run")
            print(f"run_id = {run_id}")

            loop_all_events(
                ana_tag,
                training_sample_path,
                ctapipe_output,
                telescope_type,
                denoiser_model_pkl,
                select_evt=select_evt,
            )

            #exit()
