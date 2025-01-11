import os
import pickle

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from common_tools import cleaning_image, denoising_image, univ_inv_sol, cleaning_level_significance_default
from scipy.stats import norm
from tqdm.auto import tqdm

from ctapipe.image import (
    hillas_parameters,
)
from ctapipe.image.toymodel import Gaussian
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")

rng = np.random.default_rng(0)

config = {}
config["keep_main"] = True
config["restore"] = True
config["denoise_threshold"] = 7.
config["cleaning_picture"] = 3.
config["cleaning_boundary"] = 2.

cam_type = 'LSTCam'
#cam_type = "NectarCam"
#cam_type = 'FlashCam'
#cam_type = 'DigiCam'
#cam_type = 'ASTRICam'
#cam_type = 'CHEC'
#cam_type = 'SCTCam'

cam = CameraGeometry.from_name(cam_type)
pix_area = float(cam.pix_area.to_value(u.m**2)[0])
pix_width = 0.5 * float(cam.pixel_width.to_value(u.m)[0])

true_width = 2.0 * pix_width * u.m
true_length = 6.0 * true_width
# true_width = 0.05 * u.m
# true_length = 0.3 * u.m
true_psi = 45.0 * u.deg
#true_x = 0.5 * u.m
#true_y = -0.2 * u.m
true_x = 0. * u.m
true_y = 0. * u.m

# test_nsb_level_pe = 3
test_nsb_level_pe = 5

image_intensity_sigma = 20.0
image_intensity = (
    image_intensity_sigma
    * test_nsb_level_pe
    * (true_length.to_value(u.m) * true_width.to_value(u.m) / pix_area)
)
# image_intensity = 2000

n_sample = 1000

model = Gaussian(true_x, true_y, true_length, true_width, true_psi)


#output_filename = f"{ctapipe_output}/output_machines/denoiser_model_{cam_type}.pkl"
#output_filename = f"{ctapipe_output}/output_machines/denoiser_model_LSTCam.pkl"
output_filename = f"{ctapipe_output}/output_machines/denoiser_model_LSTCam_10pe.pkl"
denoiser_model_pkl = None
if not os.path.exists(output_filename):
    print(f"{output_filename} does not exist.")
    exit()
else:
    denoiser_model_pkl = pickle.load(open(output_filename, "rb"))


def sample_no_noise_no_cleaning():
    _, signal, _ = model.generate_image(
        cam, intensity=image_intensity, nsb_level_pe=0, rng=rng
    )

    if np.sum(signal) <= 0.0:
        return None

    h = hillas_parameters(cam, signal)
    return h


def sample_noise_with_cleaning_original():
    image, _, _ = model.generate_image(
        cam, intensity=image_intensity, nsb_level_pe=test_nsb_level_pe, rng=rng
    )

    image_clean = np.zeros_like(image)
    image_mask, noisy_image_mean, noisy_background_mean, noisy_background_rms = cleaning_image(cam, image, image_clean, original_count=True)

    if np.sum(image_clean) <= 0.0:
        return None

    h = hillas_parameters(cam[image_mask], image_clean[image_mask])
    return h


def sample_noise_with_cleaning(tag):
    image, _, _ = model.generate_image(
        cam, intensity=image_intensity, nsb_level_pe=test_nsb_level_pe, rng=rng
    )

    image_clean = np.zeros_like(image)
    image_mask, noisy_image_mean, noisy_background_mean, noisy_background_rms = cleaning_image(cam, image, image_clean, original_count=False)

    if np.sum(image_clean) <= 0.0:
        return None

    h = hillas_parameters(cam[image_mask], image_clean[image_mask])

    if h.psi.to_value(u.deg) - true_psi.to_value(u.deg) > 10.0:
        values = [image, image_clean]
        titles = ["noisy image", "cleaned image"]
        fig, axs = plt.subplots(1, 2, figsize=(2.0 * 6.4, 6.4))
        for ax, trials, title in zip(axs, values, titles):
            display = CameraDisplay(cam, ax=ax)
            display.image = trials
            display.cmap = "inferno"
            display.add_colorbar(ax=ax)
            ax.set_title(title)
        fig.savefig(
            f"{ctapipe_output}/output_plots/image_trial_{tag}_clean.png", dpi=300
        )
        del fig
        del axs
        plt.close()

    return h


def sample_noise_with_denoising(tag):
    image, _, _ = model.generate_image(
        cam, intensity=image_intensity, nsb_level_pe=test_nsb_level_pe, rng=rng
    )

    denoising_signal = np.zeros_like(image)
    denoising_mask = denoising_image(denoiser_model_pkl, cam, image, denoising_signal)

    if np.sum(denoising_signal) <= 0.0:
        return None

    h = hillas_parameters(cam[denoising_mask], denoising_signal[denoising_mask])

    if h.psi.to_value(u.deg) - true_psi.to_value(u.deg) > 10.0:
        values = [image, denoising_signal]
        titles = ["noisy image", "denoised image"]
        fig, axs = plt.subplots(1, 2, figsize=(2.0 * 6.4, 6.4))
        for ax, trials, title in zip(axs, values, titles):
            display = CameraDisplay(cam, ax=ax)
            display.image = trials
            display.cmap = "inferno"
            display.add_colorbar(ax=ax)
            ax.set_title(title)
        fig.savefig(
            f"{ctapipe_output}/output_plots/image_trial_{tag}_denoise.png", dpi=300
        )
        del fig
        del axs
        plt.close()

    return h

def sample_noise_with_universal_denoising(tag):
    image, _, _ = model.generate_image(
        cam, intensity=image_intensity, nsb_level_pe=test_nsb_level_pe, rng=rng
    )

    denoising_signal = np.zeros_like(image)
    denoising_mask, interm_Ys, noisy_image_mean, noisy_background_mean, noisy_background_rms = univ_inv_sol(
        denoiser_model_pkl,
        cam,
        image,
        denoising_signal,
        freq = 1,
        config = config,
    )

    if np.sum(denoising_signal) <= 0.0:
        return None

    h = hillas_parameters(cam[denoising_mask], denoising_signal[denoising_mask])

    if h.psi.to_value(u.deg) - true_psi.to_value(u.deg) > 10.0:
        values = [image, denoising_signal]
        titles = ["noisy image", "denoised image"]
        fig, axs = plt.subplots(1, 2, figsize=(2.0 * 6.4, 6.4))
        for ax, trials, title in zip(axs, values, titles):
            display = CameraDisplay(cam, ax=ax)
            display.image = trials
            display.cmap = "inferno"
            display.add_colorbar(ax=ax)
            ax.set_title(title)
        fig.savefig(
            f"{ctapipe_output}/output_plots/image_trial_{tag}_denoise.png", dpi=300
        )
        del fig
        del axs
        plt.close()

    return h


trials_no_noise_no_cleaning = []
for trial in tqdm(range(n_sample)):
    h = sample_no_noise_no_cleaning()
    if h == None:
        continue
    if np.isnan(h.psi):
        continue
    if np.isnan(h.psi_uncertainty):
        continue
    trials_no_noise_no_cleaning += [h]
trials_noise_cleaning_original = []
for trial in tqdm(range(n_sample)):
    h = sample_noise_with_cleaning_original()
    if h == None:
        continue
    if np.isnan(h.psi):
        continue
    if np.isnan(h.psi_uncertainty):
        continue
    trials_noise_cleaning_original += [h]
trials_noise_cleaning = []
for trial in tqdm(range(n_sample)):
    h = sample_noise_with_cleaning(f"{trial}")
    if h == None:
        continue
    if np.isnan(h.psi):
        continue
    if np.isnan(h.psi_uncertainty):
        continue
    trials_noise_cleaning += [h]
trials_noise_denoising = []
for trial in tqdm(range(n_sample)):
    h = sample_noise_with_universal_denoising(f"{trial}")
    if h == None:
        continue
    if np.isnan(h.psi):
        continue
    if np.isnan(h.psi_uncertainty):
        continue
    trials_noise_denoising += [h]

boundary_significance = cleaning_level_significance_default[0] 
picture_significance = cleaning_level_significance_default[1]

titles = [
    "No Noise, all Pixels",
    f"With Noise ({test_nsb_level_pe} p.e.), Tailcuts({picture_significance}$\sigma$, {boundary_significance}$\sigma$), counting orignal p.e.",
    f"With Noise ({test_nsb_level_pe} p.e.), Tailcuts({picture_significance}$\sigma$, {boundary_significance}$\sigma$), p.e. above {boundary_significance}$\sigma$",
]
values = [
    trials_no_noise_no_cleaning,
    trials_noise_cleaning_original,
    trials_noise_cleaning,
]
fig, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
for ax, trials, title in zip(axs, values, titles):
    pred = np.array([t.psi.to_value(u.deg) - true_psi.to_value(u.deg) for t in trials])
    unc = np.array([t.psi_uncertainty.to_value(u.deg) for t in trials])
    limits = np.quantile(pred, [0.001, 0.999])
    hist, edges, plot = ax.hist(pred, bins=51, range=limits, density=True)
    x = np.linspace(edges[0], edges[-1], 500)
    ax.plot(x, norm.pdf(x, pred.mean(), pred.std()))
    ax.plot(x, norm.pdf(x, 0.0, unc.mean()))
    ax.set_title(title)
axs[len(axs) - 1].set_xlabel("Psi / deg")
fig.savefig(f"{ctapipe_output}/output_plots/hillas_uncertainties_original.png", dpi=300)

titles = [
    "No Noise, all Pixels",
    f"With Noise ({test_nsb_level_pe} p.e.), Tailcuts({picture_significance}$\sigma$, {boundary_significance}$\sigma$), p.e. above {boundary_significance}$\sigma$",
    f"With Noise ({test_nsb_level_pe} p.e.), Denoising, Tailcuts({picture_significance}$\sigma$, {boundary_significance}$\sigma$), p.e. above {boundary_significance}$\sigma$",
]
values = [
    trials_no_noise_no_cleaning,
    trials_noise_cleaning,
    trials_noise_denoising,
]
fig, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
for ax, trials, title in zip(axs, values, titles):
    pred = np.array([t.psi.to_value(u.deg) - true_psi.to_value(u.deg) for t in trials])
    unc = np.array([t.psi_uncertainty.to_value(u.deg) for t in trials])
    limits = np.quantile(pred, [0.001, 0.999])
    hist, edges, plot = ax.hist(pred, bins=51, range=limits, density=True)
    x = np.linspace(edges[0], edges[-1], 500)
    ax.plot(x, norm.pdf(x, pred.mean(), pred.std()))
    ax.plot(x, norm.pdf(x, 0.0, unc.mean()))
    ax.set_title(title)
axs[len(axs) - 1].set_xlabel("Psi / deg")
fig.savefig(f"{ctapipe_output}/output_plots/hillas_uncertainties_denoising.png", dpi=300)
