import numpy as np
import torch
import torch.nn as nn
import time

from ctapipe.image import (
    number_of_islands,
    tailcuts_clean,
)

cleaning_level = {
    "DigiCam": (3, 5, 2),
    "ASTRICam": (3, 5, 2),
    "CHEC": (3, 5, 2),
    "LSTCam": (3, 5, 2),
    "FlashCam": (3, 5, 2),
    "NectarCam": (3, 5, 2),
    "SCTCam": (3, 5, 2),
}
#cleaning_level_significance_default = (4, 5, 2)
cleaning_level_significance_default = (3, 4, 2)
#cleaning_level_significance_default = (2, 3, 2)
# cleaning_level_significance_default = (1, 2, 2)
cleaning_level_significance = {
    "DigiCam": cleaning_level_significance_default,
    "ASTRICam": cleaning_level_significance_default,
    "CHEC": cleaning_level_significance_default,
    "LSTCam": cleaning_level_significance_default,
    "FlashCam": cleaning_level_significance_default,
    "NectarCam": cleaning_level_significance_default,
    "SCTCam": cleaning_level_significance_default,
}


class DenoisingCNN(nn.Module):
    def __init__(self):
        super(DenoisingCNN, self).__init__()

        #Relationship between padding and kernel_size:
        #For kernel_size=3:
        #padding=1 is commonly used to keep the output size the same as the input size.
        #For kernel_size=5:
        #padding=2 is commonly used to keep the output size the same as the input size.
        #padding='same': Calculates the padding required to keep the output size the same as the input size.
        #To calculate the padding required for 'same' behavior, you can use the following formula: padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding='same')
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding='same')

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        #x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x


def keep_main_island(geometry, image_mask):
    n_islands, island_labels = number_of_islands(geometry, image_mask)

    max_island_id = 0
    max_n_pix = 0
    for i in range(1,n_islands+1):
        n_pix = 0
        for pix in range(0, len(island_labels)):
            if island_labels[pix]==i:
                n_pix += 1
        if n_pix>max_n_pix:
            max_n_pix = n_pix
            max_island_id = i

    for pix in range(0, len(island_labels)):
        if island_labels[pix] == max_island_id:
            image_mask[pix] = True
        else:
            image_mask[pix] = False


def dynamic_cleaning(
    geometry,
    image,
    boundary_significance,
    picture_significance,
    min_neighbors_significance,
):

    boundary, picture, min_neighbors = cleaning_level[geometry.name]
    image_mask = tailcuts_clean(
        geometry,
        image,
        boundary_thresh=boundary,
        picture_thresh=picture,
        min_number_picture_neighbors=min_neighbors,
    )

    # first pass, calculate initial mask
    night_sky_pixels = 0.0
    night_sky_mean = 0.0
    night_sky_rms = 0.0
    for pix in range(0, len(image)):
        if not image_mask[pix]:
            night_sky_pixels += 1.0
            night_sky_mean += image[pix]
    if night_sky_pixels > 0.0:
        night_sky_mean = night_sky_mean / (night_sky_pixels)
        for pix in range(0, len(image)):
            if not image_mask[pix]:
                night_sky_rms += pow(image[pix] - night_sky_mean, 2)
        night_sky_rms = pow(night_sky_rms / (night_sky_pixels), 0.5)

    boundary = boundary_significance * night_sky_rms + night_sky_mean
    picture = picture_significance * night_sky_rms + night_sky_mean

    image_mask = tailcuts_clean(
        geometry,
        image,
        boundary_thresh=boundary,
        picture_thresh=picture,
        min_number_picture_neighbors=min_neighbors_significance,
    )

    # second pass, calculate refined mask
    night_sky_pixels = 0.0
    night_sky_mean = 0.0
    night_sky_rms = 0.0
    image_pixels = 0.0
    image_mean = 0.0
    for pix in range(0, len(image_mask)):
        if not image_mask[pix]:
            night_sky_pixels += 1.0
            night_sky_mean += image[pix]
        else:
            image_pixels += 1.0
            image_mean += image[pix]
    if night_sky_pixels > 0.0:
        night_sky_mean = night_sky_mean / (night_sky_pixels)
        for pix in range(0, len(image_mask)):
            if not image_mask[pix]:
                night_sky_rms += pow(image[pix] - night_sky_mean, 2)
        night_sky_rms = pow(night_sky_rms / (night_sky_pixels), 0.5)
    if image_pixels > 0.0:
        image_mean = image_mean / image_pixels

    boundary = boundary_significance * night_sky_rms + night_sky_mean
    picture = picture_significance * night_sky_rms + night_sky_mean

    image_mask = tailcuts_clean(
        geometry,
        image,
        boundary_thresh=boundary,
        picture_thresh=picture,
        min_number_picture_neighbors=min_neighbors_significance,
    )
    keep_main_island(geometry, image_mask)

    return image_mask, night_sky_mean, night_sky_rms, image_mean


def cleaning_image(geometry, noisy_image, clean_image, original_count=False):

    (
        boundary_significance,
        picture_significance,
        min_neighbors_significance,
    ) = cleaning_level_significance[geometry.name]
    image_mask, night_sky_mean, night_sky_rms, image_mean = dynamic_cleaning(
        geometry,
        noisy_image,
        boundary_significance,
        picture_significance,
        min_neighbors_significance,
    )

    for pix in range(0, len(clean_image)):
        if image_mask[pix]:
            if original_count:
                clean_image[pix] = max(0.0, noisy_image[pix])
            else:
                clean_image[pix] = max(
                    0.0,
                    noisy_image[pix]
                    - night_sky_mean
                    - boundary_significance * night_sky_rms,
                )

    noisy_image_mean = 0.
    image_npix = 0.
    noisy_background_mean = 0.
    noisy_background_rms = 0.
    background_npix = 0.
    for pix in range(0, len(image_mask)):
        if image_mask[pix] == False:
            noisy_background_mean += noisy_image[pix]
            background_npix += 1.
        else:
            noisy_image_mean += noisy_image[pix]
            image_npix += 1.
    if background_npix>0. and image_npix>0.:
        noisy_background_mean = noisy_background_mean/background_npix
        noisy_image_mean = noisy_image_mean/image_npix
        for pix in range(0, len(image_mask)):
            if image_mask[pix] == False:
                noisy_background_rms += pow(noisy_image[pix]-noisy_background_mean,2)
        noisy_background_rms = pow(noisy_background_rms/background_npix,0.5)
    else:
        noisy_image_mean = 0.
        noisy_background_mean = 0.
        noisy_background_rms = 1.

    return image_mask, noisy_image_mean, noisy_background_mean, noisy_background_rms


def denoising_image(denoiser_model_pkl, geometry, noisy_image, clean_image, apply_tailcut=True):
    clean_signal = np.zeros_like(noisy_image)
    clean_mask = cleaning_image(geometry, noisy_image, clean_signal)

    with torch.no_grad():
        (
            boundary_significance,
            picture_significance,
            min_neighbors_significance,
        ) = cleaning_level_significance[geometry.name]

        denoiser_model_pkl.eval()
        noisy_signal_2d = np.array(
            [geometry.image_to_cartesian_representation(noisy_image)]
        )
        for ch in range(0, len(noisy_signal_2d)):
            for pix_x in range(0, len(noisy_signal_2d[ch])):
                for pix_y in range(0, len(noisy_signal_2d[ch][pix_x])):
                    if np.isnan(noisy_signal_2d[ch][pix_x][pix_y]):
                        noisy_signal_2d[ch][pix_x][pix_y] = 0.0
        denoising_2d = denoiser_model_pkl(
            torch.from_numpy(noisy_signal_2d.astype(np.float32))
        )
        denoising_1d = geometry.image_from_cartesian_representation(
            np.array(denoising_2d[0])
        )

        denoise_image_1d = np.zeros_like(noisy_image)
        for pix in range(0, len(denoise_image_1d)):
            denoise_image_1d[pix] = denoising_1d[pix]

        denoise_mask, denoise_night_sky_mean, denoise_night_sky_rms, denoise_image_mean = dynamic_cleaning(
            geometry,
            denoise_image_1d,
            boundary_significance,
            picture_significance,
            min_neighbors_significance,
        )

        for pix in range(0, len(clean_image)):
            if apply_tailcut and denoise_mask[pix] == False:
                clean_image[pix] = 0.0
            else:
                clean_image[pix] = denoise_image_1d[pix]

        #for pix in range(0, len(clean_image)):
        #    if denoise_mask[pix] == False:
        #        clean_image[pix] = 0.0

        return denoise_mask


def univ_inv_sol(model, geometry, noisy_image, clean_image, h0=0.3, freq=1, threshold=5.):
    """
    @h0: 1st step size
    """

    noisy_signal_2d = np.array(
        [geometry.image_to_cartesian_representation(noisy_image)]
    )
    n_ch = len(noisy_signal_2d)
    n_pix_x = len(noisy_signal_2d[0])
    n_pix_y = len(noisy_signal_2d[0][0])
    for ch in range(0, len(noisy_signal_2d)):
        for pix_x in range(0, len(noisy_signal_2d[ch])):
            for pix_y in range(0, len(noisy_signal_2d[ch][pix_x])):
                if np.isnan(noisy_signal_2d[ch][pix_x][pix_y]):
                    noisy_signal_2d[ch][pix_x][pix_y] = 0.0

    intermed_Ys = []

    # initialize y
    y = torch.from_numpy(noisy_signal_2d.astype(np.float32))
    y_1d = geometry.image_from_cartesian_representation(
        np.array(y[0])
    )

    (
        boundary_significance,
        picture_significance,
        min_neighbors_significance,
    ) = cleaning_level_significance[geometry.name]
    denoise_mask, denoise_night_sky_mean, denoise_night_sky_rms, denoise_image_mean = dynamic_cleaning(
        geometry,
        y_1d,
        boundary_significance,
        picture_significance,
        min_neighbors_significance,
    )

    N = n_ch*n_pix_x*n_pix_y
    sigma_0 = torch.norm(y) / np.sqrt(N)

    if freq > 0:
        intermed_Ys.append(y_1d)

    with torch.no_grad():
        f_y_0 = model(y)
        d = f_y_0 - y

    sigma = torch.norm(d) / np.sqrt(N)
    sigma_minus1 = sigma
    denoise_night_sky_mean_minus1 = denoise_night_sky_mean
    denoise_night_sky_rms_minus1 = denoise_night_sky_rms
    #print (f"sigma = {sigma}, ratio = {sigma/sigma_0}")

    (
        boundary_significance,
        picture_significance,
        min_neighbors_significance,
    ) = cleaning_level_significance[geometry.name]

    t = 1
    start_time_total = time.time()
    #while t < 10:
    #while sigma/sigma_0 > 0.05:
    while denoise_image_mean/denoise_night_sky_rms<threshold*boundary_significance:
    #while sigma/sigma_0 > 0.1 and denoise_image_mean/denoise_night_sky_rms<threshold*boundary_significance:

        #h = h0 * t / (1 + (h0 * (t - 1)))
        h = h0
        #h = 1.0
        with torch.no_grad():
            f_y = model(y)

        d = f_y - y

        sigma = torch.norm(d) / np.sqrt(N)
        if sigma>sigma_minus1:
            break
        sigma_minus1 = sigma

        #beta = 0.9
        #gamma = sigma*np.sqrt(((1 - (beta*h))**2 - (1-h)**2 ))
        #noise = torch.randn(n_ch, n_pix_x, n_pix_y)
        #y = y + h*d + gamma*noise
        y = y + h*d
        y = y.to(torch.float32)


        y_1d = geometry.image_from_cartesian_representation(
            np.array(y[0])
        )

        denoise_mask, denoise_night_sky_mean, denoise_night_sky_rms, denoise_image_mean = dynamic_cleaning(
            geometry,
            y_1d,
            boundary_significance,
            picture_significance,
            min_neighbors_significance,
        )
        #if denoise_night_sky_mean>denoise_night_sky_mean_minus1:
        #    break
        denoise_night_sky_mean_minus1 = denoise_night_sky_mean
        denoise_night_sky_rms_minus1 = denoise_night_sky_rms

        for pix in range(0, len(denoise_mask)):
            if denoise_mask[pix] == True:
                y_1d[pix] = 1.0*noisy_image[pix] + 0.0*y_1d[pix]
        for pix in range(0,len(y_1d)):
            if y_1d[pix]<0.:
                y_1d[pix] = 0.

        if freq > 0 and t % freq == 0:
            #print("-----------------------------", t)
            #print(f"sigma {sigma.item()}, nsb_mean = {denoise_night_sky_mean}, nsb_rms = {denoise_night_sky_rms}")
            intermed_Ys.append(y_1d)

        t += 1

    #print("-------- total number of iterations, ", t)
    #print(
    #    "-------- average time per iteration (s), ",
    #    np.round((time.time() - start_time_total) / (t - 1), 4),
    #)

    # denoised_y = y - model(y)
    denoised_y = y

    denoised_y_1d = geometry.image_from_cartesian_representation(
        np.array(denoised_y[0])
    )

    denoise_mask, denoise_night_sky_mean, denoise_night_sky_rms, denoise_image_mean = dynamic_cleaning(
        geometry,
        denoised_y_1d,
        boundary_significance,
        picture_significance,
        min_neighbors_significance,
    )

    noisy_image_mean = 0.
    image_npix = 0.
    noisy_background_mean = 0.
    noisy_background_rms = 0.
    background_npix = 0.
    for pix in range(0, len(denoise_mask)):
        if denoise_mask[pix] == False:
            noisy_background_mean += noisy_image[pix]
            background_npix += 1.
        else:
            noisy_image_mean += noisy_image[pix]
            image_npix += 1.
    if background_npix>0. and image_npix>0.:
        noisy_background_mean = noisy_background_mean/background_npix
        noisy_image_mean = noisy_image_mean/image_npix
        for pix in range(0, len(denoise_mask)):
            if denoise_mask[pix] == False:
                noisy_background_rms += pow(noisy_image[pix]-noisy_background_mean,2)
        noisy_background_rms = pow(noisy_background_rms/background_npix,0.5)
    else:
        noisy_image_mean = 0.
        noisy_background_mean = 0.
        noisy_background_rms = 1.

    for pix in range(0, len(denoise_mask)):
        if denoise_mask[pix] == False:
            denoised_y_1d[pix] = 0.0
            clean_image[pix] = 0.0
        else:
            denoised_y_1d[pix] = max(0.0,denoised_y_1d[pix] - denoise_night_sky_mean - boundary_significance * denoise_night_sky_rms)
            clean_image[pix] = max(0.0,noisy_image[pix] - denoise_night_sky_mean - boundary_significance * denoise_night_sky_rms)

    #denoised_norm = np.sum(denoised_y_1d)
    #original_norm = np.sum(clean_image)
    #for pix in range(0, len(denoise_mask)):
    #    clean_image[pix] = denoised_y_1d[pix] * original_norm/denoised_norm

    return denoise_mask, intermed_Ys, noisy_image_mean, noisy_background_mean, noisy_background_rms

