import math
import pickle
import time

import matplotlib.animation as animation
import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, angular_separation
from astropy.time import Time
from matplotlib import pyplot as plt

from ctapipe.calib import CameraCalibrator
from ctapipe.coordinates import CameraFrame
from ctapipe.image import ImageProcessor, tailcuts_clean
from ctapipe.io import SimTelEventSource
from ctapipe.reco import ShowerProcessor
from ctapipe.visualization import CameraDisplay

__all__ = [
    "image_translation_and_rotation",
]

# unoptimized cleaning levels
cleaning_level = {
    "CHEC": (2, 4, 2),
    "LSTCam": (3.5, 7, 2),
    "FlashCam": (3.5, 7, 2),
    "NectarCam": (4, 8, 2),
    # "SCTCam": (5, 8, 2),
    "SCTCam": (2, 5, 2),
}

# image_size_cut = 30.0
image_size_cut = 100.0
plot_image_size_cut = 200.0

n_bins_arrival = 20
arrival_lower = 0.0
arrival_upper = 0.4
n_bins_impact = 20
impact_lower = 0.0
impact_upper = 800.0
n_bins_energy = 15
log_energy_lower = -1.0
log_energy_upper = 2.0

n_samples_per_window = 2
total_samples = 64
select_samples = 16

select_event_id = 0
# select_event_id = 27007


def image_translation_and_rotation(
    geometry, list_input_image_1d, shift_x, shift_y, angle_rad
):
    """
    Function to perform rotation and translation of a list of images.

    Parameters
    ----------
    geometry:
        geometry of camera
    list_input_image_1d: ndarray
        Array of 1-D images
    shift_x: float
        Translation of position in x coordinates
    shift_y: float
        Translation of position in y coordinates
    angle_rad: float
        Rotation angle of pixels in rad

    Returns
    -------
    list_output_image_1d: ndarray
        Array of 1-D images

    """

    pixel_width = float(geometry.pixel_width[0] / u.m)
    rotation_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )

    all_coord_x = []
    all_coord_y = []
    for pix in range(0, len(list_input_image_1d[0])):
        x = float(geometry.pix_x[pix] / u.m)
        y = float(geometry.pix_y[pix] / u.m)
        all_coord_x += [x]
        all_coord_y += [y]

    list_output_image_1d = []
    for img in range(0, len(list_input_image_1d)):
        old_coord_x = []
        old_coord_y = []
        old_image = []
        for pix in range(0, len(list_input_image_1d[img])):
            x = all_coord_x[pix]
            y = all_coord_y[pix]
            if list_input_image_1d[img][pix] == 0.0:
                continue
            old_coord_x += [x]
            old_coord_y += [y]
            old_image += [list_input_image_1d[img][pix]]

        trans_x = []
        trans_y = []
        for pix in range(0, len(old_coord_x)):
            x = old_coord_x[pix]
            y = old_coord_y[pix]
            new_x = x - shift_x
            new_y = y - shift_y
            trans_x += [new_x]
            trans_y += [new_y]

        rotat_x = []
        rotat_y = []
        for pix in range(0, len(old_coord_x)):
            x = trans_x[pix]
            y = trans_y[pix]
            initi_coord = np.array([y, x])
            rotat_coord = rotation_matrix @ initi_coord
            new_x = float(rotat_coord[0])
            new_y = float(rotat_coord[1])
            rotat_x += [new_x]
            rotat_y += [-new_y]

        output_image_1d = np.zeros_like(list_input_image_1d[img])
        for pix1 in range(0, len(old_coord_x)):
            min_dist = 1e10
            nearest_pix = 0
            for pix2 in range(0, len(list_input_image_1d[img])):
                x = all_coord_x[pix2]
                y = all_coord_y[pix2]
                if abs(x - rotat_x[pix1]) > pixel_width:
                    continue
                if abs(y - rotat_y[pix1]) > pixel_width:
                    continue
                dist = (x - rotat_x[pix1]) * (x - rotat_x[pix1]) + (
                    y - rotat_y[pix1]
                ) * (y - rotat_y[pix1])
                if min_dist > dist:
                    min_dist = dist
                    nearest_pix = pix2
            output_image_1d[nearest_pix] += old_image[pix1]
        list_output_image_1d += [output_image_1d]

    return list_output_image_1d


def remove_nan_pixels(image_2d):
    num_rows, num_cols = image_2d.shape
    for x_idx in range(0, num_cols):
        for y_idx in range(0, num_rows):
            if math.isnan(image_2d[y_idx, x_idx]):
                image_2d[y_idx, x_idx] = 0.0


def reset_time(input_image_1d, input_time_1d):
    center_time = 0.0
    image_size = 0.0
    for pix in range(0, len(input_image_1d)):
        if input_image_1d[pix] == 0.0:
            continue
        image_size += input_image_1d[pix]
        center_time += input_time_1d[pix] * input_image_1d[pix]

    if image_size == 0.0:
        return 0.0

    center_time = center_time / image_size

    for pix in range(0, len(input_image_1d)):
        if input_image_1d[pix] == 0.0:
            continue
        input_time_1d[pix] += -1.0 * center_time

    return center_time


def least_square_fit(power, input_data, target_data, weight):
    # solve x*A = y using SVD
    # y_{0} = ( x_{0,0} x_{0,1} ... 1 )  a_{0}
    # y_{1} = ( x_{1,0} x_{1,1} ... 1 )  a_{1}
    # y_{2} = ( x_{2,0} x_{2,1} ... 1 )  .
    #                                    b

    # for a line equation y = a*x + b
    # solve x*A = y using SVD
    # y_{0} = ( x_{0} 1 )  a
    # y_{1} = ( x_{1} 1 )  b
    # y_{2} = ( x_{2} 1 )

    x = []
    y = []
    w = []
    for evt in range(0, len(input_data)):
        input_x = []
        for p in range(0, power):
            input_x += [pow(input_data[evt], p)]
        x += [input_x]
        y += [target_data[evt]]
        w += [weight[evt]]
    x = np.array(x)
    y = np.array(y)
    w = np.diag(w)

    # Have a look: https://en.wikipedia.org/wiki/Weighted_least_squares
    # Compute the weighted SVD
    U, S, Vt = np.linalg.svd(x.T @ w @ x, full_matrices=False)
    # Calculate the weighted pseudo-inverse
    S_pseudo_inv = np.diag(1 / S)
    for entry in range(0, len(S)):
        if S[entry] / S[0] < 1e-5:
            S_pseudo_inv[entry, entry] = 0.0
    x_pseudo_inv = (Vt.T @ S_pseudo_inv @ U.T) @ x.T
    # Compute the weighted least-squares solution
    A_svd = x_pseudo_inv @ (w @ y)
    # Compute parameter error
    A_cov = Vt.T @ S_pseudo_inv @ U.T
    A_err = np.sqrt(np.diag(A_cov))

    # Compute chi
    chi = (x.dot(A_svd) - y) @ np.sqrt(w)
    chi2 = np.sum(np.square(chi))

    return A_svd, A_err, chi2


def fit_image_to_line(geometry, image_input_1d, transpose=False):
    x = []
    y = []
    w = []
    for pix in range(0, len(image_input_1d)):
        if image_input_1d[pix] == 0.0:
            continue
        if not transpose:
            x += [float(geometry.pix_x[pix] / u.m)]
            y += [float(geometry.pix_y[pix] / u.m)]
            w += [image_input_1d[pix]]
        else:
            x += [float(geometry.pix_y[pix] / u.m)]
            y += [float(geometry.pix_x[pix] / u.m)]
            w += [image_input_1d[pix]]
    x = np.array(x)
    y = np.array(y)
    w = np.array(w)

    # if np.sum(w)==0.:
    #    return 0., 0., np.inf

    avg_x = np.sum(w * x) / np.sum(w)
    avg_y = np.sum(w * y) / np.sum(w)

    pix_width = float(geometry.pixel_width[0] / u.m)
    x_rms = 0.0
    y_rms = 0.0
    weight = 0.0
    for pix in range(0, len(image_input_1d)):
        if image_input_1d[pix] == 0.0:
            continue
        if not transpose:
            x_rms += (
                pow(float(geometry.pix_x[pix] / u.m) - avg_x, 2) * image_input_1d[pix]
            )
            y_rms += (
                pow(float(geometry.pix_y[pix] / u.m) - avg_y, 2) * image_input_1d[pix]
            )
            weight += image_input_1d[pix]
        else:
            x_rms += (
                pow(float(geometry.pix_y[pix] / u.m) - avg_y, 2) * image_input_1d[pix]
            )
            y_rms += (
                pow(float(geometry.pix_x[pix] / u.m) - avg_x, 2) * image_input_1d[pix]
            )
            weight += image_input_1d[pix]
    x_rms = pow(x_rms / weight, 0.5)
    y_rms = pow(y_rms / weight, 0.5)
    projection = y_rms / x_rms

    pix_width = float(geometry.pixel_width[0] / u.m)
    xc = []
    yc = []
    wc = []
    for pix in range(0, len(image_input_1d)):
        if image_input_1d[pix] == 0.0:
            continue
        if not transpose:
            xc += [float(geometry.pix_x[pix] / u.m)]
            yc += [float(geometry.pix_y[pix] / u.m)]
        else:
            xc += [float(geometry.pix_y[pix] / u.m)]
            yc += [float(geometry.pix_x[pix] / u.m)]
        wc += [image_input_1d[pix] / (pix_width * pix_width)]
    xc = np.array(xc)
    yc = np.array(yc)
    wc = np.array(wc)

    a_mtx, a_mtx_err, chi2 = least_square_fit(2, xc, yc, wc)
    fit_a = a_mtx[1]
    fit_a_err = 1.0 * a_mtx_err[1]
    fit_b = a_mtx[0]
    fit_b_err = 1.0 * a_mtx_err[0]

    return fit_a, fit_b, fit_a_err, fit_b_err, chi2 * projection


def find_intersection_multiple_lines(
    list_x,
    list_y,
    list_a,
    list_b,
    list_a_err,
    list_b_err,
    list_intensity,
    list_length,
    list_width,
):
    # y = a*x + b, weight = 1./b_err

    x = np.array(list_x)
    y = np.array(list_y)
    a = np.array(list_a)
    b = np.array(list_b)
    a_err = np.array(list_a_err)
    b_err = np.array(list_b_err)
    intensity = np.array(list_intensity)
    length = np.array(list_length)
    width = np.array(list_width)

    # avg_open_angle = 0.
    # for i1 in range(0, len(a) - 1):
    #    for i2 in range(i1 + 1, len(a)):
    #        open_angle = abs(np.arctan(a[i1]) - np.arctan(a[i2]))
    #        avg_open_angle += open_angle
    # avg_open_angle = avg_open_angle/float(len(a)-1)

    pair_weight = []
    pair_x = []
    pair_y = []
    pair_err = []
    for i1 in range(0, len(a) - 1):
        for i2 in range(i1 + 1, len(a)):
            pair_a = np.array([a[i1], a[i2]])
            pair_b = np.array([b[i1], b[i2]])
            pair_w = np.array([1.0, 1.0])
            open_angle = abs(np.arctan(a[i1]) - np.arctan(a[i2]))
            x_mtx, x_mtx_err, chi2 = least_square_fit(2, pair_a, pair_b, pair_w)
            pair_fit_x = -x_mtx[1]
            pair_fit_y = -x_mtx[0]
            dist_sq_1 = (pair_fit_x - x[i1]) * (pair_fit_x - x[i1]) + (
                pair_fit_y - y[i1]
            ) * (pair_fit_y - y[i1])
            dist_sq_2 = (pair_fit_x - x[i2]) * (pair_fit_x - x[i2]) + (
                pair_fit_y - y[i2]
            ) * (pair_fit_y - y[i2])
            pair_fit_err = a_err[i1] * a_err[i1] * dist_sq_1 + b_err[i1] * b_err[i1]
            pair_fit_err += a_err[i2] * a_err[i2] * dist_sq_2 + b_err[i2] * b_err[i2]
            pair_fit_err = pow(pair_fit_err, 0.5)
            pair_x += [pair_fit_x]
            pair_y += [pair_fit_y]
            pair_err += [pair_fit_err * 1.0 / abs(np.sin(open_angle))]
            pair_weight += [
                (intensity[i1] * length[i1] / width[i1])
                * (intensity[i2] * length[i2] / width[i2])
            ]

    pair_x = np.array(pair_x)
    pair_y = np.array(pair_y)
    pair_err = np.array(pair_err)
    pair_weight = np.array(pair_weight)

    fit_x = 0.0
    fit_y = 0.0
    fit_err = 0.0
    fit_weight = 0.0
    for xing in range(0, len(pair_x)):
        error_sq = pair_err[xing] * pair_err[xing]
        weight = 1.0 / error_sq
        # weight = pair_weight[xing]
        fit_weight += weight
        fit_x += pair_x[xing] * weight
        fit_y += pair_y[xing] * weight
        fit_err += pair_err[xing] * pair_err[xing] * weight
    fit_x = fit_x / fit_weight
    fit_y = fit_y / fit_weight
    fit_err = pow(fit_err / fit_weight, 0.5)

    fit_rms = 0.0
    fit_weight = 0.0
    for xing in range(0, len(pair_x)):
        error_sq = pair_err[xing] * pair_err[xing]
        weight = 1.0 / error_sq
        # weight = pair_weight[xing]
        fit_weight += weight
        fit_rms += (
            pow(pair_x[xing] - fit_x, 2) + pow(pair_y[xing] - fit_y, 2)
        ) * weight
    fit_rms = pow(fit_rms / fit_weight, 0.5)

    fit_err = pow(fit_rms * fit_rms + fit_err * fit_err, 0.5)

    # x_mtx, x_mtx_err, chi2 = least_square_fit(2, a, b, w)
    # fit_x = -x_mtx[1]
    # fit_y = -x_mtx[0]
    # fit_x_err = 2.0 * x_mtx_err[1]
    # fit_y_err = 2.0 * x_mtx_err[0]

    return fit_x, fit_y, fit_err


def find_image_moments(
    geometry, input_image_1d, input_time_1d, flip=False, star_cam_xy=None
):
    image_center_x = 0.0
    image_center_y = 0.0
    mask_center_x = 0.0
    mask_center_y = 0.0
    center_time = 0.0
    image_size = 0.0
    mask_size = 0.0
    for pix in range(0, len(input_image_1d)):
        if input_image_1d[pix] == 0.0:
            continue
        mask_size += 1.0
        mask_center_x += float(geometry.pix_x[pix] / u.m)
        mask_center_y += float(geometry.pix_y[pix] / u.m)
        image_size += input_image_1d[pix]
        image_center_x += float(geometry.pix_x[pix] / u.m) * input_image_1d[pix]
        image_center_y += float(geometry.pix_y[pix] / u.m) * input_image_1d[pix]
        center_time += input_time_1d[pix] * input_image_1d[pix]

    if image_size < image_size_cut:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    mask_center_x = mask_center_x / mask_size
    mask_center_y = mask_center_y / mask_size
    image_center_x = image_center_x / image_size
    image_center_y = image_center_y / image_size
    center_time = center_time / image_size

    cov_xx = 0.0
    cov_xy = 0.0
    cov_yx = 0.0
    cov_yy = 0.0
    for pix in range(0, len(input_image_1d)):
        if input_image_1d[pix] == 0.0:
            continue
        diff_x = float(geometry.pix_x[pix] / u.m) - image_center_x
        diff_y = float(geometry.pix_y[pix] / u.m) - image_center_y
        weight = input_image_1d[pix]
        cov_xx += diff_x * diff_x * weight
        cov_xy += diff_x * diff_y * weight
        cov_yx += diff_y * diff_x * weight
        cov_yy += diff_y * diff_y * weight
    cov_xx = cov_xx / image_size
    cov_xy = cov_xy / image_size
    cov_yx = cov_yx / image_size
    cov_yy = cov_yy / image_size

    covariance_matrix = np.array([[cov_xx, cov_xy], [cov_yx, cov_yy]])
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    semi_major_sq = eigenvalues[0]
    semi_minor_sq = eigenvalues[1]
    if semi_minor_sq > semi_major_sq:
        x = semi_minor_sq
        semi_minor_sq = semi_major_sq
        semi_major_sq = x

    truth_a = image_center_y / image_center_x

    a, b, a_err, b_err, chi2 = fit_image_to_line(geometry, input_image_1d)
    aT, bT, aT_err, bT_err, chi2T = fit_image_to_line(
        geometry, input_image_1d, transpose=True
    )
    # if chi2 > chi2T and aT != 0.0:
    #    a = 1.0 / aT
    #    b = -bT / aT
    #    a_err = aT_err
    #    b_err = bT_err

    if aT != 0.0:
        weight = 1.0 / chi2 + 1.0 / chi2T
        a = (a * 1.0 / chi2 + 1.0 / aT * 1.0 / chi2T) / weight
        b = (b * 1.0 / chi2 - bT / aT * 1.0 / chi2T) / weight
        a_err = (a_err * 1.0 / chi2 + aT_err * 1.0 / chi2T) / weight
        b_err = (b_err * 1.0 / chi2 + bT_err * 1.0 / chi2T) / weight

    if a_err == np.inf:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    angle = np.arctan(a)
    angle_err = abs(np.arctan(a + a_err) - np.arctan(a - a_err))

    rotation_matrix = np.array(
        [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]
    )
    diff_x = mask_center_x - image_center_x
    diff_y = mask_center_y - image_center_y
    delta_coord = np.array([diff_x, diff_y])
    rot_coord = rotation_matrix @ delta_coord
    direction_of_image = rot_coord[0] * image_size

    direction_of_time = 0.0
    diff_t_norm = 0.0
    for pix in range(0, len(input_image_1d)):
        if input_image_1d[pix] == 0.0:
            continue
        diff_x = float(geometry.pix_x[pix] / u.m) - image_center_x
        diff_y = float(geometry.pix_y[pix] / u.m) - image_center_y
        diff_t = input_time_1d[pix] - center_time
        delta_coord = np.array([diff_x, diff_y])
        rot_coord = rotation_matrix @ delta_coord
        if rot_coord[0] == 0.0:
            continue
        direction_of_time += rot_coord[0] * diff_t * input_image_1d[pix]
        diff_t_norm += diff_t * diff_t
    if diff_t_norm > 0.0:
        direction_of_time = direction_of_time / pow(diff_t_norm, 0.5)

    if (direction_of_time * direction_of_image) > 0.0:
        if (direction_of_time) > 0.0 and (direction_of_image) > 0.0:
            angle = angle + np.pi
    else:
        if abs(direction_of_image) > abs(direction_of_time):
            if (direction_of_image) > 0.0:
                angle = angle + np.pi
        else:
            if (direction_of_time) > 0.0:
                angle = angle + np.pi

    truth_angle = np.arctan2(-image_center_y, -image_center_x)
    if not star_cam_xy == None:
        truth_angle = np.arctan2(
            star_cam_xy[1] - image_center_y, star_cam_xy[0] - image_center_x
        )

    truth_projection = np.cos(truth_angle - angle)

    if not star_cam_xy == None:
        # angle = truth_angle
        if truth_projection < 0.0:
            angle = angle + np.pi

    if flip:
        angle = angle + np.pi

    return [
        image_size,
        image_center_x,
        image_center_y,
        angle,
        pow(semi_major_sq, 0.5),
        pow(semi_minor_sq, 0.5),
        direction_of_time,
        direction_of_image,
        a,
        b,
        truth_projection,
        a_err,
        b_err,
        angle_err,
    ]


def image_cutout(geometry, image_input_1d, pixs_to_keep=[]):
    eco_image_1d = []
    for pix in pixs_to_keep:
        eco_image_1d += [image_input_1d[pix]]
    return eco_image_1d


def image_cutout_restore(geometry, eco_image_1d, origin_image_1d):
    eco_pix = 0
    for pix in range(0, len(origin_image_1d)):
        x = float(geometry.pix_x[pix] / u.m)
        y = float(geometry.pix_y[pix] / u.m)
        if abs(y) < 0.05:
            origin_image_1d[pix] = eco_image_1d[eco_pix]
            eco_pix += 1
        else:
            origin_image_1d[pix] = 0.0


def find_image_truth(source, subarray, run_id, tel_id, event):
    truth_energy = event.simulation.shower.energy
    truth_core_x = event.simulation.shower.core_x
    truth_core_y = event.simulation.shower.core_y
    truth_alt = event.simulation.shower.alt
    truth_az = event.simulation.shower.az
    truth_height = event.simulation.shower.h_first_int
    truth_x_max = event.simulation.shower.x_max

    geometry = subarray.tel[tel_id].camera.geometry

    obstime = Time("2013-11-01T03:00")
    location = EarthLocation.of_site("Roque de los Muchachos")
    altaz = AltAz(location=location, obstime=obstime)

    tel_pointing_alt = source.observation_blocks[run_id].subarray_pointing_lat
    tel_pointing_az = source.observation_blocks[run_id].subarray_pointing_lon
    tel_pointing = SkyCoord(
        alt=tel_pointing_alt,
        az=tel_pointing_az,
        frame=altaz,
    )
    alt_offset = truth_alt - tel_pointing_alt
    az_offset = truth_az - tel_pointing_az
    star_altaz = SkyCoord(
        alt=tel_pointing_alt + alt_offset,
        az=tel_pointing_az + az_offset,
        frame=altaz,
    )

    focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length
    tel_x = subarray.positions[tel_id][0]
    tel_y = subarray.positions[tel_id][1]
    impact_x = float((truth_core_x - tel_x) / u.m)
    impact_y = float((truth_core_y - tel_y) / u.m)

    camera_frame = CameraFrame(
        telescope_pointing=tel_pointing,
        focal_length=focal_length,
    )

    star_cam = star_altaz.transform_to(camera_frame)
    star_cam_x = star_cam.x.to_value(u.m)
    star_cam_y = star_cam.y.to_value(u.m)

    truth_info_array = [
        truth_energy,
        truth_core_x,
        truth_core_y,
        truth_alt,
        truth_az,
        truth_height,
        truth_x_max,
        star_cam_x,
        star_cam_y,
        impact_x,
        impact_y,
    ]

    return truth_info_array


def make_standard_movie(
    ctapipe_output,
    telescope_type,
    subarray,
    run_id,
    tel_id,
    event,
    flip=False,
    star_cam_xy=None,
):
    event_id = event.index["event_id"]
    geometry = subarray.tel[tel_id].camera.geometry

    clean_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
    clean_time_1d = np.zeros_like(event.dl1.tel[tel_id].peak_time)
    boundary, picture, min_neighbors = cleaning_level[geometry.name]
    image_mask = tailcuts_clean(
        geometry,
        event.dl1.tel[tel_id].image,
        boundary_thresh=boundary,
        picture_thresh=picture,
        min_number_picture_neighbors=min_neighbors,
    )
    # for pix in range(0, len(image_mask)):
    #    if event.dl1.tel[tel_id].image[pix]==0.:
    #        image_mask[pix] = False
    #    else:
    #        image_mask[pix] = True

    for pix in range(0, len(image_mask)):
        if not image_mask[pix]:
            clean_image_1d[pix] = 0.0
            clean_time_1d[pix] = 0.0
        else:
            clean_image_1d[pix] = event.dl1.tel[tel_id].image[pix]
            clean_time_1d[pix] = event.dl1.tel[tel_id].peak_time[pix]

    center_time = reset_time(clean_image_1d, clean_time_1d)

    waveform = event.dl0.tel[tel_id].waveform
    # print(f"waveform.shape = {waveform.shape}")
    n_pix = waveform.shape[1]
    n_samp = waveform.shape[2]

    n_windows = int(total_samples / n_samples_per_window)
    clean_movie_1d = []
    for win in range(0, n_windows):
        clean_movie_1d += [np.zeros_like(clean_image_1d)]
    for pix in range(0, n_pix):
        if not image_mask[pix]:
            continue  # select signal
        for win in range(0, n_windows):
            for sample in range(0, n_samples_per_window):
                sample_idx = int(sample + win * n_samples_per_window)
                if sample_idx < 0:
                    continue
                if sample_idx >= n_samp:
                    continue
                clean_movie_1d[win][pix] += waveform[0, pix, sample_idx]

    # is_edge_image = False
    # image_sum = np.sum(clean_image_1d)
    # mask_sum = np.sum(image_mask)
    ## image_leakage = leakage_parameters(geometry,clean_image_1d,image_mask) # this function has memory problem
    ## if image_leakage.pixels_width_1>0.:
    ##    is_edge_image = True
    ##    for pix in range(0,len(movie_mask)):
    ##        clean_movie_1d[win][pix] = 0.
    # border_pixels = geometry.get_border_pixel_mask(1)
    # border_mask = border_pixels & image_mask
    # leakage_intensity = np.sum(clean_image_1d[border_mask])
    # n_pe_cleaning = np.sum(clean_image_1d)
    # frac_leakage_intensity = leakage_intensity / n_pe_cleaning
    # if frac_leakage_intensity > 0.05:
    #    is_edge_image = True

    is_edge_image = False
    for win in range(0, n_windows):
        image_sum = np.sum(clean_movie_1d[win])
        if image_sum == 0.0:
            continue
        movie_mask = image_mask
        mask_sum = np.sum(movie_mask)
        if mask_sum == 0.0:
            continue
        # image_leakage = leakage_parameters(geometry,clean_movie_1d[win],movie_mask) # this function has memory problem
        # if image_leakage.pixels_width_1>0.:
        #    is_edge_image = True
        #    for pix in range(0,len(movie_mask)):
        #        clean_movie_1d[win][pix] = 0.
        border_pixels = geometry.get_border_pixel_mask(1)
        border_mask = border_pixels & movie_mask
        leakage_intensity = np.sum(clean_movie_1d[win][border_mask])
        n_pe_cleaning = np.sum(clean_movie_1d[win])
        frac_leakage_intensity = leakage_intensity / n_pe_cleaning
        if frac_leakage_intensity > 0.05:
            is_edge_image = True

    # clean_image_2d = geometry.image_to_cartesian_representation(clean_image_1d)
    # remove_nan_pixels(clean_image_2d)
    # clean_time_2d = geometry.image_to_cartesian_representation(clean_time_1d)
    # remove_nan_pixels(clean_time_2d)

    # image_max = np.max(clean_image_2d[:, :])

    pixel_width = float(geometry.pixel_width[0] / u.m)

    image_moment_array = find_image_moments(
        geometry, clean_image_1d, clean_time_1d, flip=flip, star_cam_xy=star_cam_xy
    )
    image_size = image_moment_array[0]
    image_center_x = image_moment_array[1]
    image_center_y = image_moment_array[2]
    angle = image_moment_array[3]
    semi_major = image_moment_array[4]
    semi_minor = image_moment_array[5]
    time_direction = image_moment_array[6]
    image_direction = image_moment_array[7]
    line_a = image_moment_array[8]
    line_b = image_moment_array[9]
    truth_projection = image_moment_array[10]
    line_a_err = image_moment_array[11]
    line_b_err = image_moment_array[12]
    angle_err = image_moment_array[13]
    # print(f"image_size = {image_size:0.1f}")

    if image_size < image_size_cut:
        return is_edge_image, image_moment_array, [], [], []

    center_time_window = 0.0
    total_weight = 0.0
    for win in range(0, n_windows):
        total_weight += np.sum(clean_movie_1d[win][:])
        center_time_window += float(win) * np.sum(clean_movie_1d[win][:])
    if total_weight == 0.0:
        center_time_window = 0
    else:
        center_time_window = round(center_time_window / total_weight)
    # print(f"center_time_window = {center_time_window}")

    n_windows_slim = int(select_samples / n_samples_per_window)
    slim_movie_1d = []
    for win in range(0, n_windows_slim):
        slim_movie_1d += [np.zeros_like(clean_image_1d)]

    for pix in range(0, n_pix):
        for win in range(0, n_windows_slim):
            old_win = int(center_time_window - n_windows_slim / 2 + win)
            if old_win < 0:
                continue
            slim_movie_1d[win][pix] = clean_movie_1d[old_win][pix]

    # image_max = np.max(slim_movie_1d[:][:])

    list_rotat_movie_1d = image_translation_and_rotation(
        geometry,
        slim_movie_1d,
        image_center_x,
        image_center_y,
        (angle + 0.5 * np.pi) * u.rad,
    )

    whole_movie_1d = []
    pixs_to_keep = []
    for pix in range(0, len(clean_image_1d)):
        x = float(geometry.pix_x[pix] / u.m)
        y = float(geometry.pix_y[pix] / u.m)
        if abs(y) < 0.05:
            pixs_to_keep += [pix]
    for win in range(0, n_windows_slim):
        rotate_movie_1d = list_rotat_movie_1d[win]
        eco_movie_1d = image_cutout(
            geometry, rotate_movie_1d, pixs_to_keep=pixs_to_keep
        )
        whole_movie_1d.extend(eco_movie_1d)

    list_rotat_image_1d = image_translation_and_rotation(
        geometry,
        [clean_image_1d, clean_time_1d],
        image_center_x,
        image_center_y,
        (angle + 0.5 * np.pi) * u.rad,
    )
    rotate_image_1d = list_rotat_image_1d[0]
    rotate_time_1d = list_rotat_image_1d[1]
    # rotate_image_2d = geometry.image_to_cartesian_representation(rotate_image_1d)
    # rotate_time_2d = geometry.image_to_cartesian_representation(rotate_time_1d)

    pixs_to_keep = []
    for pix in range(0, len(clean_image_1d)):
        x = float(geometry.pix_x[pix] / u.m)
        y = float(geometry.pix_y[pix] / u.m)
        if abs(y) < 0.05:
            pixs_to_keep += [pix]
    eco_image_1d = image_cutout(geometry, rotate_image_1d, pixs_to_keep=pixs_to_keep)
    eco_time_1d = image_cutout(geometry, rotate_time_1d, pixs_to_keep=pixs_to_keep)

    # if image_size > plot_image_size_cut:
    #    xmax = max(geometry.pix_x) / u.m
    #    xmin = min(geometry.pix_x) / u.m
    #    ymax = max(geometry.pix_y) / u.m
    #    ymin = min(geometry.pix_y) / u.m
    #    font = {
    #        "family": "serif",
    #        "color": "white",
    #        "weight": "normal",
    #        "size": 10,
    #        "rotation": 0.0,
    #    }

    #    fig, ax = plt.subplots()
    #    figsize_x = 8.6
    #    figsize_y = 6.4
    #    fig.set_figheight(figsize_y)
    #    fig.set_figwidth(figsize_x)
    #    label_x = "X"
    #    label_y = "Y"
    #    ax.set_xlabel(label_x)
    #    ax.set_ylabel(label_y)
    #    im = ax.imshow(clean_image_2d, origin="lower", extent=(xmin, xmax, ymin, ymax))
    #    cbar = fig.colorbar(im)
    #    ax.scatter(0.0, 0.0, s=90, facecolors="none", edgecolors="r", marker="o")
    #    if np.cos(angle * u.rad) > 0.0:
    #        line_x = np.linspace(image_center_x, xmax, 100)
    #        line_y = -(line_a * line_x + line_b)
    #        ax.plot(line_x, line_y, color="w", alpha=0.3, linestyle="dashed")
    #        line_y = -(
    #            (line_a + line_a_err) * line_x + line_b - line_a_err * image_center_x
    #        )
    #        ax.plot(line_x, line_y, color="w", alpha=0.3, linestyle="solid")
    #        line_y = -(
    #            (line_a - line_a_err) * line_x + line_b + line_a_err * image_center_x
    #        )
    #        ax.plot(line_x, line_y, color="w", alpha=0.3, linestyle="solid")
    #    else:
    #        line_x = np.linspace(xmin, image_center_x, 100)
    #        line_y = -(line_a * line_x + line_b)
    #        ax.plot(line_x, line_y, color="w", alpha=0.3, linestyle="dashed")
    #        line_y = -(
    #            (line_a + line_a_err) * line_x + line_b - line_a_err * image_center_x
    #        )
    #        ax.plot(line_x, line_y, color="w", alpha=0.3, linestyle="solid")
    #        line_y = -(
    #            (line_a - line_a_err) * line_x + line_b + line_a_err * image_center_x
    #        )
    #        ax.plot(line_x, line_y, color="w", alpha=0.3, linestyle="solid")
    #    ax.set_xlim(xmin, xmax)
    #    ax.set_ylim(ymin, ymax)
    #    # txt = ax.text(-0.35, 0.35, 'image size = %0.2e'%(image_size), fontdict=font)
    #    # txt = ax.text(-0.35, 0.32, 'image direction = %0.2e'%(image_direction), fontdict=font)
    #    # txt = ax.text(-0.35, 0.29, 'time direction = %0.2e'%(time_direction), fontdict=font)
    #    fig.savefig(
    #        f"{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_clean_image.png",
    #        bbox_inches="tight",
    #    )
    #    del fig
    #    del ax
    #    plt.close()

    return is_edge_image, image_moment_array, whole_movie_1d, eco_image_1d, eco_time_1d


def analyze_a_training_image(
    ctapipe_output, telescope_type, event, source, run_id, tel_id
):
    truth_info_array = find_image_truth(source, source.subarray, run_id, tel_id, event)
    star_cam_x = truth_info_array[7]
    star_cam_y = truth_info_array[8]
    star_cam_xy = [star_cam_x, star_cam_y]

    (
        is_edge_image,
        image_moment_array,
        eco_movie_1d,
        eco_image_1d,
        eco_time_1d,
    ) = make_standard_movie(
        ctapipe_output,
        telescope_type,
        source.subarray,
        run_id,
        tel_id,
        event,
        flip=False,
        star_cam_xy=star_cam_xy,
    )

    image_size = image_moment_array[0]
    # print(f"image_size = {image_size:0.3f}")

    if image_size < image_size_cut:
        print("failed image_size_cut.")
        return None
    if is_edge_image:
        print("failed: edge image.")
        return None

    return [
        eco_movie_1d,
        eco_image_1d,
        eco_time_1d,
        image_moment_array,
        truth_info_array,
    ]


def analyze_a_training_event(
    ctapipe_output,
    telescope_type,
    event,
    source,
    run_id,
    movie_matrix,
    image_matrix,
    time_matrix,
    moment_matrix,
    truth_matrix,
):
    event_id = event.index["event_id"]

    ntel = len(event.r0.tel)

    for tel_idx in range(0, len(list(event.dl0.tel.keys()))):
        tel_id = list(event.dl0.tel.keys())[tel_idx]
        # if event_id!=27002: continue
        # if tel_id!=31: continue

        if str(telescope_type) != str(source.subarray.tel[tel_id]):
            continue

        print(
            "===================================================================================="
        )
        print(f"Select telescope type: {telescope_type}")
        print(f"event_id = {event_id}, tel_id = {tel_id}")
        print("TEL{:03}: {}".format(tel_id, source.subarray.tel[tel_id]))

        analysis_results = analyze_a_training_image(
            ctapipe_output, telescope_type, event, source, run_id, tel_id
        )

        if analysis_results == None:
            continue

        eco_movie_1d = analysis_results[0]
        eco_image_1d = analysis_results[1]
        eco_time_1d = analysis_results[2]
        image_moment_array = analysis_results[3]
        truth_info_array = analysis_results[4]

        movie_matrix += [eco_movie_1d]
        image_matrix += [eco_image_1d]
        time_matrix += [eco_time_1d]
        moment_matrix += [image_moment_array]
        truth_matrix += [truth_info_array]


def run_save_training_matrix(training_sample_path, telescope_type, ctapipe_output):
    big_movie_matrix = []
    big_image_matrix = []
    big_time_matrix = []
    big_moment_matrix = []
    big_truth_matrix = []

    print(f"loading file: {training_sample_path}")
    source = SimTelEventSource(training_sample_path, focal_length_choice="EQUIVALENT")

    # Explore the instrument description
    subarray = source.subarray
    print("Array info:")
    print(subarray.info())
    print(subarray.to_table())

    ob_keys = source.observation_blocks.keys()
    run_id = list(ob_keys)[0]

    tel_pointing_alt = float(
        source.observation_blocks[run_id].subarray_pointing_lat / u.rad
    )
    tel_pointing_az = float(
        source.observation_blocks[run_id].subarray_pointing_lon / u.rad
    )
    print(f"tel_pointing_alt = {tel_pointing_alt}")
    print(f"tel_pointing_az = {tel_pointing_az}")

    calib = CameraCalibrator(subarray=subarray)
    image_processor = ImageProcessor(subarray=subarray)

    for event in source:
        calib(event)  # fills in r1, dl0, and dl1
        image_processor(event)
        analyze_a_training_event(
            ctapipe_output,
            telescope_type,
            event,
            source,
            run_id,
            big_movie_matrix,
            big_image_matrix,
            big_time_matrix,
            big_moment_matrix,
            big_truth_matrix,
        )

    ana_tag = "training_sample"
    output_filename = (
        f"{ctapipe_output}/output_samples/{ana_tag}_run{run_id}_{telescope_type}.pkl"
    )
    print(f"writing file to {output_filename}")
    with open(output_filename, "wb") as file:
        pickle.dump(
            [
                big_truth_matrix,
                big_moment_matrix,
                big_image_matrix,
                big_time_matrix,
                big_movie_matrix,
            ],
            file,
        )

    print(f"total images saved = {len(big_image_matrix)}")

    return


class veritas_histogram_3D:
    def __init__(
        self,
        x_bins=10,
        start_x=0.0,
        end_x=10.0,
        y_bins=10,
        start_y=0.0,
        end_y=10.0,
        z_bins=10,
        start_z=0.0,
        end_z=10.0,
        overflow=True,
    ):
        array_shape = (x_bins, y_bins, z_bins)
        self.delta_x = (end_x - start_x) / float(x_bins)
        self.delta_y = (end_y - start_y) / float(y_bins)
        self.delta_z = (end_z - start_z) / float(z_bins)
        self.xaxis = np.zeros(array_shape[0] + 1)
        self.yaxis = np.zeros(array_shape[1] + 1)
        self.zaxis = np.zeros(array_shape[2] + 1)
        self.waxis = np.zeros(array_shape)
        self.overflow = overflow
        for idx in range(0, len(self.xaxis)):
            self.xaxis[idx] = start_x + idx * self.delta_x
        for idx in range(0, len(self.yaxis)):
            self.yaxis[idx] = start_y + idx * self.delta_y
        for idx in range(0, len(self.zaxis)):
            self.zaxis[idx] = start_z + idx * self.delta_z

    def reset(self):
        for idx_x in range(0, len(self.xaxis) - 1):
            for idx_y in range(0, len(self.yaxis) - 1):
                for idx_z in range(0, len(self.zaxis) - 1):
                    self.waxis[idx_x, idx_y, idx_z] = 0.0

    def add(self, add_array, factor=1.0):
        for idx_x in range(0, len(self.xaxis) - 1):
            for idx_y in range(0, len(self.yaxis) - 1):
                for idx_z in range(0, len(self.zaxis) - 1):
                    self.waxis[idx_x, idx_y, idx_z] = (
                        self.waxis[idx_x, idx_y, idx_z]
                        + add_array.waxis[idx_x, idx_y, idx_z] * factor
                    )

    def get_bin(self, value_x, value_y, value_z):
        key_idx_x = -1
        key_idx_y = -1
        key_idx_z = -1
        for idx_x in range(0, len(self.xaxis) - 1):
            if self.xaxis[idx_x] <= value_x and self.xaxis[idx_x + 1] > value_x:
                key_idx_x = idx_x
        for idx_y in range(0, len(self.yaxis) - 1):
            if self.yaxis[idx_y] <= value_y and self.yaxis[idx_y + 1] > value_y:
                key_idx_y = idx_y
        for idx_z in range(0, len(self.zaxis) - 1):
            if self.zaxis[idx_z] <= value_z and self.zaxis[idx_z + 1] > value_z:
                key_idx_z = idx_z
        if value_x > self.xaxis.max():
            key_idx_x = len(self.xaxis) - 2
        if value_y > self.yaxis.max():
            key_idx_y = len(self.yaxis) - 2
        if value_z > self.zaxis.max():
            key_idx_z = len(self.zaxis) - 2
        return [key_idx_x, key_idx_y, key_idx_z]

    def get_heaviest_axis(self):
        max_weight = 0.0
        key_idx_x = -1
        key_idx_y = -1
        key_idx_z = -1
        for idx_x in range(0, len(self.xaxis) - 1):
            for idx_y in range(0, len(self.yaxis) - 1):
                for idx_z in range(0, len(self.zaxis) - 1):
                    local_weight = abs(self.waxis[idx_x, idx_y, idx_z])
                    if max_weight < local_weight:
                        max_weight = local_weight
                        key_idx_x = idx_x
                        key_idx_y = idx_y
                        key_idx_z = idx_z
        return [self.xaxis[key_idx_x], self.yaxis[key_idx_y], self.zaxis[key_idx_z]]

    def fill(self, value_x, value_y, value_z, weight=1.0):
        key_idx = self.get_bin(value_x, value_y, value_z)
        key_idx_x = key_idx[0]
        key_idx_y = key_idx[1]
        key_idx_z = key_idx[2]
        if key_idx_x == -1:
            key_idx_x = 0
            if not self.overflow:
                weight = 0.0
        if key_idx_y == -1:
            key_idx_y = 0
            if not self.overflow:
                weight = 0.0
        if key_idx_z == -1:
            key_idx_z = 0
            if not self.overflow:
                weight = 0.0
        if key_idx_x == len(self.xaxis):
            key_idx_x = len(self.xaxis) - 2
            if not self.overflow:
                weight = 0.0
        if key_idx_y == len(self.yaxis):
            key_idx_y = len(self.yaxis) - 2
            if not self.overflow:
                weight = 0.0
        if key_idx_z == len(self.zaxis):
            key_idx_z = len(self.zaxis) - 2
            if not self.overflow:
                weight = 0.0
        self.waxis[key_idx_x, key_idx_y, key_idx_z] += 1.0 * weight

    def divide(self, add_array):
        for idx_x in range(0, len(self.xaxis) - 1):
            for idx_y in range(0, len(self.yaxis) - 1):
                for idx_z in range(0, len(self.zaxis) - 1):
                    if add_array.waxis[idx_x, idx_y, idx_z] == 0.0:
                        self.waxis[idx_x, idx_y, idx_z] = 0.0
                    else:
                        self.waxis[idx_x, idx_y, idx_z] = (
                            self.waxis[idx_x, idx_y, idx_z]
                            / add_array.waxis[idx_x, idx_y, idx_z]
                        )

    def get_bin_center(self, idx_x, idx_y, idx_z):
        return [
            self.xaxis[idx_x] + 0.5 * self.delta_x,
            self.yaxis[idx_y] + 0.5 * self.delta_y,
            self.zaxis[idx_z] + 0.5 * self.delta_z,
        ]

    def get_bin_content(self, value_x, value_y, value_z):
        key_idx = self.get_bin(value_x, value_y, value_z)
        key_idx_x = key_idx[0]
        key_idx_y = key_idx[1]
        key_idx_z = key_idx[2]
        if key_idx_x == -1:
            key_idx_x = 0
        if key_idx_y == -1:
            key_idx_y = 0
        if key_idx_z == -1:
            key_idx_z = 0
        if key_idx_x == len(self.xaxis):
            key_idx_x = len(self.xaxis) - 2
        if key_idx_y == len(self.yaxis):
            key_idx_y = len(self.yaxis) - 2
        if key_idx_z == len(self.zaxis):
            key_idx_z = len(self.zaxis) - 2
        return self.waxis[key_idx_x, key_idx_y, key_idx_z]


def MakeLookupTable(
    ctapipe_output,
    telescope_type,
    eigenvectors,
    big_matrix,
    moment_matrix,
    truth_matrix,
    image_rank,
    pkl_name,
    nvar=3,
):
    lookup_table = []
    lookup_table_norm = veritas_histogram_3D(
        x_bins=n_bins_arrival,
        start_x=arrival_lower,
        end_x=arrival_upper,
        y_bins=n_bins_impact,
        start_y=impact_lower,
        end_y=impact_upper,
        z_bins=n_bins_energy,
        start_z=log_energy_lower,
        end_z=log_energy_upper,
    )

    list_impact = []
    list_arrival = []
    list_log_energy = []
    list_height = []
    list_xmax = []
    list_image_qual = []

    for r in range(0, image_rank):
        lookup_table += [
            veritas_histogram_3D(
                x_bins=n_bins_arrival,
                start_x=arrival_lower,
                end_x=arrival_upper,
                y_bins=n_bins_impact,
                start_y=impact_lower,
                end_y=impact_upper,
                z_bins=n_bins_energy,
                start_z=log_energy_lower,
                end_z=log_energy_upper,
            )
        ]

    for img in range(0, len(big_matrix)):
        image_center_x = moment_matrix[img][1]
        image_center_y = moment_matrix[img][2]
        time_direction = moment_matrix[img][6]
        image_direction = moment_matrix[img][7]
        image_angle_err = moment_matrix[img][13]
        image_qual = abs(image_direction + time_direction)

        truth_energy = float(truth_matrix[img][0] / u.TeV)
        truth_height = float(truth_matrix[img][5] / u.m)
        truth_x_max = float(truth_matrix[img][6] / (u.g / (u.cm * u.cm)))
        star_cam_x = truth_matrix[img][7]
        star_cam_y = truth_matrix[img][8]
        impact_x = truth_matrix[img][9]
        impact_y = truth_matrix[img][10]

        arrival = pow(
            pow(star_cam_x - image_center_x, 2) + pow(star_cam_y - image_center_y, 2),
            0.5,
        )
        impact = pow(impact_x * impact_x + impact_y * impact_y, 0.5)
        log_energy = np.log10(truth_energy)

        list_log_energy += [log_energy]
        list_height += [truth_height]
        list_xmax += [truth_x_max]
        list_arrival += [arrival]
        list_impact += [impact]

        image_1d = np.array(big_matrix[img])
        image_latent_space = eigenvectors @ image_1d
        for r in range(0, image_rank):
            lookup_table[r].fill(
                arrival,
                impact,
                log_energy,
                weight=image_latent_space[r] * 1.0 / image_angle_err,
            )

        lookup_table_norm.fill(
            arrival, impact, log_energy, weight=1.0 / image_angle_err
        )

    for r in range(0, image_rank):
        lookup_table[r].divide(lookup_table_norm)

    n_empty_cells = 0.0
    n_filled_cells = 0.0
    n_training_images = 0.0
    for idx_x in range(0, len(lookup_table_norm.xaxis) - 1):
        for idx_y in range(0, len(lookup_table_norm.yaxis) - 1):
            for idx_z in range(0, len(lookup_table_norm.zaxis) - 1):
                count = lookup_table_norm.waxis[idx_x, idx_y, idx_z]
                if count == 0:
                    n_empty_cells += 1.0
                else:
                    n_filled_cells += 1.0
                n_training_images += count
    avg_images_per_cell = n_training_images / n_filled_cells
    print(
        f"n_empty_cells = {n_empty_cells}, n_filled_cells = {n_filled_cells}, n_training_images = {n_training_images}, avg_images_per_cell = {avg_images_per_cell:0.1f}"
    )

    output_filename = (
        f"{ctapipe_output}/output_machines/{pkl_name}_lookup_table_{telescope_type}.pkl"
    )
    with open(output_filename, "wb") as file:
        pickle.dump(lookup_table, file)


def BigMatrixSVD(
    ctapipe_output,
    telescope_type,
    big_matrix,
    moment_matrix,
    truth_matrix,
    image_rank,
    pkl_name,
):
    big_matrix = np.array(big_matrix)

    n_images, n_pixels = big_matrix.shape
    print(f"n_images = {n_images}, n_pixels = {n_pixels}")

    U_full, S_full, VT_full = np.linalg.svd(big_matrix, full_matrices=False)
    U_eco = U_full[:, :image_rank]
    VT_eco = VT_full[:image_rank, :]

    print(f"saving image eigenvector to {ctapipe_output}/output_machines...")
    output_filename = f"{ctapipe_output}/output_machines/{pkl_name}_eigen_vectors_{telescope_type}.pkl"
    with open(output_filename, "wb") as file:
        pickle.dump(VT_eco, file)

    fig, ax = plt.subplots()
    figsize_x = 8.6
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "Rank"
    label_y = "Signular value"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_xscale("log")
    ax.plot(S_full)
    fig.savefig(
        f"{ctapipe_output}/output_plots/training_{pkl_name}_signularvalue_{telescope_type}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()

    MakeLookupTable(
        ctapipe_output,
        telescope_type,
        VT_eco,
        big_matrix,
        moment_matrix,
        truth_matrix,
        image_rank,
        pkl_name + "_box3d",
        nvar=3,
    )

    return VT_eco


def linear_regression(input_data, target_data, weight):
    # solve x*A = y using SVD
    # y_{0} = ( x_{0,0} x_{0,1} ... 1 )  a_{0}
    # y_{1} = ( x_{1,0} x_{1,1} ... 1 )  a_{1}
    # y_{2} = ( x_{2,0} x_{2,1} ... 1 )  .
    #                                    b

    x = []
    y = []
    w = []
    for evt in range(0, len(input_data)):
        single_x = []
        for entry in range(0, len(input_data[evt])):
            single_x += [input_data[evt][entry]]
            single_x += [input_data[evt][entry] ** 2]
            single_x += [input_data[evt][entry] ** 3]
            single_x += [input_data[evt][entry] ** 4]
        single_x += [1.0]
        x += [single_x]
        y += [target_data[evt]]
        w += [weight[evt]]
    x = np.array(x)
    y = np.array(y)
    w = np.diag(w)

    # Compute the weighted SVD
    U, S, Vt = np.linalg.svd(w @ x, full_matrices=False)
    # Calculate the weighted pseudo-inverse
    S_pseudo_w = np.diag(1 / S)
    x_pseudo_w = Vt.T @ S_pseudo_w @ U.T
    # Compute the weighted least-squares solution
    A_svd = x_pseudo_w @ (w @ y)
    # Compute chi2
    chi2 = np.linalg.norm((w @ x).dot(A_svd) - (w @ y), 2) / np.trace(w)

    return A_svd, chi2


def linear_model(input_data, A):
    x = []
    for entry in range(0, len(input_data)):
        x += [input_data[entry]]
        x += [input_data[entry] ** 2]
        x += [input_data[entry] ** 3]
        x += [input_data[entry] ** 4]
    x += [1.0]
    x = np.array(x)

    y = x @ A

    return y


def MakeFastConversionImage(
    ctapipe_output,
    telescope_type,
    image_eigenvectors,
    big_image_matrix,
    time_eigenvectors,
    big_time_matrix,
    moment_matrix,
    truth_matrix,
    pkl_name,
):
    list_image_size = []
    list_evt_weight = []
    list_arrival = []
    list_impact = []
    list_log_energy = []
    list_latent_space = []

    for img in range(0, len(big_image_matrix)):
        image_size = moment_matrix[img][0]
        image_center_x = moment_matrix[img][1]
        image_center_y = moment_matrix[img][2]
        time_direction = moment_matrix[img][6]
        image_direction = moment_matrix[img][7]
        image_angle_err = moment_matrix[img][13]
        image_qual = abs(image_direction + time_direction)

        truth_energy = float(truth_matrix[img][0] / u.TeV)
        truth_height = float(truth_matrix[img][5] / u.m)
        truth_x_max = float(truth_matrix[img][6] / (u.g / (u.cm * u.cm)))
        star_cam_x = truth_matrix[img][7]
        star_cam_y = truth_matrix[img][8]
        impact_x = truth_matrix[img][9]
        impact_y = truth_matrix[img][10]

        arrival = pow(
            pow(star_cam_x - image_center_x, 2) + pow(star_cam_y - image_center_y, 2),
            0.5,
        )
        impact = pow(impact_x * impact_x + impact_y * impact_y, 0.5)
        log_energy = np.log10(truth_energy)

        image_1d = np.array(big_image_matrix[img])
        image_latent_space = image_eigenvectors @ image_1d
        time_1d = np.array(big_time_matrix[img])
        time_latent_space = time_eigenvectors @ time_1d
        list_latent_space += [np.concatenate((image_latent_space, time_latent_space))]

        list_image_size += [image_size]
        list_evt_weight += [1.0 / image_angle_err]
        list_arrival += [arrival]
        list_impact += [impact]
        list_log_energy += [log_energy]

    target = list_arrival
    model, chi = linear_regression(list_latent_space, target, list_evt_weight)

    output_filename = f"{ctapipe_output}/output_machines/{pkl_name}_fast_conversion_arrival_{telescope_type}.pkl"
    with open(output_filename, "wb") as file:
        pickle.dump(model, file)

    target = list_impact
    model, chi = linear_regression(list_latent_space, target, list_evt_weight)

    output_filename = f"{ctapipe_output}/output_machines/{pkl_name}_fast_conversion_impact_{telescope_type}.pkl"
    with open(output_filename, "wb") as file:
        pickle.dump(model, file)

    target = list_log_energy
    model, chi = linear_regression(list_latent_space, target, list_evt_weight)

    output_filename = f"{ctapipe_output}/output_machines/{pkl_name}_fast_conversion_log_energy_{telescope_type}.pkl"
    with open(output_filename, "wb") as file:
        pickle.dump(model, file)


def sqaure_difference_between_1d_images(
    init_params, data_latent_space, lookup_table, eigen_vectors, full_table=False
):
    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]
    if not full_table:
        if (
            lookup_table[0].get_bin_content(fit_arrival, fit_impact, fit_log_energy)
            == 0.0
        ):
            return 1e10

    fit_latent_space = []
    for r in range(0, len(lookup_table)):
        fit_latent_space += [
            lookup_table[r].get_bin_content(fit_arrival, fit_impact, fit_log_energy)
        ]
    fit_latent_space = np.array(fit_latent_space)

    sum_chi2 = 0.0
    n_rows = len(data_latent_space)
    for row in range(0, n_rows):
        if data_latent_space[row] == 0.0 and fit_latent_space[row] == 0.0:
            continue
        diff = data_latent_space[row] - fit_latent_space[row]
        sum_chi2 += diff * diff

    return sum_chi2


def sortFirst(val):
    return val[0]


def box_search(
    init_params,
    input_image_1d,
    image_lookup_table,
    image_eigen_vectors,
    input_time_1d,
    time_lookup_table,
    time_eigen_vectors,
    input_movie_1d,
    movie_lookup_table,
    movie_eigen_vectors,
    arrival_range,
    impact_range,
    log_energy_range,
):
    movie_latent_space = movie_eigen_vectors @ input_movie_1d
    image_latent_space = image_eigen_vectors @ input_image_1d
    time_latent_space = time_eigen_vectors @ input_time_1d

    image_norm = np.sum(np.abs(image_latent_space))
    time_norm = np.sum(np.abs(time_latent_space))
    movie_norm = np.sum(np.abs(movie_latent_space))

    init_arrival = init_params[0]
    init_impact = init_params[1]
    init_log_energy = init_params[2]
    short_list = []

    while len(short_list) == 0:
        fit_idx_x = 0
        fit_idx_y = 0
        fit_idx_z = 0
        for idx_x in range(0, n_bins_arrival):
            try_arrival = image_lookup_table[0].xaxis[idx_x]
            if abs(init_arrival - try_arrival) > arrival_range:
                continue
            for idx_y in range(0, n_bins_impact):
                try_impact = image_lookup_table[0].yaxis[idx_y]
                if abs(init_impact - try_impact) > impact_range:
                    continue
                for idx_z in range(0, n_bins_energy):
                    try_log_energy = image_lookup_table[0].zaxis[idx_z]
                    if abs(init_log_energy - try_log_energy) > log_energy_range:
                        continue

                    try_params = [try_arrival, try_impact, try_log_energy]

                    try_chi2 = 0.0
                    try_chi2_image = (
                        sqaure_difference_between_1d_images(
                            try_params,
                            image_latent_space,
                            image_lookup_table,
                            image_eigen_vectors,
                        )
                        / image_norm
                    )
                    try_chi2 += try_chi2_image
                    try_chi2_time = (
                        sqaure_difference_between_1d_images(
                            try_params,
                            time_latent_space,
                            time_lookup_table,
                            time_eigen_vectors,
                        )
                        / time_norm
                    )
                    try_chi2 += try_chi2_time
                    # try_chi2_movie = (
                    #    sqaure_difference_between_1d_images(
                    #        try_params,
                    #        movie_latent_space,
                    #        movie_lookup_table,
                    #        movie_eigen_vectors,
                    #    )
                    #    / movie_norm
                    # )
                    # try_chi2 += try_chi2_movie

                    short_list += [(try_chi2, try_arrival, try_impact, try_log_energy)]

        if len(short_list) == 0:
            print("short_list is zero. expand search range.")
            arrival_range = 1e10
            impact_range = 1e10
            log_energy_range = 1e10
        else:
            break

    short_list.sort(key=sortFirst)
    fit_chi2 = short_list[0][0]
    fit_arrival = short_list[0][1]
    fit_impact = short_list[0][2]
    fit_log_energy = short_list[0][3]

    err_arrival = 1e10
    err_impact = 1e10
    err_log_energy = 1e10
    if len(short_list) > 1:
        err_arrival = 0.0
        err_impact = 0.0
        err_log_energy = 0.0
        sum_likelihood = 0.0
        for entry in range(0, min(5, len(short_list))):
            chi2 = short_list[entry][0]
            arrival = short_list[entry][1]
            impact = short_list[entry][2]
            log_energy = short_list[entry][3]
            norm_likelihood = 1.0 / chi2
            err_arrival += pow(arrival - fit_arrival, 2) * norm_likelihood
            err_impact += pow(impact - fit_impact, 2) * norm_likelihood
            err_log_energy += pow(log_energy - fit_log_energy, 2) * norm_likelihood
            sum_likelihood += norm_likelihood
        err_arrival = 0.5 * pow(err_arrival / sum_likelihood, 0.5)
        err_impact = 0.5 * pow(err_impact / sum_likelihood, 0.5)
        err_log_energy = 0.5 * pow(err_log_energy / sum_likelihood, 0.5)

    return (
        short_list,
        fit_arrival,
        fit_impact,
        fit_log_energy,
        err_arrival,
        err_impact,
        err_log_energy,
        fit_chi2,
    )


def sqaure_difference_between_1d_images_poisson(
    init_params, image_1d_data, lookup_table, eigen_vectors, full_table=False
):
    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]
    if not full_table:
        if (
            lookup_table[0].get_bin_content(fit_arrival, fit_impact, fit_log_energy)
            == 0.0
        ):
            return 1e10

    fit_latent_space = []
    for r in range(0, len(lookup_table)):
        fit_latent_space += [
            lookup_table[r].get_bin_content(fit_arrival, fit_impact, fit_log_energy)
        ]
    fit_latent_space = np.array(fit_latent_space)

    data_latent_space = eigen_vectors @ image_1d_data

    sum_log_likelihood = 0.0
    image_1d_fit = eigen_vectors.T @ fit_latent_space
    n_rows = len(image_1d_fit)
    for row in range(0, n_rows):
        n_expect = max(0.0001, image_1d_fit[row])
        n_data = max(0.0, image_1d_data[row])
        if n_data == 0.0:
            sum_log_likelihood += n_expect
        else:
            sum_log_likelihood += -1.0 * (
                n_data * np.log(n_expect)
                - n_expect
                - (n_data * np.log(n_data) - n_data)
            )

    return sum_log_likelihood


def analyze_short_list(
    short_list, init_params, input_movie_1d, movie_lookup_table, movie_eigen_vectors
):
    # print(f"len(short_list) = {len(short_list)}")
    fit_chi2 = 1e10
    fit_likelihood = 0.0
    n_short_list = 5
    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]
    list_chi2 = []
    list_likelihood = []
    list_arrival = []
    list_impact = []
    list_log_energy = []
    movie_latent_space = movie_eigen_vectors @ input_movie_1d
    for entry in range(0, min(n_short_list, len(short_list))):
        try_arrival = short_list[entry][1]
        try_impact = short_list[entry][2]
        try_log_energy = short_list[entry][3]
        init_params = [try_arrival, try_impact, try_log_energy]
        try_chi2 = sqaure_difference_between_1d_images_poisson(
            init_params, input_movie_1d, movie_lookup_table, movie_eigen_vectors
        )
        # try_chi2 = (
        #    sqaure_difference_between_1d_images(
        #        init_params,
        #        movie_latent_space,
        #        movie_lookup_table,
        #        movie_eigen_vectors,
        #    )
        # )
        try_likelihood = np.exp(-try_chi2)
        list_chi2 += [try_chi2]
        list_likelihood += [try_likelihood]
        list_arrival += [try_arrival]
        list_impact += [try_impact]
        list_log_energy += [try_log_energy]
        if try_chi2 < fit_chi2:
            fit_chi2 = try_chi2
            fit_likelihood = try_likelihood
            fit_arrival = try_arrival
            fit_impact = try_impact
            fit_log_energy = try_log_energy

    err_arrival = 1e10
    err_impact = 1e10
    err_log_energy = 1e10
    if len(list_likelihood) > 1:
        err_arrival = 0.0
        err_impact = 0.0
        err_log_energy = 0.0
        sum_likelihood = 0.0
        for entry in range(0, len(list_likelihood)):
            norm_likelihood = 1.0 / (list_chi2[entry])
            err_arrival += pow(list_arrival[entry] - fit_arrival, 2) * norm_likelihood
            err_impact += pow(list_impact[entry] - fit_impact, 2) * norm_likelihood
            err_log_energy += (
                pow(list_log_energy[entry] - fit_log_energy, 2) * norm_likelihood
            )
            sum_likelihood += norm_likelihood
        err_arrival = pow(err_arrival / sum_likelihood, 0.5)
        err_impact = pow(err_impact / sum_likelihood, 0.5)
        err_log_energy = pow(err_log_energy / sum_likelihood, 0.5)

    return (
        fit_arrival,
        fit_impact,
        fit_log_energy,
        err_arrival,
        err_impact,
        err_log_energy,
        fit_chi2,
    )


def single_movie_reconstruction(
    input_image_1d,
    image_lookup_table,
    image_eigen_vectors,
    input_time_1d,
    time_lookup_table,
    time_eigen_vectors,
    input_movie_1d,
    movie_lookup_table,
    movie_eigen_vectors,
    fast_conversion_poly,
):
    arrival_step_size = (arrival_upper - arrival_lower) / float(n_bins_arrival)
    impact_step_size = (impact_upper - impact_lower) / float(n_bins_impact)
    log_energy_step_size = (log_energy_upper - log_energy_lower) / float(n_bins_energy)

    movie_latent_space = movie_eigen_vectors @ input_movie_1d
    image_latent_space = image_eigen_vectors @ input_image_1d
    time_latent_space = time_eigen_vectors @ input_time_1d
    combine_latent_space = np.concatenate((image_latent_space, time_latent_space))

    image_size = np.sum(input_movie_1d)
    fit_arrival = linear_model(combine_latent_space, fast_conversion_poly[0])
    fit_impact = linear_model(combine_latent_space, fast_conversion_poly[1])
    fit_log_energy = linear_model(combine_latent_space, fast_conversion_poly[2])
    # print(
    #    f"initial fit_arrival = {fit_arrival}, fit_impact = {fit_impact}, fit_log_energy = {fit_log_energy}"
    # )

    init_params = [fit_arrival, fit_impact, fit_log_energy]
    fit_chi2 = 0.0

    found_minimum = False
    fit_chi2 = 1e10

    param_search_range = 1e5
    arrival_range = param_search_range * arrival_step_size
    impact_range = param_search_range * impact_step_size
    log_energy_range = param_search_range * log_energy_step_size

    init_params = [fit_arrival, fit_impact, fit_log_energy]
    (
        short_list,
        fit_arrival,
        fit_impact,
        fit_log_energy,
        err_arrival,
        err_impact,
        err_log_energy,
        fit_chi2,
    ) = box_search(
        init_params,
        input_image_1d,
        image_lookup_table,
        image_eigen_vectors,
        input_time_1d,
        time_lookup_table,
        time_eigen_vectors,
        input_movie_1d,
        movie_lookup_table,
        movie_eigen_vectors,
        arrival_range,
        impact_range,
        log_energy_range,
    )

    # init_params = [fit_arrival, fit_impact, fit_log_energy]
    # poisson_chi2 = sqaure_difference_between_1d_images_poisson(
    #    init_params, input_movie_1d, movie_lookup_table, movie_eigen_vectors
    # )
    # to_be_optimized_factor = 0.5
    # normalized_chi2 = to_be_optimized_factor*(poisson_chi2/image_size)
    # err_arrival = pow(pow(err_arrival,2)+pow(fit_arrival*normalized_chi2,2),0.5)
    # err_impact = pow(pow(err_impact,2)+pow(fit_impact*normalized_chi2,2),0.5)
    # err_log_energy = pow(pow(err_log_energy,2)+pow(fit_log_energy*normalized_chi2,2),0.5)

    init_params = [fit_arrival, fit_impact, fit_log_energy]
    (
        fit_arrival,
        fit_impact,
        fit_log_energy,
        err_arrival,
        err_impact,
        err_log_energy,
        poisson_chi2,
    ) = analyze_short_list(
        short_list,
        init_params,
        input_movie_1d,
        movie_lookup_table,
        movie_eigen_vectors,
    )

    # param_search_range = 3.0
    # arrival_range = param_search_range * arrival_step_size
    # impact_range = param_search_range * impact_step_size
    # log_energy_range = param_search_range * log_energy_step_size

    # while not found_minimum:

    #    init_params = [fit_arrival,fit_impact,fit_log_energy]
    #    short_list = box_search(init_params,image_latent_space,image_lookup_table,image_eigen_vectors,time_latent_space,time_lookup_table,time_eigen_vectors,arrival_range,impact_range,log_energy_range)
    #    try_chi2 = short_list[0][0]
    #    try_arrival = short_list[0][1]
    #    try_impact = short_list[0][2]
    #    try_log_energy = short_list[0][3]

    #    if fit_chi2>try_chi2:
    #        fit_chi2 = try_chi2
    #        fit_arrival = try_arrival
    #        fit_impact = try_impact
    #        fit_log_energy = try_log_energy
    #        best_short_list = short_list
    #    else:
    #        found_minimum = True

    # init_params = [fit_arrival,fit_impact,fit_log_energy]
    # fit_arrival, fit_impact, fit_log_energy, err_arrival, err_impact, err_log_energy, fit_chi2 = analyze_short_list(best_short_list,init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors)

    # normalized_chi2 = fit_chi2/image_size

    # chi2_cut = 1.5
    # is_good_result = True
    # if normalized_chi2>chi2_cut:
    #    print ('chi2 is bad.')
    #    is_good_result = False

    # if not is_good_result:

    #    param_search_range = 100.0
    #    arrival_range = param_search_range*arrival_step_size
    #    impact_range = param_search_range*impact_step_size
    #    log_energy_range = param_search_range*log_energy_step_size

    #    found_minimum = False
    #    fit_chi2 = 1e10
    #    fit_arrival = 0.2
    #    fit_impact = 300.
    #    fit_log_energy = 0.0
    #    best_short_list = []

    #    init_params = [fit_arrival,fit_impact,fit_log_energy]
    #    best_short_list = box_search(init_params,image_latent_space,image_lookup_table,image_eigen_vectors,time_latent_space,time_lookup_table,time_eigen_vectors,arrival_range,impact_range,log_energy_range)
    #    fit_arrival, fit_impact, fit_log_energy, err_arrival, err_impact, err_log_energy, fit_chi2 = analyze_short_list(best_short_list,init_params,input_movie_1d,movie_lookup_table,movie_eigen_vectors)

    return (
        fit_arrival + 0.5 * arrival_step_size,
        fit_impact + 0.5 * impact_step_size,
        fit_log_energy + 0.5 * log_energy_step_size,
        err_arrival,
        err_impact,
        err_log_energy,
        poisson_chi2,
    )

    # return (
    #    fit_arrival,
    #    fit_impact,
    #    fit_log_energy,
    #    err_arrival,
    #    err_impact,
    #    err_log_energy,
    #    fit_chi2,
    # )


def camxy_to_altaz(source, subarray, run_id, tel_id, star_cam_x, star_cam_y):
    obstime = Time("2013-11-01T03:00")
    location = EarthLocation.of_site("Roque de los Muchachos")
    altaz = AltAz(location=location, obstime=obstime)

    tel_pointing_alt = source.observation_blocks[run_id].subarray_pointing_lat
    tel_pointing_az = source.observation_blocks[run_id].subarray_pointing_lon

    focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length

    tel_pointing = SkyCoord(
        alt=tel_pointing_alt,
        az=tel_pointing_az,
        frame=altaz,
    )

    camera_frame = CameraFrame(
        telescope_pointing=tel_pointing,
        focal_length=focal_length,
    )

    star_cam = SkyCoord(
        x=star_cam_x * u.m,
        y=star_cam_y * u.m,
        frame=camera_frame,
    )

    star_altaz = star_cam.transform_to(altaz)
    star_alt = star_altaz.alt.to_value(u.rad)
    star_az = star_altaz.az.to_value(u.rad)

    star_az_2pi = star_az - 2.0 * np.pi
    if abs(star_az_2pi - 0.0) < abs(star_az - 0.0):
        star_az = star_az_2pi

    return star_alt, star_az


def altaz_to_camxy(source, subarray, run_id, tel_id, star_alt, star_az):
    obstime = Time("2013-11-01T03:00")
    location = EarthLocation.of_site("Roque de los Muchachos")
    altaz = AltAz(location=location, obstime=obstime)

    star_altaz = SkyCoord(
        alt=star_alt,
        az=star_az,
        frame=altaz,
    )

    tel_pointing_alt = source.observation_blocks[run_id].subarray_pointing_lat
    tel_pointing_az = source.observation_blocks[run_id].subarray_pointing_lon

    focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length

    tel_pointing = SkyCoord(
        alt=tel_pointing_alt,
        az=tel_pointing_az,
        frame=altaz,
    )

    camera_frame = CameraFrame(
        telescope_pointing=tel_pointing,
        focal_length=focal_length,
    )

    star_cam = star_altaz.transform_to(camera_frame)
    star_cam_x = star_cam.x.to_value(u.m)
    star_cam_y = star_cam.y.to_value(u.m)

    return star_cam_x, star_cam_y


def plot_xing_reconstruction(
    ctapipe_output,
    subarray,
    event,
    list_tel_id,
    list_image_moment,
    tag,
):
    event_id = event.index["event_id"]
    geometry = subarray.tel[list_tel_id[0]].camera.geometry

    xmax = max(geometry.pix_x) / u.m
    xmin = min(geometry.pix_x) / u.m
    ymax = max(geometry.pix_y) / u.m
    ymin = min(geometry.pix_y) / u.m

    fig, ax = plt.subplots()
    figsize_x = 8.6
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    display = CameraDisplay(geometry, ax=ax)

    list_cen_x = []
    list_cen_y = []
    list_angle = []
    list_a = []
    list_b = []
    list_a_err = []
    for img in range(0, len(list_tel_id)):
        tel_id = list_tel_id[img]
        clean_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
        boundary, picture, min_neighbors = cleaning_level[geometry.name]
        image_mask = tailcuts_clean(
            geometry,
            event.dl1.tel[tel_id].image,
            boundary_thresh=boundary,
            picture_thresh=picture,
            min_number_picture_neighbors=min_neighbors,
        )
        # for pix in range(0, len(image_mask)):
        #    if event.dl1.tel[tel_id].image[pix]==0.:
        #        image_mask[pix] = False
        #    else:
        #        image_mask[pix] = True

        for pix in range(0, len(image_mask)):
            if not image_mask[pix]:
                clean_image_1d[pix] = 0.0
            else:
                clean_image_1d[pix] = event.dl1.tel[tel_id].image[pix]

        line_a = list_image_moment[img][8]
        line_b = list_image_moment[img][9]
        line_a_err = list_image_moment[img][11]
        angle = list_image_moment[img][3]
        image_center_x = list_image_moment[img][1]
        image_center_y = list_image_moment[img][2]

        display.image += clean_image_1d
        list_a += [line_a]
        list_b += [line_b]
        list_a_err += [line_a_err]
        list_angle += [angle]
        list_cen_x += [image_center_x]
        list_cen_y += [image_center_y]

    display.cmap = "Reds"
    display.add_colorbar(ax=ax)

    for img in range(0, len(list_tel_id)):
        line_a = list_a[img]
        line_b = list_b[img]
        line_a_err = list_a_err[img]
        angle = list_angle[img]
        image_center_x = list_cen_x[img]
        image_center_y = list_cen_y[img]
        if np.cos(angle * u.rad) > 0.0:
            line_x = np.linspace(image_center_x, xmax, 100)
            line_y = line_a * line_x + line_b
            ax.plot(line_x, line_y, color="k", alpha=0.3, linestyle="dashed")
            # line_y = (
            #    (line_a + line_a_err) * line_x + line_b - line_a_err * image_center_x
            # )
            # ax.plot(line_x, line_y, color="k", alpha=0.3, linestyle="solid")
            # line_y = (
            #    (line_a - line_a_err) * line_x + line_b + line_a_err * image_center_x
            # )
            # ax.plot(line_x, line_y, color="k", alpha=0.3, linestyle="solid")
        else:
            line_x = np.linspace(xmin, image_center_x, 100)
            line_y = line_a * line_x + line_b
            ax.plot(line_x, line_y, color="k", alpha=0.3, linestyle="dashed")
            # line_y = (
            #    (line_a + line_a_err) * line_x + line_b - line_a_err * image_center_x
            # )
            # ax.plot(line_x, line_y, color="k", alpha=0.3, linestyle="solid")
            # line_y = (
            #    (line_a - line_a_err) * line_x + line_b + line_a_err * image_center_x
            # )
            # ax.plot(line_x, line_y, color="k", alpha=0.3, linestyle="solid")

    # ax.scatter(
    #    star_cam_x, star_cam_y, s=90, facecolors="none", c="r", marker="+"
    # )
    # ax.scatter(fit_cam_x, fit_cam_y, s=90, facecolors="none", edgecolors="r", marker="o")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.savefig(
        f"{ctapipe_output}/output_plots/evt{event_id}_xing_{tag}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()


def plot_monotel_reconstruction(
    ctapipe_output,
    subarray,
    run_id,
    tel_id,
    event,
    image_moment_array,
    star_cam_x,
    star_cam_y,
    fit_cam_x,
    fit_cam_y,
    tag,
):
    event_id = event.index["event_id"]
    geometry = subarray.tel[tel_id].camera.geometry

    dirty_image_1d = event.dl1.tel[tel_id].image
    dirty_image_2d = geometry.image_to_cartesian_representation(dirty_image_1d)
    remove_nan_pixels(dirty_image_2d)

    clean_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
    clean_time_1d = np.zeros_like(event.dl1.tel[tel_id].peak_time)
    boundary, picture, min_neighbors = cleaning_level[geometry.name]
    image_mask = tailcuts_clean(
        geometry,
        event.dl1.tel[tel_id].image,
        boundary_thresh=boundary,
        picture_thresh=picture,
        min_number_picture_neighbors=min_neighbors,
    )
    # for pix in range(0, len(image_mask)):
    #    if event.dl1.tel[tel_id].image[pix]==0.:
    #        image_mask[pix] = False
    #    else:
    #        image_mask[pix] = True

    for pix in range(0, len(image_mask)):
        if not image_mask[pix]:
            clean_image_1d[pix] = 0.0
            clean_time_1d[pix] = 0.0
        else:
            clean_image_1d[pix] = event.dl1.tel[tel_id].image[pix]
            clean_time_1d[pix] = event.dl1.tel[tel_id].peak_time[pix]

    center_time = reset_time(clean_image_1d, clean_time_1d)

    clean_image_2d = geometry.image_to_cartesian_representation(clean_image_1d)
    remove_nan_pixels(clean_image_2d)
    clean_time_2d = geometry.image_to_cartesian_representation(clean_time_1d)
    remove_nan_pixels(clean_time_2d)

    image_size = image_moment_array[0]
    image_center_x = image_moment_array[1]
    image_center_y = image_moment_array[2]
    angle = image_moment_array[3]
    semi_major = image_moment_array[4]
    semi_minor = image_moment_array[5]
    time_direction = image_moment_array[6]
    image_direction = image_moment_array[7]
    line_a = image_moment_array[8]
    line_b = image_moment_array[9]
    line_a_err = image_moment_array[11]

    xmax = max(geometry.pix_x) / u.m
    xmin = min(geometry.pix_x) / u.m
    ymax = max(geometry.pix_y) / u.m
    ymin = min(geometry.pix_y) / u.m

    fig, ax = plt.subplots()
    figsize_x = 8.6
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    display = CameraDisplay(geometry, ax=ax)
    display.image = clean_image_1d
    display.cmap = "Reds"
    display.add_colorbar(ax=ax)
    ax.scatter(star_cam_x, star_cam_y, s=90, facecolors="none", c="r", marker="+")
    ax.scatter(
        fit_cam_x, fit_cam_y, s=90, facecolors="none", edgecolors="r", marker="o"
    )
    if np.cos(angle * u.rad) > 0.0:
        line_x = np.linspace(image_center_x, xmax, 100)
        line_y = line_a * line_x + line_b
        ax.plot(line_x, line_y, color="k", alpha=0.3, linestyle="dashed")
        line_y = (line_a + line_a_err) * line_x + line_b - line_a_err * image_center_x
        ax.plot(line_x, line_y, color="k", alpha=0.3, linestyle="solid")
        line_y = (line_a - line_a_err) * line_x + line_b + line_a_err * image_center_x
        ax.plot(line_x, line_y, color="k", alpha=0.3, linestyle="solid")
    else:
        line_x = np.linspace(xmin, image_center_x, 100)
        line_y = line_a * line_x + line_b
        ax.plot(line_x, line_y, color="k", alpha=0.3, linestyle="dashed")
        line_y = (line_a + line_a_err) * line_x + line_b - line_a_err * image_center_x
        ax.plot(line_x, line_y, color="k", alpha=0.3, linestyle="solid")
        line_y = (line_a - line_a_err) * line_x + line_b + line_a_err * image_center_x
        ax.plot(line_x, line_y, color="k", alpha=0.3, linestyle="solid")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.savefig(
        f"{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_clean_image_{tag}.png",
        bbox_inches="tight",
    )
    del fig
    del ax
    plt.close()


def movie_simulation(
    telescope_type,
    subarray,
    run_id,
    tel_id,
    event,
    init_params,
    movie_lookup_table,
    movie_eigen_vectors,
    eco_image_1d,
):
    event_id = event.index["event_id"]
    geometry = subarray.tel[tel_id].camera.geometry

    n_eco_pix = len(eco_image_1d)

    fit_arrival = init_params[0]
    fit_impact = init_params[1]
    fit_log_energy = init_params[2]

    fit_movie_latent_space = []
    for r in range(0, len(movie_lookup_table)):
        fit_movie_latent_space += [
            movie_lookup_table[r].get_bin_content(
                fit_arrival, fit_impact, fit_log_energy
            )
        ]
    fit_movie_latent_space = np.array(fit_movie_latent_space)

    eco_movie_1d_fit = movie_eigen_vectors.T @ fit_movie_latent_space
    n_windows = int(select_samples / n_samples_per_window)
    sim_eco_image_1d = []
    sim_image_1d = []
    for win in range(0, n_windows):
        sim_eco_image_1d += [np.zeros_like(eco_image_1d)]
        sim_image_1d += [np.zeros_like(event.dl1.tel[tel_id].image)]

    sim_image_2d = []
    for win in range(0, n_windows):
        for pix in range(0, n_eco_pix):
            movie_pix_idx = pix + win * n_eco_pix
            sim_eco_image_1d[win][pix] = eco_movie_1d_fit[movie_pix_idx]
        image_cutout_restore(geometry, sim_eco_image_1d[win], sim_image_1d[win])
        sim_image_2d += [geometry.image_to_cartesian_representation(sim_image_1d[win])]

    return sim_image_2d


def display_a_movie(subarray, run_id, tel_id, event, eco_image_size, eco_movie_1d):
    n_windows = int(select_samples / n_samples_per_window)
    eco_image_1d = []
    for win in range(0, n_windows):
        eco_image_1d += [np.zeros(eco_image_size)]

    for win in range(0, n_windows):
        for pix in range(0, eco_image_size):
            entry = pix + win * eco_image_size
            eco_image_1d[win][pix] = eco_movie_1d[entry]

    event_id = event.index["event_id"]
    geometry = subarray.tel[tel_id].camera.geometry

    dirty_image_1d = event.dl1.tel[tel_id].image
    image_max = np.max(dirty_image_1d[:])
    image_1d = []
    for win in range(0, n_windows):
        image_1d += [np.zeros_like(dirty_image_1d)]

    list_image_2d = []
    for win in range(0, n_windows):
        image_cutout_restore(geometry, eco_image_1d[win], image_1d[win])
        image_2d = geometry.image_to_cartesian_representation(image_1d[win])
        list_image_2d += [image_2d]

    return dirty_image_1d, list_image_2d


def make_a_gif(
    ctapipe_output, subarray, run_id, tel_id, event, eco_image_1d, movie1_2d, movie2_2d
):
    event_id = event.index["event_id"]
    geometry = subarray.tel[tel_id].camera.geometry

    xmax = max(geometry.pix_x) / u.m
    xmin = min(geometry.pix_x) / u.m
    ymax = max(geometry.pix_y) / u.m
    ymin = min(geometry.pix_y) / u.m

    movie_2d = []
    for m in range(0, len(movie1_2d)):
        movie_2d += [np.vstack((movie1_2d[m], movie2_2d[m]))]

    image_max = np.max(eco_image_1d[:])
    n_windows = len(movie_2d)

    fig, ax = plt.subplots()
    figsize_x = 8.6
    figsize_y = 6.4
    fig.set_figheight(figsize_y)
    fig.set_figwidth(figsize_x)
    label_x = "X"
    label_y = "Y"
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    im = ax.imshow(
        movie_2d[0],
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        vmin=0.0,
        vmax=2.0 * image_max / float(n_windows),
    )
    cbar = fig.colorbar(im)

    def animate(i):
        im.set_array(movie_2d[i])
        return (im,)

    ani = animation.FuncAnimation(
        fig, animate, repeat=True, frames=len(movie_2d), interval=200
    )
    ani.save(
        f"{ctapipe_output}/output_plots/evt{event_id}_tel{tel_id}_movie.gif",
        writer=animation.PillowWriter(fps=4),
    )
    del fig
    del ax
    del ani
    plt.close()


def run_monoscopic_analysis(
    ctapipe_output,
    telescope_type,
    run_id,
    source,
    event,
    fast_conversion_poly_pkl,
    movie_lookup_table_pkl,
    movie_eigen_vectors_pkl,
    image_lookup_table_pkl,
    image_eigen_vectors_pkl,
    time_lookup_table_pkl,
    time_eigen_vectors_pkl,
    xing_alt,
    xing_az,
    xing_weight,
):
    analysis_result = []

    event_id = event.index["event_id"]
    ntel = len(event.r0.tel)

    truth_alt = float(event.simulation.shower.alt / u.rad)
    truth_az = float(event.simulation.shower.az / u.rad)
    truth_energy = event.simulation.shower.energy / u.TeV
    truth_log_energy = np.log10(truth_energy)

    list_tel_alt = []
    list_tel_az = []
    list_tel_log_energy = []
    list_tel_weight = []

    list_tel_id = []
    list_image_moment = []

    xing_err = 1e10
    if xing_weight > 0.0:
        xing_err = 1.0 / (pow(xing_weight, 0.5))
    use_seed = False
    if xing_err < 0.3 * np.pi / 180.0:
        use_seed = True
    print(f"use_seed = {use_seed}")

    ref_tel_id = 0
    for tel_idx in range(0, len(list(event.dl0.tel.keys()))):
        tel_id = list(event.dl0.tel.keys())[tel_idx]

        if str(telescope_type) != str(source.subarray.tel[tel_id]):
            continue
        ref_tel_id = tel_id

        xing_cam_x, xing_cam_y = altaz_to_camxy(
            source,
            source.subarray,
            run_id,
            tel_id,
            xing_alt * u.rad,
            xing_az * u.rad,
        )
        xing_camxy = None
        if use_seed:
            xing_camxy = [xing_cam_x, xing_cam_y]

        truth_info_array = find_image_truth(
            source, source.subarray, run_id, tel_id, event
        )
        truth_energy = float(truth_info_array[0] / u.TeV)
        truth_core_x = truth_info_array[1]
        truth_core_y = truth_info_array[2]
        # truth_alt = float(truth_info_array[3]/u.rad)
        # truth_az = float(truth_info_array[4]/u.rad)
        truth_height = truth_info_array[5]
        truth_xmax = truth_info_array[6]
        star_cam_x = truth_info_array[7]
        star_cam_y = truth_info_array[8]
        impact_x = truth_info_array[9]
        impact_y = truth_info_array[10]
        focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length / u.m

        tic_standard = time.perf_counter()
        (
            is_edge_image,
            image_moment_array,
            eco_movie_1d,
            eco_image_1d,
            eco_time_1d,
        ) = make_standard_movie(
            ctapipe_output,
            telescope_type,
            source.subarray,
            run_id,
            tel_id,
            event,
            flip=False,
            star_cam_xy=xing_camxy,
        )
        toc_standard = time.perf_counter()
        # print(f"standard: {toc_standard-tic_standard:0.1f} sec")

        image_size = image_moment_array[0]
        if image_size < image_size_cut:
            continue
        if is_edge_image:
            continue

        tic_reco = time.perf_counter()
        (
            image_fit_arrival,
            image_fit_impact,
            image_fit_log_energy,
            image_fit_arrival_err,
            image_fit_impact_err,
            image_fit_log_energy_err,
            image_fit_chi2,
        ) = single_movie_reconstruction(
            eco_image_1d,
            image_lookup_table_pkl,
            image_eigen_vectors_pkl,
            eco_time_1d,
            time_lookup_table_pkl,
            time_eigen_vectors_pkl,
            eco_movie_1d,
            movie_lookup_table_pkl,
            movie_eigen_vectors_pkl,
            fast_conversion_poly_pkl,
        )
        toc_reco = time.perf_counter()
        # print(f"reco: {toc_reco-tic_reco:0.1f} sec")

        time_direction = image_moment_array[6]
        image_direction = image_moment_array[7]

        add_ambiguity_unc = 0.0
        if not use_seed:
            (
                is_edge_image_flip,
                image_moment_array_flip,
                eco_movie_1d_flip,
                eco_image_1d_flip,
                eco_time_1d_flip,
            ) = make_standard_movie(
                ctapipe_output,
                telescope_type,
                source.subarray,
                run_id,
                tel_id,
                event,
                flip=True,
            )

            (
                image_fit_arrival_flip,
                image_fit_impact_flip,
                image_fit_log_energy_flip,
                image_fit_arrival_err_flip,
                image_fit_impact_err_flip,
                image_fit_log_energy_err_flip,
                image_fit_chi2_flip,
            ) = single_movie_reconstruction(
                eco_image_1d_flip,
                image_lookup_table_pkl,
                image_eigen_vectors_pkl,
                eco_time_1d_flip,
                time_lookup_table_pkl,
                time_eigen_vectors_pkl,
                eco_movie_1d_flip,
                movie_lookup_table_pkl,
                movie_eigen_vectors_pkl,
                fast_conversion_poly_pkl,
            )

            # print (f"image_fit_chi2 = {image_fit_chi2}")
            # print (f"image_fit_chi2_flip = {image_fit_chi2_flip}")
            image_ambiguity = 1.0 / (
                # abs(image_direction) * abs(image_fit_chi2 - image_fit_chi2_flip) / abs(image_fit_chi2 + image_fit_chi2_flip)
                abs(image_direction) * abs(image_fit_chi2 - image_fit_chi2_flip) / 25.0
            )
            add_ambiguity_unc = max(0.0, image_ambiguity - 1.0)

            if image_fit_chi2_flip < image_fit_chi2 and abs(image_direction) < 1.0:
                image_fit_arrival = image_fit_arrival_flip
                image_fit_impact = image_fit_impact_flip
                image_fit_log_energy = image_fit_log_energy_flip
                image_fit_arrival_err = image_fit_arrival_err_flip
                image_fit_impact_err = image_fit_impact_err_flip
                image_fit_log_energy_err = image_fit_log_energy_err_flip
                image_fit_chi2 = image_fit_chi2_flip
                image_moment_array = image_moment_array_flip

        image_center_x = image_moment_array[1]
        image_center_y = image_moment_array[2]
        angle = image_moment_array[3]
        angle_err = image_moment_array[13]

        line_a = image_moment_array[8]
        line_b = image_moment_array[9]
        line_a_err = image_moment_array[11]
        line_b_err = image_moment_array[12]

        image_fit_cam_x = image_center_x + image_fit_arrival * np.cos(angle * u.rad)
        image_fit_cam_y = image_center_y + image_fit_arrival * np.sin(angle * u.rad)
        image_fit_alt, image_fit_az = camxy_to_altaz(
            source, source.subarray, run_id, tel_id, image_fit_cam_x, image_fit_cam_y
        )
        image_center_alt, image_center_az = camxy_to_altaz(
            source, source.subarray, run_id, tel_id, image_center_x, image_center_y
        )

        line_altaz_a = (image_fit_az - image_center_az) / (
            image_fit_alt - image_center_alt
        )
        line_altaz_b = image_center_az - line_altaz_a * image_center_alt
        # line_altaz_err = pow(pow(image_fit_arrival*angle_err,2)+pow(line_b_err,2),0.5)
        line_altaz_err = angle_err

        xing_arrival = pow(
            pow(xing_cam_x - image_center_x, 2) + pow(xing_cam_y - image_center_y, 2),
            0.5,
        )
        image_method_unc = (
            pow(
                pow(angle_err * xing_arrival, 2) + pow(image_fit_arrival_err, 2),
                # pow(angle_err * xing_arrival, 2),
                0.5,
            )
            / focal_length
        )
        image_method_unc = pow(
            pow(add_ambiguity_unc, 2) + pow(image_method_unc, 2), 0.5
        )

        image_method_error = (
            pow(
                pow(image_fit_cam_x - star_cam_x, 2)
                + pow(image_fit_cam_y - star_cam_y, 2),
                0.5,
            )
            / focal_length
            * 180.0
            / np.pi
        )

        # print(f"truth_energy     = {truth_energy}")
        # print(f"image_fit_energy = {pow(10.,image_fit_log_energy)}")
        # print(f"truth_fit_impact = {pow(impact_x*impact_x+impact_y*impact_y,0.5)}")
        # print(f"image_fit_impact = {image_fit_impact}")
        # print(f"image_method_error = {image_method_error:0.3f} deg")
        # print(f"image_method_unc = {image_method_unc:0.3f} deg")

        # if image_method_unc < 1.*xing_err:
        # if image_method_unc < 0.5:
        list_tel_alt += [image_fit_alt]
        list_tel_az += [image_fit_az]
        list_tel_log_energy += [image_fit_log_energy]
        list_tel_weight += [pow(1.0 / image_method_unc, 2)]
        print(
            f"image_size = {image_size:0.1f}, image_method_unc = {image_method_unc*180./np.pi:0.3f}, image_method_off_angle = {image_method_error:0.3f}"
        )

        list_tel_id += [tel_id]
        list_image_moment += [image_moment_array]

        if (
            # image_size > plot_image_size_cut
            image_size > image_size_cut
            and image_method_error > 0.6
            and image_method_unc * 180.0 / np.pi < 0.5
        ):
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"image_size = {image_size}")
            print(f"image_method_unc = {image_method_unc*180./np.pi:0.3f} deg")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            fit_params = [image_fit_arrival, image_fit_impact, image_fit_log_energy]
            plot_monotel_reconstruction(
                ctapipe_output,
                source.subarray,
                run_id,
                tel_id,
                event,
                image_moment_array,
                star_cam_x,
                star_cam_y,
                image_fit_cam_x,
                image_fit_cam_y,
                "movie",
            )
            sim_movie = movie_simulation(
                telescope_type,
                source.subarray,
                run_id,
                tel_id,
                event,
                fit_params,
                movie_lookup_table_pkl,
                movie_eigen_vectors_pkl,
                eco_image_1d,
            )
            data_image, data_movie = display_a_movie(
                source.subarray, run_id, tel_id, event, len(eco_image_1d), eco_movie_1d
            )
            make_a_gif(
                ctapipe_output,
                source.subarray,
                run_id,
                tel_id,
                event,
                data_image,
                data_movie,
                sim_movie,
            )

    avg_tmp_alt = 0.0
    avg_tmp_az = 0.0
    avg_tmp_log_energy = 0.0
    avg_tmp_weight = 0.0
    new_list_tel_alt = []
    new_list_tel_az = []
    new_list_tel_log_energy = []
    new_list_tel_weight = []
    if len(list_tel_alt) > 0:
        avg_tmp_err = 0.0
        tmp_weight = 0.0
        for tel in range(0, len(list_tel_alt)):
            evt_alt = list_tel_alt[tel]
            evt_az = list_tel_az[tel]
            evt_log_energy = list_tel_log_energy[tel]

            evt_az_2pi = evt_az - 2.0 * np.pi
            if abs(evt_az_2pi - 0.0) < abs(evt_az - 0.0):
                evt_az = evt_az_2pi

            # TS = pow((evt_alt-init_tmp_alt)/rms_tmp_alt,2) + pow((evt_az-init_tmp_az)/rms_tmp_az,2)
            # if TS>1.: continue

            avg_tmp_alt += evt_alt * list_tel_weight[tel]
            avg_tmp_az += evt_az * list_tel_weight[tel]
            avg_tmp_log_energy += evt_log_energy * list_tel_weight[tel]
            avg_tmp_err += 1.0
            tmp_weight += list_tel_weight[tel]

            new_list_tel_alt += [evt_alt]
            new_list_tel_az += [evt_az]
            new_list_tel_log_energy += [evt_log_energy]
            new_list_tel_weight += [list_tel_weight[tel]]

        avg_tmp_alt = avg_tmp_alt / tmp_weight
        avg_tmp_az = avg_tmp_az / tmp_weight
        avg_tmp_log_energy = avg_tmp_log_energy / tmp_weight
        avg_tmp_err = pow(avg_tmp_err / tmp_weight, 0.5)
        avg_tmp_weight = 1.0 / (avg_tmp_err * avg_tmp_err)

    # if not use_seed and len(list_tel_id)>=2:
    #    plot_xing_reconstruction(
    #        ctapipe_output,
    #        source.subarray,
    #        event,
    #        list_tel_id,
    #        list_image_moment,
    #        "xing",
    #    )

    return (
        truth_log_energy,
        avg_tmp_log_energy,
        avg_tmp_alt,
        avg_tmp_az,
        avg_tmp_weight,
        new_list_tel_log_energy,
        new_list_tel_alt,
        new_list_tel_az,
    )


def run_multiscopic_analysis(ctapipe_output, telescope_type, run_id, source, event):
    event_id = event.index["event_id"]
    ntel = len(event.r0.tel)

    truth_alt = float(event.simulation.shower.alt / u.rad)
    truth_az = float(event.simulation.shower.az / u.rad)

    list_line_x = []
    list_line_y = []
    list_line_intensity = []
    list_line_length = []
    list_line_width = []
    list_line_a = []
    list_line_b = []
    list_line_a_err = []
    list_line_b_err = []
    list_line_w = []
    list_tel_id = []
    list_image_moment = []

    for tel_idx in range(0, len(list(event.dl0.tel.keys()))):
        tel_id = list(event.dl0.tel.keys())[tel_idx]

        if str(telescope_type) != str(source.subarray.tel[tel_id]):
            continue

        truth_info_array = find_image_truth(
            source, source.subarray, run_id, tel_id, event
        )
        truth_energy = float(truth_info_array[0] / u.TeV)
        truth_core_x = truth_info_array[1]
        truth_core_y = truth_info_array[2]
        # truth_alt = float(truth_info_array[3]/u.rad)
        # truth_az = float(truth_info_array[4]/u.rad)
        truth_height = truth_info_array[5]
        truth_xmax = truth_info_array[6]
        star_cam_x = truth_info_array[7]
        star_cam_y = truth_info_array[8]
        impact_x = truth_info_array[9]
        impact_y = truth_info_array[10]
        focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length / u.m

        (
            is_edge_image,
            image_moment_array,
            eco_movie_1d,
            eco_image_1d,
            eco_time_1d,
        ) = make_standard_movie(
            ctapipe_output,
            telescope_type,
            source.subarray,
            run_id,
            tel_id,
            event,
            flip=False,
        )

        image_size = image_moment_array[0]
        if image_size < image_size_cut:
            continue

        list_tel_id += [tel_id]
        list_image_moment += [image_moment_array]

        image_center_x = image_moment_array[1]
        image_center_y = image_moment_array[2]
        angle = image_moment_array[3]
        length = image_moment_array[4]
        width = image_moment_array[5]
        angle_err = image_moment_array[13]

        line_a = image_moment_array[8]
        line_b = image_moment_array[9]
        line_a_err = image_moment_array[11]
        line_b_err = image_moment_array[12]

        image_center_alt, image_center_az = camxy_to_altaz(
            source, source.subarray, run_id, tel_id, image_center_x, image_center_y
        )

        image_tail_alt, image_tail_az = camxy_to_altaz(
            source,
            source.subarray,
            run_id,
            tel_id,
            image_center_x + 0.01,
            line_a * (image_center_x + 0.01) + line_b,
        )

        line_altaz_a = (image_tail_az - image_center_az) / (
            image_tail_alt - image_center_alt
        )
        line_altaz_b = image_center_az - line_altaz_a * image_center_alt
        line_altaz_a_err = line_a_err
        line_altaz_b_err = line_b_err / focal_length

        list_line_x += [image_center_alt]
        list_line_y += [image_center_az]
        list_line_a += [line_altaz_a]
        list_line_b += [line_altaz_b]
        list_line_a_err += [line_altaz_a_err]
        list_line_b_err += [line_altaz_b_err]
        list_line_intensity += [image_size]
        list_line_length += [length]
        list_line_width += [width]

    xing_alt = 0.0
    xing_az = 0.0
    xing_weight = 0.0
    if len(list_line_a) > 1:
        (
            xing_alt,
            xing_az,
            xing_err,
        ) = find_intersection_multiple_lines(
            list_line_x,
            list_line_y,
            list_line_a,
            list_line_b,
            list_line_a_err,
            list_line_b_err,
            list_line_intensity,
            list_line_length,
            list_line_width,
        )

        xing_weight = 1.0 / (xing_err * xing_err)

        xing_off_angle = (
            angular_separation(
                truth_az * u.rad, truth_alt * u.rad, xing_az * u.rad, xing_alt * u.rad
            )
            .to(u.rad)
            .value
        )

        # if xing_err*180./np.pi>1. and len(list_tel_id)>=2:
        if xing_off_angle * 180.0 / np.pi > 0.5 and len(list_tel_id) >= 2:
            print("plot xing reconstruction.")
            # print (f"xing_off_angle = {xing_off_angle*180./np.pi}")
            # print (f"xing_err = {xing_err*180./np.pi}")
            plot_xing_reconstruction(
                ctapipe_output,
                source.subarray,
                event,
                list_tel_id,
                list_image_moment,
                "xing",
            )
            for img in range(0, len(list_image_moment)):
                tel_id = list_tel_id[img]
                image_moment_array = list_image_moment[img]
                plot_monotel_reconstruction(
                    ctapipe_output,
                    source.subarray,
                    run_id,
                    tel_id,
                    event,
                    image_moment_array,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    "movie",
                )
            # exit()

    return xing_alt, xing_az, xing_weight


def loop_all_events(training_sample_path, ctapipe_output, telescope_type):
    analysis_result = []
    lookup_table_type = "box3d"

    print("loading svd pickle data... ")
    fast_conversion_type = "image"
    fast_conversion_poly_pkl = []
    output_filename = f"{ctapipe_output}/output_machines/{fast_conversion_type}_fast_conversion_arrival_{telescope_type}.pkl"
    fast_conversion_poly_pkl += [pickle.load(open(output_filename, "rb"))]
    output_filename = f"{ctapipe_output}/output_machines/{fast_conversion_type}_fast_conversion_impact_{telescope_type}.pkl"
    fast_conversion_poly_pkl += [pickle.load(open(output_filename, "rb"))]
    output_filename = f"{ctapipe_output}/output_machines/{fast_conversion_type}_fast_conversion_log_energy_{telescope_type}.pkl"
    fast_conversion_poly_pkl += [pickle.load(open(output_filename, "rb"))]

    output_filename = f"{ctapipe_output}/output_machines/movie_{lookup_table_type}_lookup_table_{telescope_type}.pkl"
    movie_lookup_table_pkl = pickle.load(open(output_filename, "rb"))
    output_filename = (
        f"{ctapipe_output}/output_machines/movie_eigen_vectors_{telescope_type}.pkl"
    )
    movie_eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))

    output_filename = f"{ctapipe_output}/output_machines/image_{lookup_table_type}_lookup_table_{telescope_type}.pkl"
    image_lookup_table_pkl = pickle.load(open(output_filename, "rb"))
    output_filename = (
        f"{ctapipe_output}/output_machines/image_eigen_vectors_{telescope_type}.pkl"
    )
    image_eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))

    output_filename = f"{ctapipe_output}/output_machines/time_{lookup_table_type}_lookup_table_{telescope_type}.pkl"
    time_lookup_table_pkl = pickle.load(open(output_filename, "rb"))
    output_filename = (
        f"{ctapipe_output}/output_machines/time_eigen_vectors_{telescope_type}.pkl"
    )
    time_eigen_vectors_pkl = pickle.load(open(output_filename, "rb"))

    print(f"loading file: {training_sample_path}")
    source = SimTelEventSource(training_sample_path, focal_length_choice="EQUIVALENT")

    # Explore the instrument description
    subarray = source.subarray
    print(subarray.to_table())

    calib = CameraCalibrator(subarray=subarray)
    image_processor = ImageProcessor(subarray=subarray)
    shower_processor = ShowerProcessor(subarray=subarray)

    ob_keys = source.observation_blocks.keys()
    run_id = list(ob_keys)[0]

    tel_pointing_alt = float(
        source.observation_blocks[run_id].subarray_pointing_lat / u.rad
    )
    tel_pointing_az = float(
        source.observation_blocks[run_id].subarray_pointing_lon / u.rad
    )
    print(f"tel_pointing_alt = {tel_pointing_alt}")
    print(f"tel_pointing_az = {tel_pointing_az}")

    sum_combine_result = []
    sum_hillas_result = []
    sum_xing_result = []
    sum_template_result = []

    for event in source:
        print(
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )

        event_id = event.index["event_id"]
        if select_event_id != 0:
            if event_id != select_event_id:
                continue

        print(f"Select telescope type: {telescope_type}")
        print(f"event_id = {event_id}")

        calib(event)  # fills in r1, dl0, and dl1
        image_processor(event)
        shower_processor(event)
        # print(f"event = {event}")
        # exit()

        reco_result = event.dl2.stereo.geometry["HillasReconstructor"]
        n_tels = len(reco_result.telescopes)
        if n_tels == 0:
            continue
        # if n_tels>2: continue

        truth_alt = float(event.simulation.shower.alt / u.rad)
        truth_az = float(event.simulation.shower.az / u.rad)

        hillas_alt = 0.0
        hillas_az = 0.0
        hillas_err = 0.0
        hillas_weight = 0.0
        if reco_result.is_valid:
            hillas_alt = reco_result.alt.to(u.rad).value
            hillas_az = reco_result.az.to(u.rad).value
            hillas_az_2pi = hillas_az - 2.0 * np.pi
            if abs(hillas_az_2pi - 0.0) < abs(hillas_az - 0.0):
                hillas_az = hillas_az_2pi
            hillas_alt_err = reco_result.alt_uncert.to(u.rad).value
            hillas_az_err = reco_result.az_uncert.to(u.rad).value
            hillas_err = pow(
                hillas_alt_err * hillas_alt_err + hillas_az_err * hillas_az_err, 0.5
            )
            if hillas_err > 0.01 * np.pi / 180.0:
                hillas_weight = 1.0 / (hillas_err * hillas_err)
            hillas_off_angle = angular_separation(
                truth_az * u.rad, truth_alt * u.rad, reco_result.az, reco_result.alt
            )
            print(
                f"hillas_off_angle = {hillas_off_angle.to(u.deg).value:0.3f} +/- {hillas_err*180./np.pi:0.3f} deg"
            )
            sum_hillas_result += [
                [
                    hillas_off_angle.to(u.deg).value,
                    hillas_err * 180.0 / np.pi,
                    reco_result.alt.to(u.deg).value,
                    reco_result.az.to(u.deg).value,
                ]
            ]
        else:
            print("hillas reconstruction is not valid.")

        tic_xing = time.perf_counter()
        xing_alt, xing_az, xing_weight = run_multiscopic_analysis(
            ctapipe_output, telescope_type, run_id, source, event
        )
        toc_xing = time.perf_counter()
        xing_err = 0.0

        if xing_weight > 0.0:
            xing_off_angle = angular_separation(
                truth_az * u.rad, truth_alt * u.rad, xing_az * u.rad, xing_alt * u.rad
            )
            xing_err = pow(1.0 / xing_weight, 0.5)
            print(
                f"xing_off_angle = {xing_off_angle.to(u.deg).value:0.3f} +/- {xing_err*180./np.pi:0.3f} deg ({toc_xing-tic_xing:0.1f} sec)"
            )
            sum_xing_result += [
                [
                    xing_off_angle.to(u.deg).value,
                    xing_err * 180.0 / np.pi,
                    xing_alt * 180.0 / np.pi,
                    xing_az * 180.0 / np.pi,
                ]
            ]

        seed_alt = 0.0
        seed_az = 0.0
        seed_weight = 0.0
        if hillas_weight > xing_weight:
            print("use hillas seed.")
            seed_alt = hillas_alt
            seed_az = hillas_az
            seed_weight = hillas_weight
        else:
            print("use xing seed.")
            seed_alt = xing_alt
            seed_az = xing_az
            seed_weight = xing_weight

        tic_template = time.perf_counter()
        (
            truth_log_energy,
            avg_tmp_log_energy,
            avg_tmp_alt,
            avg_tmp_az,
            avg_tmp_weight,
            list_tmp_log_energy,
            list_tmp_alt,
            list_tmp_az,
        ) = run_monoscopic_analysis(
            ctapipe_output,
            telescope_type,
            run_id,
            source,
            event,
            fast_conversion_poly_pkl,
            movie_lookup_table_pkl,
            movie_eigen_vectors_pkl,
            image_lookup_table_pkl,
            image_eigen_vectors_pkl,
            time_lookup_table_pkl,
            time_eigen_vectors_pkl,
            seed_alt,
            seed_az,
            seed_weight,
        )
        toc_template = time.perf_counter()

        if avg_tmp_weight > 0.0:
            tmp_err = pow(1.0 / avg_tmp_weight, 0.5)
            tmp_off_angle = angular_separation(
                truth_az * u.rad,
                truth_alt * u.rad,
                avg_tmp_az * u.rad,
                avg_tmp_alt * u.rad,
            )
            print(
                f"tmp_off_angle = {tmp_off_angle.to(u.deg).value:0.3f} +/- {tmp_err*180./np.pi:0.3f} deg ({toc_template-tic_template:0.1f} sec)"
            )
            print(
                f"truth_log_energy = {pow(10.,truth_log_energy):0.2f} TeV, tmp_log_energy = {pow(10.,avg_tmp_log_energy):0.2f} TeV"
            )
            list_tmp_alt = np.array(list_tmp_alt) * 180.0 / np.pi
            list_tmp_az = np.array(list_tmp_az) * 180.0 / np.pi
            sum_template_result += [
                [
                    tmp_off_angle.to(u.deg).value,
                    tmp_err * 180.0 / np.pi,
                    avg_tmp_alt * 180.0 / np.pi,
                    avg_tmp_az * 180.0 / np.pi,
                    truth_log_energy,
                    avg_tmp_log_energy,
                    list_tmp_alt,
                    list_tmp_az,
                    list_tmp_log_energy,
                    toc_template - tic_template,
                ]
            ]

        combine_alt = 0.0
        combine_az = 0.0
        combine_weight = 0.0
        combine_err = 0.0
        combine_alt += avg_tmp_alt * avg_tmp_weight
        combine_az += avg_tmp_az * avg_tmp_weight
        combine_weight += avg_tmp_weight
        combine_err += 1.0
        combine_alt += xing_alt * xing_weight
        combine_az += xing_az * xing_weight
        combine_weight += xing_weight
        combine_err += 1.0
        # combine_alt += hillas_alt * hillas_weight
        # combine_az += hillas_az * hillas_weight
        # combine_weight += hillas_weight
        # combine_err += 1.0
        if combine_weight > 0.0:
            combine_alt = combine_alt / combine_weight
            combine_az = combine_az / combine_weight
            combine_err = pow(combine_err / combine_weight, 0.5)
            combine_off_angle = angular_separation(
                truth_az * u.rad,
                truth_alt * u.rad,
                combine_az * u.rad,
                combine_alt * u.rad,
            )
            print(
                f"combine_off_angle = {combine_off_angle.to(u.deg).value:0.3f} +/- {combine_err*180./np.pi:0.3f} deg"
            )
            sum_combine_result += [
                [combine_off_angle.to(u.deg).value, combine_err * 180.0 / np.pi]
            ]

        ana_tag = "veritas"
        analysis_result = [
            sum_hillas_result,
            sum_xing_result,
            sum_template_result,
            sum_combine_result,
        ]
        output_filename = f"{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}_{telescope_type}.pkl"
        print(f"writing file to {output_filename}")
        with open(output_filename, "wb") as file:
            pickle.dump(analysis_result, file)

        print(
            "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        )
        if select_event_id != 0:
            exit()
