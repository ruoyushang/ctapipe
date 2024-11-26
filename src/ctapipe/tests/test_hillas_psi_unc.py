import os, sys
import pickle

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, angular_separation
from astropy.time import Time
from common_tools import denoising_image, univ_inv_sol, cleaning_image
from matplotlib import pyplot as plt
from scipy.optimize import brute, minimize

from ctapipe.calib import CameraCalibrator
from ctapipe.coordinates import CameraFrame, NominalFrame
from ctapipe.image import ImageProcessor, hillas_parameters, number_of_islands
from ctapipe.io import SimTelEventSource
from ctapipe.reco import ShowerProcessor
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.visualization import CameraDisplay

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")


# array_type = 'LST_Nectar_ASTRI'
array_type = 'SCT'
#array_type = "NectarCam"
# array_type = 'Flash'
# array_type = 'LSTCam'
#array_type = 'ASTRI'
# array_type = 'CHEC'
#array_type = 'DigiCam'
# array_type = 'MIX'
#array_type = 'LSTCam_NectarCam_ASTRICam'

pointing = 'proton'
#pointing = "onaxis"
# pointing = 'diffuse'

if len(sys.argv)>1:
    array_type = sys.argv[1]
    pointing = sys.argv[2]
print (f"array_type = {array_type}")


make_plot = False
#make_plot = True

ana_tag = f"psi_unc_{array_type}_{pointing}"

select_evt = None
#run_id = 401
#event_id = 104
#select_evt = [run_id, event_id]

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


#denoiser_model_pkl = {}
#if "Flash" in ana_tag:
#    output_filename = f"{ctapipe_output}/output_machines/denoiser_model_FlashCam.pkl"
#    if not os.path.exists(output_filename):
#        print(f"{output_filename} does not exist.")
#        exit()
#    else:
#        denoiser_model_pkl["FlashCam"] = pickle.load(open(output_filename, "rb"))
#if "Nectar" in ana_tag:
#    output_filename = f"{ctapipe_output}/output_machines/denoiser_model_NectarCam.pkl"
#    if not os.path.exists(output_filename):
#        print(f"{output_filename} does not exist.")
#        exit()
#    else:
#        denoiser_model_pkl["NectarCam"] = pickle.load(open(output_filename, "rb"))
#if "LST" in ana_tag:
#    output_filename = f"{ctapipe_output}/output_machines/denoiser_model_LSTCam.pkl"
#    if not os.path.exists(output_filename):
#        print(f"{output_filename} does not exist.")
#        exit()
#    else:
#        denoiser_model_pkl["LSTCam"] = pickle.load(open(output_filename, "rb"))

denoiser_model_pkl = None
output_filename = f"{ctapipe_output}/output_machines/denoiser_model_LSTCam.pkl"
if not os.path.exists(output_filename):
    print(f"{output_filename} does not exist.")
    exit()
else:
    denoiser_model_pkl = pickle.load(open(output_filename, "rb"))

def source_location_chi2(
    input_xy,
    list_img_size,
    list_img_length,
    list_img_width,
    list_img_nom_cen_x,
    list_img_nom_cen_y,
    list_img_angle,
    list_img_unc_angle,
    list_img_ry_unc,
    list_img_frac_leakage,
):
    input_x = input_xy[0]
    input_y = input_xy[1]

    # sorted_unc = sorted(list_img_unc_angle)
    # second_smallest_unc = sorted_unc[1]
    # sorted_size = sorted(list_img_size, reverse=True)
    # second_largest_size = sorted_size[1]

    chi2_default = 0.0
    chi2_new = 0.0
    total_weight = 0.0
    for img in range(0, len(list_img_angle)):
        # if list_img_unc_angle[img] > 3.*second_smallest_unc:
        #    continue
        # if list_img_size[img] < 0.1 * second_largest_size:
        #    continue

        angle_rad = list_img_angle[img]
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad)],
                [np.sin(angle_rad), np.cos(angle_rad)],
            ]
        )

        trans_try_x = np.array(input_x) - list_img_nom_cen_x[img]
        trans_try_y = np.array(input_y) - list_img_nom_cen_y[img]
        initi_coord = np.array([trans_try_x, trans_try_y])
        rotat_coord = rotation_matrix @ initi_coord
        rotat_try_x = 1.0 * rotat_coord[0]
        rotat_try_y = -1.0 * rotat_coord[1]

        unc_minor = pow(
            pow(rotat_try_y * list_img_unc_angle[img], 2)
            + pow(list_img_ry_unc[img], 2),
            0.5,
        )
        # unc_minor = rotat_try_y * list_img_unc_angle[img]

        # weight = list_img_size[img] * list_img_length[img] / list_img_width[img]
        # chi2_default += weight * pow(rotat_try_x,2)

        # weight_new = np.sqrt(list_img_size[img])
        weight_new = 1.0
        total_weight += weight_new
        chi2_new += weight_new * pow((rotat_try_x) / unc_minor, 2)

    return chi2_new / total_weight


def compute_location_uncertainty(
    input_xy,
    list_img_size,
    list_img_length,
    list_img_width,
    list_img_nom_cen_x,
    list_img_nom_cen_y,
    list_img_angle,
    list_img_unc_angle,
    list_img_ry_unc,
    list_img_frac_leakage,
):
    src_nom_x = input_xy[0]
    src_nom_y = input_xy[1]
    src_chi2 = source_location_chi2(
        [src_nom_x, src_nom_y],
        list_img_size,
        list_img_length,
        list_img_width,
        list_img_nom_cen_x,
        list_img_nom_cen_y,
        list_img_angle,
        list_img_unc_angle,
        list_img_ry_unc,
        list_img_frac_leakage,
    )
    src_chi = pow(src_chi2, 0.5)

    map_npix = 101
    map_size_deg = 2.0
    xmax = map_size_deg
    xmin = -map_size_deg
    ymax = map_size_deg
    ymin = -map_size_deg

    map_shape = (map_npix, map_npix)
    map_pix_w = []
    map_pix_x = []
    map_pix_y = []

    center_pix = round(float(map_npix) / 2.0)
    delta_radius_deg = 0.1
    for r in range(0, int(map_size_deg / delta_radius_deg)):
        local_min_chi = 10.0
        radius_deg = float(r + 1) * delta_radius_deg
        for pix_x in range(0, map_npix):
            for pix_y in range(0, map_npix):
                nom_x = (
                    float(pix_x - center_pix)
                    / float(map_npix)
                    * 2.0
                    * map_size_deg
                    / (180.0 / np.pi)
                    + src_nom_x
                )
                nom_y = (
                    float(pix_y - center_pix)
                    / float(map_npix)
                    * 2.0
                    * map_size_deg
                    / (180.0 / np.pi)
                    + src_nom_y
                )
                dist_to_src = (
                    pow(pow(src_nom_x - nom_x, 2) + pow(src_nom_y - nom_y, 2), 0.5)
                    * 180.0
                    / np.pi
                )
                if (
                    dist_to_src >= radius_deg
                    or dist_to_src < radius_deg - delta_radius_deg
                ):
                    continue
                chi2 = source_location_chi2(
                    [nom_x, nom_y],
                    list_img_size,
                    list_img_length,
                    list_img_width,
                    list_img_nom_cen_x,
                    list_img_nom_cen_y,
                    list_img_angle,
                    list_img_unc_angle,
                    list_img_ry_unc,
                    list_img_frac_leakage,
                )
                if np.isnan(chi2):
                    continue
                chi = pow(chi2, 0.5)
                map_pix_w += [chi2]
                map_pix_x += [nom_x]
                map_pix_y += [nom_y]
                if local_min_chi > chi:
                    local_min_chi = chi
        if local_min_chi > src_chi + 1.0:
            break

    unc_center_x = 0.0
    unc_center_y = 0.0
    total_weight = 0.0
    for pix in range(0, len(map_pix_w)):
        unc_center_x += map_pix_x[pix] * np.exp(-map_pix_w[pix])
        unc_center_y += map_pix_y[pix] * np.exp(-map_pix_w[pix])
        total_weight += np.exp(-map_pix_w[pix])
    if total_weight == 0.0:
        return 45.0 * np.pi / 180.0
    unc_center_x = unc_center_x / total_weight
    unc_center_y = unc_center_y / total_weight

    cov_xx = 0.0
    cov_xy = 0.0
    cov_yx = 0.0
    cov_yy = 0.0
    total_weight = 0.0
    for pix in range(0, len(map_pix_w)):
        diff_x = map_pix_x[pix] - unc_center_x
        diff_y = map_pix_y[pix] - unc_center_y
        weight = np.exp(-map_pix_w[pix])
        cov_xx += diff_x * diff_x * weight
        cov_xy += diff_x * diff_y * weight
        cov_yx += diff_y * diff_x * weight
        cov_yy += diff_y * diff_y * weight
        total_weight += weight
    cov_xx = cov_xx / total_weight
    cov_xy = cov_xy / total_weight
    cov_yx = cov_yx / total_weight
    cov_yy = cov_yy / total_weight

    covariance_matrix = np.array([[cov_xx, cov_xy], [cov_yx, cov_yy]])
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    semi_major_sq = eigenvalues[0]
    semi_minor_sq = eigenvalues[1]
    unc_major = 2.0 * max(pow(semi_major_sq, 0.5), pow(semi_minor_sq, 0.5))

    pix_width = map_size_deg / float(map_npix) * np.pi / 180.0
    return max(pix_width, unc_major)


def camxy_to_nominal(source, run_id, tel_id, star_cam_x, star_cam_y):
    subarray = source.subarray
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

    nominal_frame = NominalFrame(origin=tel_pointing)

    camera_frame = CameraFrame(
        telescope_pointing=tel_pointing,
        focal_length=focal_length,
    )

    star_cam = SkyCoord(
        x=star_cam_x * u.m,
        y=star_cam_y * u.m,
        frame=camera_frame,
    )

    star_nom_xy = star_cam.transform_to(nominal_frame)
    star_nom_x = star_nom_xy.fov_lon.to(u.rad).value
    star_nom_y = star_nom_xy.fov_lat.to(u.rad).value

    return star_nom_x, star_nom_y


def altaz_to_nominal(source, run_id, star_alt, star_az):
    subarray = source.subarray

    obstime = Time("2013-11-01T03:00")
    location = EarthLocation.of_site("Roque de los Muchachos")
    altaz = AltAz(location=location, obstime=obstime)

    if star_alt > np.pi / 2.0 * u.rad:
        star_alt = np.pi * u.rad - star_alt

    star_altaz = SkyCoord(
        alt=star_alt,
        az=star_az,
        frame=altaz,
    )

    tel_pointing_alt = source.observation_blocks[run_id].subarray_pointing_lat
    tel_pointing_az = source.observation_blocks[run_id].subarray_pointing_lon

    tel_pointing = SkyCoord(
        alt=tel_pointing_alt,
        az=tel_pointing_az,
        frame=altaz,
    )

    nominal_frame = NominalFrame(origin=tel_pointing)

    star_nom_xy = star_altaz.transform_to(nominal_frame)
    star_nom_x = star_nom_xy.fov_lon.to(u.rad).value
    star_nom_y = star_nom_xy.fov_lat.to(u.rad).value

    return star_nom_x, star_nom_y


def nominal_to_altaz(source, run_id, star_nom_x, star_nom_y):
    subarray = source.subarray

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

    nominal_frame = NominalFrame(origin=tel_pointing)

    star_nom_xy = SkyCoord(
        fov_lon=star_nom_x,
        fov_lat=star_nom_y,
        frame=nominal_frame,
    )

    star_altaz = star_nom_xy.transform_to(altaz)
    star_alt = star_altaz.alt.to_value(u.rad)
    star_az = star_altaz.az.to_value(u.rad)

    star_az_2pi = star_az - 2.0 * np.pi
    if abs(star_az_2pi - 0.0) < abs(star_az - 0.0):
        star_az = star_az_2pi

    return star_alt, star_az


def altaz_to_camxy(source, run_id, tel_id, star_alt, star_az):
    subarray = source.subarray

    obstime = Time("2013-11-01T03:00")
    location = EarthLocation.of_site("Roque de los Muchachos")
    altaz = AltAz(location=location, obstime=obstime)

    if star_alt > np.pi / 2.0 * u.rad:
        star_alt = np.pi * u.rad - star_alt

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


def psi_uncertainty_truncated_image(geometry, image):
    border_pixels_1 = geometry.get_border_pixel_mask(1)
    border_pixels_2 = geometry.get_border_pixel_mask(2)
    border_pixels_3 = geometry.get_border_pixel_mask(3)

    image_edge_1 = np.zeros_like(image)
    image_edge_2 = np.zeros_like(image)
    image_edge_3 = np.zeros_like(image)
    for pix in range(0, len(image)):
        if border_pixels_1[pix]:
            image_edge_1[pix] = 0.0
        else:
            image_edge_1[pix] = image[pix]
        if border_pixels_2[pix]:
            image_edge_2[pix] = 0.0
        else:
            image_edge_2[pix] = image[pix]
        if border_pixels_3[pix]:
            image_edge_3[pix] = 0.0
        else:
            image_edge_3[pix] = image[pix]

    if np.sum(image_edge_1) == 0.0:
        return 45.0 * np.pi / 180.0
    if np.sum(image_edge_2) == 0.0:
        return 45.0 * np.pi / 180.0
    if np.sum(image_edge_3) == 0.0:
        return 45.0 * np.pi / 180.0

    hillas_results_1 = hillas_parameters(geometry, image_edge_1)
    psi_1 = hillas_results_1["psi"].to_value(u.rad)
    hillas_results_2 = hillas_parameters(geometry, image_edge_2)
    psi_2 = hillas_results_2["psi"].to_value(u.rad)
    hillas_results_3 = hillas_parameters(geometry, image_edge_3)
    psi_3 = hillas_results_3["psi"].to_value(u.rad)

    delta_psi_23 = pow(psi_3 - psi_2, 2)
    delta_psi_12 = pow(psi_2 - psi_1, 2)
    if delta_psi_23 > 0.0:
        delta_psi_01 = delta_psi_12 * delta_psi_12 / delta_psi_23
    else:
        delta_psi_01 = delta_psi_12

    return pow(delta_psi_01, 0.5)


def plot_monoscopic_reconstruction(
    ctapipe_output,
    source,
    run_id,
    tel_id,
    event,
    star_alt,
    star_az,
    clean_image,
    mask_image,
    img_size,
    img_length,
    img_width,
    img_cen_x,
    img_cen_y,
    img_psi,
    img_psi_unc,
    img_ry_unc,
    img_frac_leakage,
):
    event_id = event.index["event_id"]
    geometry = source.subarray.tel[tel_id].camera.geometry
    focal_length = source.subarray.tel[tel_id].optics.equivalent_focal_length / u.m

    star_cam_x, star_cam_y = altaz_to_camxy(
        source,
        run_id,
        tel_id,
        star_alt * u.rad,
        star_az * u.rad,
    )

    list_chi2 = []
    list_cam_x = []
    list_cam_y = []
    pix_width = float(geometry.pixel_width[0] / u.m)
    for pix in range(0, len(clean_image)):
        cam_x = float(geometry.pix_x[pix] / u.m)
        cam_y = float(geometry.pix_y[pix] / u.m)
        nom_x, nom_y = camxy_to_nominal(source, run_id, tel_id, cam_x, cam_y)
        chi2 = source_location_chi2(
            [nom_x, nom_y],
            [img_size],
            [img_length],
            [img_width],
            [img_cen_x],
            [img_cen_y],
            [img_psi],
            [img_psi_unc],
            [img_ry_unc],
            [img_frac_leakage],
        )
        list_chi2 += [chi2]
        list_cam_x += [cam_x]
        list_cam_y += [cam_y]

    # xmax = max(geometry.pix_x) / u.m
    # xmin = min(geometry.pix_x) / u.m
    # ymax = max(geometry.pix_y) / u.m
    # ymin = min(geometry.pix_y) / u.m
    # map_size = xmax-xmin
    # map_npix = 101
    # map_shape = (map_npix,map_npix)
    # chi2_map = np.zeros(map_shape)
    # center_pix = round(float(map_npix)/2.)
    # if fit_cam_err/map_size<0.3:
    #    for pix_x in range(0,map_npix):
    #        for pix_y in range(0,map_npix):
    #            chi2_map[pix_x,pix_y] = 10.
    #            cam_x = float(pix_x-center_pix)/float(map_npix)*map_size
    #            cam_y = float(pix_y-center_pix)/float(map_npix)*map_size
    #            nom_x, nom_y = camxy_to_nominal(
    #                source, run_id, tel_id, cam_x, cam_y
    #            )
    #            chi2 = source_location_chi2(
    #                [nom_x,nom_y],
    #                [img_size],
    #                [img_length],
    #                [img_width],
    #                [img_cen_x],
    #                [img_cen_y],
    #                [img_psi],
    #                [img_psi_unc],
    #                [img_ry_unc],
    #                [img_frac_leakage],
    #            )
    #            chi2_map[pix_x,pix_y] = pow(chi2,0.5)
    #    chi2_min = np.min(chi2_map)
    #    levels = np.arange(chi2_min, chi2_min+3., 1.0)
    #    CS = ax.contour(chi2_map[:,:].T,levels,origin='lower',extent=(xmin,xmax,ymin,ymax),colors='k',linewidths=1.,linestyles='dashed')
    #    levels = np.arange(chi2_min, chi2_min+3., 2.0)
    #    CS = ax.contour(chi2_map[:,:].T,levels,origin='lower',extent=(xmin,xmax,ymin,ymax),colors='k',linewidths=1.,linestyles='solid')

    values = []
    values += [event.dl1.tel[tel_id].image]
    values += [clean_image]

    titles = []
    titles += ["noisy image"]
    titles += [f"cleaned image (size = {int(img_size)}, length = {img_length:0.2f}, width = {img_width:0.2f})"]

    # fig, axs = plt.subplots(1, 2, constrained_layout=True, sharex=True)
    fig, axs = plt.subplots(1, 2, figsize=(2.0 * 8.6, 6.4))
    for ax, trials, title in zip(axs, values, titles):
        display = CameraDisplay(geometry, ax=ax)
        display.image = trials
        display.cmap = "Reds"
        display.add_colorbar(ax=ax)

        if not title == "noisy image":
            for pix in range(0, len(list_chi2)):
                chi2 = list_chi2[pix]
                cam_x = list_cam_x[pix]
                cam_y = list_cam_y[pix]
                chi = pow(chi2, 0.5)
                if chi < 1.0:
                    mycircle = plt.Circle(
                        (cam_x, cam_y),
                        0.5 * pix_width,
                        fill=True,
                        color="blue",
                        alpha=0.1,
                    )
                    ax.add_patch(mycircle)
                if chi < 2.0:
                    mycircle = plt.Circle(
                        (cam_x, cam_y),
                        0.5 * pix_width,
                        fill=True,
                        color="blue",
                        alpha=0.1,
                    )
                    ax.add_patch(mycircle)

        display.highlight_pixels(
            mask_image, color="xkcd:green", linewidth=0.5, alpha=1.0
        )

        ax.scatter(star_cam_x, star_cam_y, s=90, facecolors="none", c="r", marker="+")
        ax.set_title(title)
    fig.savefig(
        f"{ctapipe_output}/output_plots/run{run_id}_evt{event_id}_tel{tel_id}_denoised_image.png",
        bbox_inches="tight",
    )
    del fig
    del axs
    plt.close()


def loop_all_events(
    ana_tag,
    training_sample_path,
    ctapipe_output,
    list_telescope_type,
    select_evt=None,
    save_output=True,
    make_plot = False,
):

    if not select_evt == None:
        make_plot = True

    print(f"loading file: {training_sample_path}")
    source = SimTelEventSource(training_sample_path, focal_length_choice="EQUIVALENT")

    list_tel_id = []
    for tel_idx in range(0, source.subarray.n_tels):
        tel_id = source.subarray.tel_ids[tel_idx]
        telescope_type = str(source.subarray.tel[tel_id])
        if telescope_type not in list_telescope_type:
            continue
        list_tel_id += [tel_id]
    # new_subarray = source.subarray.select_subarray(list_tel_id)
    source = SimTelEventSource(
        training_sample_path, focal_length_choice="EQUIVALENT", allowed_tels=list_tel_id
    )

    # Explore the instrument description
    subarray_table = source.subarray.to_table()
    nlines = len(subarray_table) + 10
    subarray_table.pprint(nlines)
    print(source.subarray.to_table())

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

    mono_analysis_result = []
    default_array_analysis_result = []
    new_array_analysis_result = []

    n_truth_events = 0

    for event in source:
        event_id = event.index["event_id"]

        if not select_evt == None:
            select_event_id = select_evt[1]
            if event_id != select_event_id:
                continue

        truth_alt = float(event.simulation.shower.alt / u.rad)
        truth_az = float(event.simulation.shower.az / u.rad)
        truth_energy = float(event.simulation.shower.energy / u.TeV)
        # if truth_energy<5.:
        #    continue

        calib(event)  # fills in r1, dl0, and dl1
        image_processor(event)
        shower_processor(event)

        n_truth_events += 1

        default_reco_result = event.dl2.stereo.geometry["HillasReconstructor"]
        average_intensity = default_reco_result.average_intensity
        default_n_tels = len(default_reco_result.telescopes)
        default_alt = default_reco_result.alt.to(u.rad).value
        default_alt_uncert = default_reco_result.alt_uncert.to(u.rad).value
        default_az = default_reco_result.az.to(u.rad).value
        default_az_uncert = default_reco_result.az_uncert.to(u.rad).value
        default_location_err_deg = (
            angular_separation(
                truth_az * u.rad,
                truth_alt * u.rad,
                default_az * u.rad,
                default_alt * u.rad,
            )
            .to(u.deg)
            .value
        )
        default_location_unc_deg = (
            pow(
                default_alt_uncert * default_alt_uncert
                + default_az_uncert * default_az_uncert,
                0.5,
            )
            * 180.0
            / np.pi
        )

        default_nom_x, default_nom_y = altaz_to_nominal(
            source,
            run_id,
            default_alt * u.rad,
            default_az * u.rad,
        )

        list_tel_id = []
        list_clean_image = []
        list_mask = []
        list_img_islands = []
        list_img_size = []
        list_img_length = []
        list_img_width = []
        list_img_cen_x = []
        list_img_cen_y = []
        list_img_psi = []
        list_img_psi_unc = []
        list_img_ry_unc = []
        list_img_frac_leakage = []
        list_truth_psi = []

        for tel_idx in range(0, len(list(event.dl0.tel.keys()))):
            tel_id = list(event.dl0.tel.keys())[tel_idx]
            telescope_type = str(source.subarray.tel[tel_id])
            if telescope_type not in list_telescope_type:
                continue

            cam_type = telescope_type.split("_")[2]

            geometry = source.subarray.tel[tel_id].camera.geometry
            focal_length = float(
                source.subarray.tel[tel_id].optics.equivalent_focal_length / u.m
            )

            init_clean_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
            init_image_mask = cleaning_image(geometry,event.dl1.tel[tel_id].image,init_clean_image_1d)
            init_image_size = np.sum(init_clean_image_1d)

            clean_image_1d = np.zeros_like(event.dl1.tel[tel_id].image)
            #image_mask = denoising_image(
            #    denoiser_model_pkl[cam_type],
            #    geometry,
            #    event.dl1.tel[tel_id].image,
            #    clean_image_1d,
            #)
            image_mask, interm_Ys = univ_inv_sol(
                denoiser_model_pkl,
                geometry,
                event.dl1.tel[tel_id].image,
                clean_image_1d,
                h0 = 0.2,
                freq = 10
            )

            image_size = np.sum(clean_image_1d)
            if image_size == 0.0:
                continue

            border_pixels = geometry.get_border_pixel_mask(1)
            border_mask = border_pixels & image_mask
            leakage_intensity = np.sum(clean_image_1d[border_mask])
            n_pe_cleaning = np.sum(clean_image_1d)
            frac_leakage_intensity = 1.0
            if n_pe_cleaning > 0.0:
                frac_leakage_intensity = leakage_intensity / n_pe_cleaning

            hillas_results = hillas_parameters(geometry, clean_image_1d)
            # print (f"hillas_results = {hillas_results}")
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

            if frac_leakage_intensity > 0.0:
                edge_psi_uncertainty = psi_uncertainty_truncated_image(
                    geometry, clean_image_1d
                )
                print(
                    f"psi_uncertainty = {psi_uncertainty*180./np.pi:0.3f} deg, edge_psi_uncertainty = {edge_psi_uncertainty*180./np.pi:0.3f} deg"
                )
                psi_uncertainty = pow(
                    pow(psi_uncertainty, 2) + pow(edge_psi_uncertainty, 2), 0.5
                )

            pix_width = float(geometry.pixel_width[0] / u.m)
            # psi_uncertainty = max(psi_uncertainty,width/length)

            truth_cam_x, truth_cam_y = altaz_to_camxy(
                source,
                run_id,
                tel_id,
                truth_alt * u.rad,
                truth_az * u.rad,
            )
            truth_psi = np.arctan((cog_y - truth_cam_y) / (cog_x - truth_cam_x))
            psi_error = abs(truth_psi - psi)
            if psi_error > 2.0 * np.pi:
                psi_error = psi_error - 2.0 * np.pi
            if psi_error > np.pi:
                psi_error = 2.0 * np.pi - psi_error
            if psi_error > 0.5 * np.pi:
                psi_error = np.pi - psi_error
            mono_analysis_result += [
                [
                    psi_error,
                    psi_uncertainty,
                    run_id,
                    event_id,
                    tel_id,
                    image_size,
                    frac_leakage_intensity,
                ]
            ]

            cog_nom_x, cog_nom_y = camxy_to_nominal(
                source, run_id, tel_id, cog_x, cog_y
            )

            if init_image_size < 20.0:
                continue
            #if psi_uncertainty > 30.0 * np.pi / 180.0:
            #    continue
            #if intensity < 10.0:
            #    continue
            if width == 0.0:
                continue
            if length / width < 1.5:
                continue

            n_islands = number_of_islands(geometry, image_mask)

            list_tel_id += [tel_id]
            list_clean_image += [clean_image_1d]
            list_mask += [image_mask]
            list_img_size += [intensity]
            list_img_islands += [n_islands[0]]
            list_img_length += [length]
            list_img_width += [width]
            list_img_cen_x += [cog_nom_x]
            list_img_cen_y += [cog_nom_y]
            list_img_psi += [psi]
            list_img_psi_unc += [psi_uncertainty]
            #list_img_ry_unc += [
            #    max(0.5*pix_width, transverse_cog_uncertainty) / focal_length
            #]
            list_img_ry_unc += [transverse_cog_uncertainty/focal_length]
            list_img_frac_leakage += [frac_leakage_intensity]
            list_truth_psi += [truth_psi]

        if len(list_img_size) < 2:
            continue

        brute_fov = 4.0 / 180.0 * np.pi
        brute_ranges = ((-brute_fov, brute_fov), (-brute_fov, brute_fov))
        grid_points = 40
        solution = brute(
            source_location_chi2,
            brute_ranges,
            args=(
                list_img_size,
                list_img_length,
                list_img_width,
                list_img_cen_x,
                list_img_cen_y,
                list_img_psi,
                list_img_psi_unc,
                list_img_ry_unc,
                list_img_frac_leakage,
            ),
            Ns=grid_points,
        )
        fit_params = solution
        fit_nom_x = fit_params[0]
        fit_nom_y = fit_params[1]

        init_params = [fit_nom_x, fit_nom_y]
        # init_params = [default_nom_x,default_nom_y]
        angular_step = 0.001 * np.pi / 180.0
        stepsize = [angular_step, angular_step]
        ftol = 0.00001
        solution = minimize(
            source_location_chi2,
            x0=init_params,
            args=(
                list_img_size,
                list_img_length,
                list_img_width,
                list_img_cen_x,
                list_img_cen_y,
                list_img_psi,
                list_img_psi_unc,
                list_img_ry_unc,
                list_img_frac_leakage,
            ),
            method="L-BFGS-B",
            jac=None,
            options={"eps": stepsize, "ftol": ftol},
        )
        fit_params = solution["x"]
        fit_nom_x = fit_params[0]
        fit_nom_y = fit_params[1]

        fit_nom_unc = compute_location_uncertainty(
            fit_params,
            list_img_size,
            list_img_length,
            list_img_width,
            list_img_cen_x,
            list_img_cen_y,
            list_img_psi,
            list_img_psi_unc,
            list_img_ry_unc,
            list_img_frac_leakage,
        )

        n_tels = len(list_img_psi)
        min_dist_to_img = 1e10
        for img in range(0, len(list_img_length)):
            length = list_img_length[img]
            dist_to_img_x = fit_nom_x - list_img_cen_x[img]
            dist_to_img_y = fit_nom_y - list_img_cen_y[img]
            dist_to_img = (
                pow(dist_to_img_x * dist_to_img_x + dist_to_img_y * dist_to_img_y, 0.5)
                / length
            )
            if min_dist_to_img > dist_to_img:
                min_dist_to_img = dist_to_img

        # location_err_deg = pow(fit_nom_x*fit_nom_x+fit_nom_y*fit_nom_y,0.5)*180./np.pi
        fit_alt = 0.0
        fit_az = 0.0
        if abs(fit_nom_x) < np.pi / 2.0 and abs(fit_nom_y) < np.pi / 2.0:
            fit_alt, fit_az = nominal_to_altaz(
                source,
                run_id,
                fit_nom_x * u.rad,
                fit_nom_y * u.rad,
            )
        location_err_deg = (
            angular_separation(
                truth_az * u.rad, truth_alt * u.rad, fit_az * u.rad, fit_alt * u.rad
            )
            .to(u.deg)
            .value
        )
        location_unc_deg = fit_nom_unc * 180.0 / np.pi

        print(
            "======================================================================================"
        )
        print(f"run_id = {run_id}")
        print(f"event_id = {event_id}")
        print(
            f"truth_alt = {truth_alt}, truth_az = {truth_az}, truth_energy = {truth_energy:0.3f} TeV"
        )
        print(f"default_n_tels = {default_n_tels}")
        print(f"n_tels = {n_tels}")
        print(f"default_location_err_deg = {default_location_err_deg:0.3f} deg")
        print(f"location_err_deg         = {location_err_deg:0.3f} deg")
        print(f"default_location_unc_deg = {default_location_unc_deg:0.3f} deg")
        print(f"location_unc_deg         = {location_unc_deg:0.3f} deg")
        print(f"min_dist_to_img = {min_dist_to_img:0.3f} (length)")
        print(f"diff = {(default_location_err_deg-location_err_deg)/location_unc_deg}")
        # if (default_location_err_deg-location_err_deg)/location_unc_deg < -2.: exit()

        is_good_result = True
        # if min_dist_to_img<0.1:
        #    is_good_result = False
        # if min_dist_to_img>100.:
        #    is_good_result = False

        # if location_err_deg/location_unc_deg>5.:
        #    if is_good_result:
        #        exit()

        if not np.isnan(default_location_unc_deg) and not np.isnan(
            default_location_err_deg
        ):
            default_array_analysis_result += [
                [default_location_err_deg, default_location_unc_deg, truth_energy]
            ]
        if is_good_result:
            new_array_analysis_result += [
                [location_err_deg, location_unc_deg, truth_energy]
            ]

        if make_plot:
            for tel in range(0, len(list_tel_id)):
                src_chi2 = source_location_chi2(
                    [truth_cam_x, truth_cam_y],
                    [list_img_size[tel]],
                    [list_img_length[tel]],
                    [list_img_width[tel]],
                    [list_img_cen_x[tel]],
                    [list_img_cen_y[tel]],
                    [list_img_psi[tel]],
                    [list_img_psi_unc[tel]],
                    [list_img_ry_unc[tel]],
                    [list_img_frac_leakage[tel]],
                )
                src_chi = pow(src_chi2, 0.5)
                #if list_img_size[tel] < 2000.0:
                #    continue
                #if src_chi < 3.0:
                #    continue
                #if list_img_islands[tel]<2:
                #    continue
                if not default_location_err_deg<0.3*location_err_deg:
                    continue
                print("making a plot...")

                plot_monoscopic_reconstruction(
                    ctapipe_output,
                    source,
                    run_id,
                    list_tel_id[tel],
                    event,
                    truth_alt,
                    truth_az,
                    list_clean_image[tel],
                    list_mask[tel],
                    list_img_size[tel],
                    list_img_length[tel],
                    list_img_width[tel],
                    list_img_cen_x[tel],
                    list_img_cen_y[tel],
                    list_img_psi[tel],
                    list_img_psi_unc[tel],
                    list_img_ry_unc[tel],
                    list_img_frac_leakage[tel],
                )
            #exit()

        if not select_evt == None:
            exit()

        if not make_plot:
            output_filename = (
                f"{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}_mono.pkl"
            )
            print(f"writing file to {output_filename}")
            with open(output_filename, "wb") as file:
                pickle.dump(
                    mono_analysis_result,
                    file,
                )

            output_filename = (
                f"{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}_array_default.pkl"
            )
            print(f"writing file to {output_filename}")
            with open(output_filename, "wb") as file:
                pickle.dump(
                    default_array_analysis_result,
                    file,
                )

            output_filename = (
                f"{ctapipe_output}/output_analysis/{ana_tag}_run{run_id}_array_new.pkl"
            )
            print(f"writing file to {output_filename}")
            with open(output_filename, "wb") as file:
                pickle.dump(
                    new_array_analysis_result,
                    file,
                )


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
            select_evt=select_evt,
            save_output=True,
            make_plot=make_plot,
        )
