
import os
import subprocess
import pickle
from ctapipe import utils
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter

from ctapipe.reco.veritas_utilities import run_save_training_matrix

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")

subprocess.call(f'rm {ctapipe_output}/output_plots/*.png', shell=True)

#telescope_type = 'MST_SCT_SCTCam'
#telescope_type = 'MST_MST_NectarCam'
telescope_type = 'MST_MST_FlashCam'
#telescope_type = 'SST_1M_DigiCam'
#telescope_type = 'SST_ASTRI_ASTRICam'
#telescope_type = 'SST_GCT_CHEC'
#telescope_type = 'LST_LST_LSTCam'

#sim_files = 'sct_onaxis_train.txt'
#sim_files = 'sct_onaxis_test.txt'
#sim_files = 'sct_diffuse_all.txt'
sim_files = 'mst_onaxis_train.txt'
#sim_files = 'mst_onaxis_test.txt'
#sim_files = 'mst_diffuse_all.txt'

big_truth_matrix = []
big_moment_matrix = []
big_image_matrix = []
big_time_matrix = []
big_movie_matrix = []
with open(f'{ctapipe_input}/{sim_files}', 'r') as file:
    for line in file:

        #training_sample_path = get_dataset_path(line.strip('\n'))
        #source = SimTelEventSource(training_sample_path, focal_length_choice='EQUIVALENT')
        #subarray = source.subarray
        #ob_keys = source.observation_blocks.keys()
        #run_id = list(ob_keys)[0]

        training_sample_path = line.strip('\n')
        run_id = training_sample_path.split("_")[3].strip("run")
        output_filename = f'{ctapipe_output}/output_samples/training_sample_run{run_id}_{telescope_type}.pkl'
        print (f'loading pickle trainging sample data: {output_filename}')
        if not os.path.exists(output_filename):
            print (f'file does not exist.')
            continue
        training_sample = pickle.load(open(output_filename, "rb"))

        big_truth_matrix += training_sample[0]
        big_moment_matrix += training_sample[1]
        big_image_matrix += training_sample[2]
        big_time_matrix += training_sample[3]
        big_movie_matrix += training_sample[4]

output_filename = f'{ctapipe_output}/output_machines/big_truth_matrix_{telescope_type}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(big_truth_matrix, file)

output_filename = f'{ctapipe_output}/output_machines/big_moment_matrix_{telescope_type}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(big_moment_matrix, file)

output_filename = f'{ctapipe_output}/output_machines/big_image_matrix_{telescope_type}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(big_image_matrix, file)

output_filename = f'{ctapipe_output}/output_machines/big_time_matrix_{telescope_type}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(big_time_matrix, file)

output_filename = f'{ctapipe_output}/output_machines/big_movie_matrix_{telescope_type}.pkl'
with open(output_filename,"wb") as file:
    pickle.dump(big_movie_matrix, file)

