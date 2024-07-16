
import os
import subprocess
import pickle
from ctapipe import utils
from ctapipe.utils.datasets import get_dataset_path
from ctapipe.io import EventSource, SimTelEventSource, HDF5TableWriter

from ctapipe.reco.veritas_utilities import BigMatrixSVD
from ctapipe.reco.veritas_utilities import MakeFastConversionImage

ctapipe_output = os.environ.get("CTAPIPE_OUTPUT_PATH")
ctapipe_input = os.environ.get("CTAPIPE_SVC_PATH")

subprocess.call(f'rm {ctapipe_output}/output_plots/*.png', shell=True)

telescope_type = 'MST_SCT_SCTCam'
#telescope_type = 'MST_MST_NectarCam'
#telescope_type = 'MST_MST_FlashCam'
#telescope_type = 'SST_1M_DigiCam'
#telescope_type = 'SST_ASTRI_ASTRICam'
#telescope_type = 'SST_GCT_CHEC'
#telescope_type = 'LST_LST_LSTCam'


matrix_rank = 20
#matrix_rank = 40

output_filename = f'{ctapipe_output}/output_machines/big_truth_matrix_{telescope_type}.pkl'
big_truth_matrix = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/big_moment_matrix_{telescope_type}.pkl'
big_moment_matrix = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/big_image_matrix_{telescope_type}.pkl'
big_image_matrix = pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/big_time_matrix_{telescope_type}.pkl'
big_time_matrix= pickle.load(open(output_filename, "rb"))

output_filename = f'{ctapipe_output}/output_machines/big_movie_matrix_{telescope_type}.pkl'
big_movie_matrix = pickle.load(open(output_filename, "rb"))

print ('Compute movie matrix SVD...')
movie_eigenvectors = BigMatrixSVD(ctapipe_output,telescope_type,big_movie_matrix,big_moment_matrix,big_truth_matrix,2*matrix_rank,'movie')
print ('Compute image matrix SVD...')
image_eigenvectors = BigMatrixSVD(ctapipe_output,telescope_type,big_image_matrix,big_moment_matrix,big_truth_matrix,matrix_rank,'image')
print ('Compute time matrix SVD...')
time_eigenvectors = BigMatrixSVD(ctapipe_output,telescope_type,big_time_matrix,big_moment_matrix,big_truth_matrix,2*matrix_rank,'time')

MakeFastConversionImage(ctapipe_output,telescope_type,image_eigenvectors,big_image_matrix,time_eigenvectors,big_time_matrix,big_moment_matrix,big_truth_matrix,'image')

