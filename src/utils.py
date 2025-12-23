"""
General utility functions for GEOSSM.
"""

import numpy as np
import datetime

# % Compute the block diagonal 3D

def block_diag_3D(*arrs):
    """
    Create a 3D block diagonal matrix from given 3D matrices where the first
    two dimensions can vary but the last dimension is the same.
    Each input array should be of shape (n_i, m_i, p), where p is constant.

    Parameters:
    *arrs : 3D matrices to be stacked in block diagonal manner.

    Returns:
    np.ndarray : 3D block diagonal matrix.
    """
    # Determine the total shape for the first two dimensions
    total_shape_0 = sum(arr.shape[0] for arr in arrs)
    total_shape_1 = sum(arr.shape[1] for arr in arrs)
    # the last dimension should be the same for all arrays
    total_shape_2 = arrs[0].shape[2]

    # Initialize the block diagonal matrix with zeros
    block_diag_matrix = np.zeros(
        (total_shape_0, total_shape_1, total_shape_2))

    # Current start index for the first two dimensions
    current_index_0 = 0
    current_index_1 = 0

    for arr in arrs:
        shape_0, shape_1, shape_2 = arr.shape
        block_diag_matrix[current_index_0:current_index_0+shape_0,
                          current_index_1:current_index_1+shape_1,
                          :] = arr
        current_index_0 += shape_0
        current_index_1 += shape_1

    return block_diag_matrix

# % Write the model information into a file

def write(filename, grid_obs, ssm_model, mode='a'):
    # write the class into a file

    # Get current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    s = f"Current Time: {current_time} \n"

    s += "OBSERVATION GRID \n"
    s += ''.join([gr.__str__() for gr in grid_obs])
    s += "\n"

    s += "SSM REPRESENTATION \n"
    s += ssm_model.__str__()
    s += "\n"

    s += "MODEL PARAMITER ESTIMATE"
    s += "\n"

    s += "HARDWARE \n"
    s += getHardware()

    with open(filename, mode) as f:
        f.write(s)

    return filename, mode

# % Get Hardware infromation

def getHardware():

    # Get CPU info
    cpu_name = platform.processor()
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_freq = psutil.cpu_freq().current

    # Get architecture
    architecture = platform.architecture()[0]

    # Get memory info
    virtual_memory = psutil.virtual_memory()
    total_memory = virtual_memory.total / (1024 ** 3)  # Convert to GB
    available_memory = virtual_memory.available / \
        (1024 ** 3)  # Convert to GB

    # Get system info
    system = platform.system()
    release = platform.release()
    version = platform.version()
    machine = platform.machine()
    node = platform.node()

    # Compile all information into a string
    info = (f"System: {system} {release} {version}\n"
            f"Node Name: {node}\n"
            f"Machine: {machine}\n"
            f"Architecture: {architecture}\n"
            f"CPU: {cpu_name}\n"
            f"Physical CPUs: {cpu_count}\n"
            f"Logical CPUs: {cpu_count_logical}\n"
            f"Current CPU Frequency: {cpu_freq:.2f} MHz\n"
            f"Total Memory: {total_memory:.2f} GB\n"
            f"Available Memory: {available_memory:.2f} GB\n")
    return info
