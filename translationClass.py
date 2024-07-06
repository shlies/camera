import numpy as np
import cv2

class Translation:
   empCount = 0
   fx=545.103
   fy=545.103
   cx=321.608
   cy=243.754
   k1=0.0887278
   k2=-0.0544192
   k3=-0.297869
   k4=0
   k5=0
   k6=0
   p1=-0.000272924
   p2=0.00631082
   def __init__(self, name, salary):
      self.name = name
      self.salary = salary
      Employee.empCount += 1

   def read_depth_image(file_path, width=640, height=480):
    """
    Read a depth image from a raw file captured by Intel RealSense D435i.

    Args:
    file_path (str): Path to the raw depth image file.
    width (int): Width of the depth image. Default is 640 pixels.
    height (int): Height of the depth image. Default is 480 pixels.

    Returns:
    numpy.ndarray: The depth image.
    """
    # Calculate the total number of bytes in the file
    num_bytes = width * height * 2  # 2 bytes (16 bits) per pixel

    # Read the file as a 1D array of 16-bit unsigned integers
    with open(file_path, 'rb') as file:
        depth_data = np.fromfile(file, dtype=np.uint16, count=num_bytes)

    # Reshape the 1D array into a 2D array of the specified dimensions
    depth_image = depth_data.reshape((height, width))

    return depth_image

def image_to_world_vectorized(coords, depth_map, M_inv, rot_mat_inv, T, rgb_distortion_matrix):
    u, v = coords[..., 0], coords[..., 1]
    z = depth_map[v, u]  # Extract depth values for each coordinate
    i_vector = np.stack([u, v, np.ones_like(u)], axis=-1)  # Convert to homogeneous coordinates

    # Apply RGB distortion matrix
    i_vector_distorted = np.matmul(rgb_distortion_matrix, i_vector.T).T[..., :2]

    i_vector_transformed = i_vector_distorted @ M_inv.T
    left_side = rot_mat_inv @ i_vector_transformed[..., np.newaxis]  # Shape: [height, width, 3, 1]

    # Ensure 'right_side' is broadcastable with 'left_side'
    right_side = rot_mat_inv @ T.reshape(-1, 1)  # Shape: [3, 1]
    right_side = right_side.reshape(1, 1, 3, 1)  # Reshaped to [1, 1, 3, 1]

    # Reshape 's' for broadcasting
    s = z[..., np.newaxis, np.newaxis]  # Shape becomes [height, width, 1, 1]
    w = s * left_side - right_side
    w = w.squeeze(-1)  # Remove the last dimension, shape: [height, width, 3]

    return w

def transform_entire_image_with_depth(depth_map, M, r_vec, T, rgb_distortion_matrix):
    M_inv = np.linalg.inv(M)
    rot_mat, _ = cv2.Rodrigues(r_vec)
    rot_mat_inv = np.linalg.inv(rot_mat)

    height, width = depth_map.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.stack([u, v], axis=-1)

    world_coords_3d = image_to_world_vectorized(coords, depth_map, M_inv, rot_mat_inv, T, rgb_distortion_matrix)
    return world_coords_3d

def show_z_channel(world_coordinates):
    # Extract the Z channel
    z_channel = world_coordinates[:, :, 2]

    # Normalize the Z channel for better visualization
    z_norm = cv2.normalize(z_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Display the Z channel
    cv2.imshow('Z Channel', z_norm)
    cv2.waitKey(0)
    cv2.destroyAllWindows()