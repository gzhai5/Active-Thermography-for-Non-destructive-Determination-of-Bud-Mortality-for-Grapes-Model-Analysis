import numpy as np


def rotate_video(video_array, angle):
    """
    Rotate a video array by a specified angle.

    Args:
        video_array (numpy.ndarray): The input video array of shape (frames, height, width, channels).
        angle (float): The angle by which to rotate the video in degrees.

    Returns:
        numpy.ndarray: The rotated video array.
    """
    from scipy.ndimage import rotate

    # Rotate each frame in the video array
    rotated_video = np.array([rotate(frame, angle, reshape=False) for frame in video_array])
    
    return rotated_video