def flip_video(video_array, horizontal=True):
    """
    Flip a video array either horizontally or vertically.

    Args:
        video_array (numpy.ndarray): The input video array of shape (frames, height, width, channels).
        horizontal (bool): If True, flip horizontally. If False, flip vertically.

    Returns:
        numpy.ndarray: The flipped video array.
    """
    if horizontal:
        return video_array[:, :, ::-1, :]
    else:
        return video_array[:, ::-1, :, :]