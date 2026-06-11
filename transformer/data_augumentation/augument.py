import random
from data_augumentation.flip import flip_video


def augumentate(video_dict_list):
    """
    Augment the minority class by duplicating samples from the majority class.
    assume majority class is 1 and minority class is 0. majority class is in the range between 2x and 3x of minority class.
    firstly count the number of majority and minority classes.
    secondly, augment the minority class by 2x.
    thirdly, randomly select the current difference between majority and current minority class, and augment the minority class to match the majority class size.
    finally, return the augmented + original minority class samples.

    Args:
        video_dict_list (list): List of dictionaries containing video data and labels.
            Each dictionary should have the following keys:
                - 'video': The video data (numpy array).
                - 'label': The label of the video.

    Returns:
        list: video_dict_list with augmented samples.
    """
    # Separate the videos into majority and minority classes
    majority_videos = [video for video in video_dict_list if video['labels'] == 1]
    minority_videos = [video for video in video_dict_list if video['labels'] == 0]
    num_majority = len(majority_videos)
    num_minority = len(minority_videos)
    print(f"Number of majority videos: {num_majority} | Number of minority videos: {num_minority}")
    if num_majority > num_minority * 3:
        raise ValueError("The number of majority videos is too high compared to minority videos.")

    # Augment the minority class by 2x
    augmented_videos = []
    for minority_video in minority_videos:
        augmented_video = flip_video(minority_video['video'], horizontal=True)
        augmented_videos.append({'video': augmented_video, 'labels': 0})

    # Augment the minority class to match the majority class size
    num_wanted = num_majority - len(augmented_videos) - len(minority_videos)
    random.seed(42)
    sampled_videos = random.choices(minority_videos, k=num_wanted)
    for video in sampled_videos:
        flipped = flip_video(video['video'], horizontal=False)
        augmented_videos.append({'video': flipped, 'labels': 0})
    
    return minority_videos + augmented_videos + majority_videos