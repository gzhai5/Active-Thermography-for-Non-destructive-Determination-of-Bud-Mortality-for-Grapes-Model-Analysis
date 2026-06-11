import re

abnormal_samples_ries = [
    r"D:\Bud Thermography Data\round_5\Ries5.0_B04_N01.npy",
    r"D:\Bud Thermography Data\round_5\Ries5.0_B18_N01.npy",
    r"D:\Bud Thermography Data\round_5\Ries5.0_B30_N07.npy",
    r"D:\Bud Thermography Data\round_5\Ries5.0_B06_N01.npy", #falling
    r"D:\Bud Thermography Data\round_5\Ries5.0_B08_N07.npy", #falling
    r"D:\Bud Thermography Data\round_5\Ries5.0_B24_N05.npy",
    r"D:\Bud Thermography Data\round_6\Ries6.1F_B34_N05.npy",
    r"D:\Bud Thermography Data\round_5\Ries5.1F_B36_N01.npy",
    r"D:\Bud Thermography Data\round_6\Ries6.0_B28_N03.npy", #falling
    r"D:\Bud Thermography Data\round_5\Ries5.1F_B42_N05.npy",
    r"D:\Bud Thermography Data\round_6\Ries6.1F_B45_N08.npy",
    r"D:\Bud Thermography Data\round_6\Ries6.0_B05_N02.npy", #falling
    r"D:\Bud Thermography Data\round_6\Ries6.0_B10_N06.npy",
    r"D:\Bud Thermography Data\round_6\Ries6.1F_B32_N01.npy", #falling
    r"D:\Bud Thermography Data\round_6\Ries6.0_B03_N07.npy", #falling
    r"D:\Bud Thermography Data\round_5\Ries5.0_B22_N06.npy",
    r"D:\Bud Thermography Data\round_6\Ries6.1F_B37_N07.npy",
    r"D:\Bud Thermography Data\round_5\Ries5.0_B02_N06.npy", #falling
    r"D:\Bud Thermography Data\round_5\Ries5.0_B23_N05.npy", #falling
    r"D:\Bud Thermography Data\round_5\Ries5.0_B14_N06.npy", #falling
    r"D:\Bud Thermography Data\round_5\Ries5.0_B05_N07.npy",
    r"D:\Bud Thermography Data\round_6\Ries6.1F_B37_N01.npy",
    r"D:\Bud Thermography Data\round_6\Ries6.0_B02_N05.npy", #falling
    r"D:\Bud Thermography Data\round_6\Ries6.0_B19_N01.npy", #falling
    r"D:\Bud Thermography Data\round_6\Ries6.0_B13_N05.npy",
    r"D:\Bud Thermography Data\round_5\RIES5.1F_B32_N05.npy", #falling
    r"D:\Bud Thermography Data\round_6\RIES6.1F_B42_N05.npy",
    r"D:\Bud Thermography Data\round_5\RIES5.0_B07_N08.npy",
    r"D:\Bud Thermography Data\round_5\RIES5.0_B04_N09.npy", #falling
    r"D:\Bud Thermography Data\round_5\RIES5.0_B21_N06.npy", #falling
    r"D:\Bud Thermography Data\round_5\RIES5.0_B27_N02.npy",
    r"D:\Bud Thermography Data\round_6\RIES6.1F_B33_N04.npy",
    r"D:\Bud Thermography Data\round_6\RIES6.1F_B33_N08.npy",
    r"D:\Bud Thermography Data\round_5\RIES5.0_B28_N08.npy",
    r"D:\Bud Thermography Data\round_6\RIES6.0_B07_N09.npy",
    r"D:\Bud Thermography Data\round_5\RIES5.1F_B43_N02.npy",
    r"D:\Bud Thermography Data\round_5\RIES5.1F_B34_N03.npy",
    r"D:\Bud Thermography Data\round_5\RIES5.1F_B40_N08.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.0_B27_N02.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.0_B08_N06.npy",
    r"D:\Bud Thermography Data\round_3\RIES3.0_B04_N08.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.0_B18_N07.npy", #falling
    r"D:\Bud Thermography Data\round_2\Ries_2.0_B11_N03.npy", #falling
    r"D:\Bud Thermography Data\round_3\RIES3.0_B06_N07.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.0_B20_N01.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B40_N05.npy", #falling
    r"D:\Bud Thermography Data\round_2\Ries_2.0_B17_N07.npy", #falling
    r"D:\Bud Thermography Data\round_4\Ries4.0_B07_N05.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.1F_B42_N03.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.0_B02_N05.npy",
    r"D:\Bud Thermography Data\round_3\Ries3.1F_B45_N06.npy", #falling
    r"D:\Bud Thermography Data\round_4\Ries4.1F_B31_N02.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.0_B04_N03.npy", #falling
    r"D:\Bud Thermography Data\round_4\Ries4.0_B22_N05.npy", #falling
    r"D:\Bud Thermography Data\round_4\Ries4.0_B19_N02.npy", #falling
    r"D:\Bud Thermography Data\round_3\RIES3.0_B09_N09.npy", #falling
    r"D:\Bud Thermography Data\round_4\Ries4.1F_B39_N08.npy", #falling
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B35_N02.npy",
    r"D:\Bud Thermography Data\round_3\RIES3.0_B01_N07.npy", #falling
    r"D:\Bud Thermography Data\round_3\Ries3.1F_B32_N03.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.0_B11_N03.npy",
    r"D:\Bud Thermography Data\round_3\Ries3.1F_B43_N06.npy", #falling
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B30_N03.npy",
    r"D:\Bud Thermography Data\round_3\Ries3.0_B23_N08.npy", #falling
    r"D:\Bud Thermography Data\round_2\Ries_2.0_B10_N09.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.0_B25_N09.npy", #falling
    r"D:\Bud Thermography Data\round_3\RIES3.0_B08_N07.npy", #falling
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B31_N01.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B30_N09.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B31_N07.npy",
    r"D:\Bud Thermography Data\round_3\Ries3.1F_B37_N08.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B41_N04.npy", #falling
    r"D:\Bud Thermography Data\round_4\Ries4.1F_B33_N01.npy", #falling
    r"D:\Bud Thermography Data\round_3\Ries3.1F_B36_N08.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.0_B01_N07.npy",
    r"D:\Bud Thermography Data\round_3\Ries3.1F_B33_N02.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.0_B23_N08.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B39_N03.npy",
    r"D:\Bud Thermography Data\round_3\RIES3.0_B17_N07.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B33_N02.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.1F_B39_N01.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.0_B17_N06.npy",
    r"D:\Bud Thermography Data\round_3\Ries3.1F_B39_N09.npy", #falling
    r"D:\Bud Thermography Data\round_4\Ries4.1F_B44_N09.npy", #falling
    r"D:\Bud Thermography Data\round_3\RIES3.0_B16_N06.npy", #falling
    r"D:\Bud Thermography Data\round_3\RIES3.0_B11_N06.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.0_B29_N05.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B34_N04.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.0_B05_N07.npy", #falling
    r"D:\Bud Thermography Data\round_2\Ries_2.0_B24_N08.npy", #falling
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B44_N03.npy",
    r"D:\Bud Thermography Data\round_3\RIES3.0_B30_N01.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.1F_B45_N08.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.1F_B32_N01.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.0_B01_N04.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.1F_B43_N02.npy",
    r"D:\Bud Thermography Data\round_3\RIES3.0_B03_N02.npy",
    r"D:\Bud Thermography Data\round_3\Ries3.1F_B33_N08.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.0_B03_N07.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B30_N01.npy", #falling
    r"D:\Bud Thermography Data\round_4\Ries4.0_B04_N09.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.0_B06_N04.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B31_N06.npy", #falling
    r"D:\Bud Thermography Data\round_3\RIES3.0_B16_N09.npy", #falling
    r"D:\Bud Thermography Data\round_3\RIES3.0_B04_N02.npy", #falling
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B30_N07.npy", #falling
    r"D:\Bud Thermography Data\round_4\Ries4.1F_B38_N03.npy",
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B41_N01.npy", #falling
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B43_N05.npy", #falling
    r"D:\Bud Thermography Data\round_3\Ries3.1F_B35_N02.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.0_B02_N08.npy",
    r"D:\Bud Thermography Data\round_4\Ries4.1F_B40_N06.npy",
    r"D:\Bud Thermography Data\round_3\Ries3.1F_B38_N08.npy", #falling
    r"D:\Bud Thermography Data\round_2\Ries_2.1_F_B31_N02.npy", #falling
    r"D:\Bud Thermography Data\round_4\Ries4.1F_B33_N07.npy", #falling
    r"D:\Bud Thermography Data\round_3\RIES3.0_B21_N01.npy",
]

abnormal_samples_ries_21 = [
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]Ries5.0_B04_N01.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]Ries5.0_B18_N01.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]Ries5.0_B30_N07.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]Ries5.0_B06_N01.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]Ries5.0_B08_N07.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]Ries5.0_B24_N05.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]Ries6.1F_B34_N05.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]Ries5.1F_B36_N01.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]Ries6.0_B28_N03.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]Ries5.1F_B42_N05.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]Ries6.1F_B45_N08.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]Ries6.0_B05_N02.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]Ries6.0_B10_N06.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]Ries6.1F_B32_N01.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]Ries6.0_B03_N07.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]Ries5.0_B22_N06.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]Ries6.1F_B37_N07.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]Ries5.0_B02_N06.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]Ries5.0_B23_N05.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]Ries5.0_B14_N06.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]Ries5.0_B05_N07.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]Ries6.1F_B37_N01.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]Ries6.0_B02_N05.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]Ries6.0_B19_N01.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]Ries6.0_B13_N05.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]RIES5.1F_B32_N05.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]RIES6.1F_B42_N05.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]RIES5.0_B07_N08.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]RIES5.0_B04_N09.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]RIES5.0_B21_N06.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]RIES5.0_B27_N02.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]RIES6.1F_B33_N04.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]RIES6.1F_B33_N08.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]RIES5.0_B28_N08.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]RIES6.0_B07_N09.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]RIES5.1F_B43_N02.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]RIES5.1F_B34_N03.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]RIES5.1F_B40_N08.npy",
]

abnormal_samples_cf = [
    r"D:\Bud Thermography Data\round_5\CF5.0_B01_N04.npy",
    r"D:\Bud Thermography Data\round_5\CF5.0_B27_N05.npy",
    r"D:\Bud Thermography Data\round_5\CF5.0_B01_N05.npy",
    r"D:\Bud Thermography Data\round_6\CF6.1F_B31_N09.npy", #falling
    r"D:\Bud Thermography Data\round_5\CF5.0_B01_N02.npy",
    r"D:\Bud Thermography Data\round_5\CF5.0_B03_N09.npy",
    r"D:\Bud Thermography Data\round_5\CF5.0_B02_N07.npy", #falling
    r"D:\Bud Thermography Data\round_5\CF5.0_B08_N02.npy", #falling
    r"D:\Bud Thermography Data\round_6\CF6.0_B15_N01.npy",
    r"D:\Bud Thermography Data\round_5\CF5.0_B04_N04.npy", #falling
    r"D:\Bud Thermography Data\round_6\CF6.1F_B33_N04.npy", #falling
    r"D:\Bud Thermography Data\round_5\CF5.0_B01_N07.npy",
    r"D:\Bud Thermography Data\round_6\CF6.1F_B41_N03.npy", #falling
    r"D:\Bud Thermography Data\round_5\CF5.0_B28_N03.npy", #falling
    r"D:\Bud Thermography Data\round_6\CF6.0_B18_N06.npy", #falling
    r"D:\Bud Thermography Data\round_5\CF5.0_B19_N02.npy",
    r"D:\Bud Thermography Data\round_5\CF5.0_B14_N07.npy",
    r"D:\Bud Thermography Data\round_5\CF5.0_B07_N02.npy",
    r"D:\Bud Thermography Data\round_5\CF5.0_B01_N09.npy",
    r"D:\Bud Thermography Data\round_6\CF6.0_B21_N02.npy",
    r"D:\Bud Thermography Data\round_6\CF6.0_B04_N08.npy",
    r"D:\Bud Thermography Data\round_5\CF5.0_B01_N08.npy",
    r"D:\Bud Thermography Data\round_5\CF5.1F_B44_N08.npy",
    r"D:\Bud Thermography Data\round_5\CF5.0_B01_N03.npy",
    r"D:\Bud Thermography Data\round_5\CF5.0_B04_N08.npy",
    r"D:\Bud Thermography Data\round_5\CF5.0_B15_N06.npy", #falling
    r"D:\Bud Thermography Data\round_6\CF6.1F_B35_N07.npy", #falling
    r"D:\Bud Thermography Data\round_6\CF6.1F_B31_N04.npy", #falling
    r"D:\Bud Thermography Data\round_5\CF5.0_B03_N02.npy",
    r"D:\Bud Thermography Data\round_6\CF6.0_B28_N05.npy", #falling
    r"D:\Bud Thermography Data\round_5\CF5.0_B04_N05.npy", #falling
    r"D:\Bud Thermography Data\round_5\CF5.0_B10_N04.npy", #falling
    r"D:\Bud Thermography Data\round_5\CF5.1F_B32_N01.npy", #falling
    r"D:\Bud Thermography Data\round_5\CF5.0_B02_N03.npy",
    r"D:\Bud Thermography Data\round_5\CF5.0_B01_N01.npy",
    r"D:\Bud Thermography Data\round_5\CF5.1F_B33_N01.npy",
    r"D:\Bud Thermography Data\round_5\CF5.0_B02_N01.npy", #falling
    r"D:\Bud Thermography Data\round_3\CF3.1F_B38_N01.npy",
    r"D:\Bud Thermography Data\round_3\CF3.0_B03_N02.npy",
    r"D:\Bud Thermography Data\round_4\CF4.1_B39_N04.npy",
    r"D:\Bud Thermography Data\round_4\CF4.0_B15_N09.npy", #falling
    r"D:\Bud Thermography Data\round_2\CF_2.0_B24_N05.npy", #falling
    r"D:\Bud Thermography Data\round_4\CF4.0_B02_N01.npy",
    r"D:\Bud Thermography Data\round_4\CF4.1_B33_N04.npy", #falling
    r"D:\Bud Thermography Data\round_4\CF4.0_B11_N04.npy",
    r"D:\Bud Thermography Data\round_2\CF_2.0_B01_N09.npy", #falling
    r"D:\Bud Thermography Data\round_2\CF_2.0_B29_N06.npy", #falling
    r"D:\Bud Thermography Data\round_4\CF4.0_B29_N01.npy",
    r"D:\Bud Thermography Data\round_3\CF3.0_B17_N09.npy", #falling
    r"D:\Bud Thermography Data\round_3\CF3.1F_B35_N02.npy", #falling
    r"D:\Bud Thermography Data\round_2\CF_2.0_B17_N02.npy", #falling
    r"D:\Bud Thermography Data\round_2\CF_2.0_B14_N02.npy", #falling
    r"D:\Bud Thermography Data\round_3\CF3.0_B07_N05.npy",
    r"D:\Bud Thermography Data\round_4\CF4.0_B07_N04.npy",
    r"D:\Bud Thermography Data\round_2\CF_2.1_F_B32_N06.npy",
    r"D:\Bud Thermography Data\round_4\CF4.0_B23_N06.npy",
    r"D:\Bud Thermography Data\round_4\CF4.0_B14_N08.npy", #falling
    r"D:\Bud Thermography Data\round_3\CF3.0_B12_N01.npy", #falling
    r"D:\Bud Thermography Data\round_4\CF4.1_B42_N08.npy", #falling
    r"D:\Bud Thermography Data\round_4\CF4.1_B40_N04.npy",
    r"D:\Bud Thermography Data\round_2\CF_2.0_B16_N05.npy", #falling
    r"D:\Bud Thermography Data\round_3\CF3.0_B10_N06.npy", #falling
    r"D:\Bud Thermography Data\round_3\CF3.0_B24_N07.npy",
    r"D:\Bud Thermography Data\round_2\CF_2.0_B02_N07.npy", #falling
    r"D:\Bud Thermography Data\round_2\CF_2.1_F_B33_N08.npy",
    r"D:\Bud Thermography Data\round_2\CF_2.1_F_B32_N08.npy",
    r"D:\Bud Thermography Data\round_4\CF4.0_B24_N04.npy",
    r"D:\Bud Thermography Data\round_2\CF_2.1_F_B31_N05.npy",
    r"D:\Bud Thermography Data\round_2\CF_2.1_F_B42_N08.npy",
    r"D:\Bud Thermography Data\round_2\CF_2.1_F_B41_N07.npy",
    r"D:\Bud Thermography Data\round_2\CF_2.0_B21_N06.npy",
    r"D:\Bud Thermography Data\round_3\CF3.1F_B40_N08.npy", #falling
    r"D:\Bud Thermography Data\round_4\CF4.1_B45_N09.npy",
    r"D:\Bud Thermography Data\round_3\CF3.1F_B43_N08.npy",
    r"D:\Bud Thermography Data\round_4\CF4.0_B01_N04.npy", #falling
    r"D:\Bud Thermography Data\round_2\CF_2.0_B15_N02.npy", #falling
    r"D:\Bud Thermography Data\round_4\CF4.0_B10_N04.npy", #falling
    r"D:\Bud Thermography Data\round_2\CF_2.1_F_B37_N06.npy", #falling
    r"D:\Bud Thermography Data\round_2\CF_2.0_B07_N09.npy", #falling
    r"D:\Bud Thermography Data\round_4\CF4.0_B19_N09.npy", #falling
    r"D:\Bud Thermography Data\round_4\CF4.0_B17_N03.npy",
    r"D:\Bud Thermography Data\round_4\CF4.0_B29_N04.npy",
    r"D:\Bud Thermography Data\round_4\CF4.1_B31_N05.npy", #falling
    r"D:\Bud Thermography Data\round_4\CF4.0_B01_N07.npy",
    r"D:\Bud Thermography Data\round_2\CF_2.0_B05_N06.npy",
    r"D:\Bud Thermography Data\round_2\CF_2.1_F_B31_N01.npy",
    r"D:\Bud Thermography Data\round_2\CF_2.0_B22_N06.npy", #falling
    r"D:\Bud Thermography Data\round_3\CF3.0_B02_N02.npy", #falling
    r"D:\Bud Thermography Data\round_3\CF3.1F_B36_N04.npy", #falling
    r"D:\Bud Thermography Data\round_4\CF4.0_B05_N06.npy",
]

abnormal_samples_cf_21 = [
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B01_N04.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B27_N05.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B01_N05.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CF6.1F_B31_N09.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B01_N02.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B03_N09.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B02_N07.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B08_N02.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CF6.0_B15_N01.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B04_N04.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CF6.1F_B33_N04.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B01_N07.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CF6.1F_B41_N03.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B28_N03.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CF6.0_B18_N06.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B19_N02.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B14_N07.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B07_N02.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B01_N09.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CF6.0_B21_N02.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CF6.0_B04_N08.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B01_N08.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.1F_B44_N08.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B01_N03.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B04_N08.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B15_N06.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CF6.1F_B35_N07.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CF6.1F_B31_N04.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B03_N02.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CF6.0_B28_N05.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B04_N05.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B10_N04.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.1F_B32_N01.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B02_N03.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B01_N01.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.1F_B33_N01.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CF5.0_B02_N01.npy", #falling
]

abnormal_samples_con = [
    r"D:\Bud Thermography Data\round_6\CON6.1F_B42_N04.npy",
    r"D:\Bud Thermography Data\round_5\CON5.0_B03_N07.npy",
    r"D:\Bud Thermography Data\round_6\CON6.0_B08_N03.npy",
    r"D:\Bud Thermography Data\round_6\CON6.1F_B38_N08.npy", #falling
    r"D:\Bud Thermography Data\round_6\CON6.0_B21_N09.npy", #falling
    r"D:\Bud Thermography Data\round_6\CON6.0_B30_N07.npy", #falling
    r"D:\Bud Thermography Data\round_6\CON6.0_B21_N01.npy",
    r"D:\Bud Thermography Data\round_6\CON6.0_B02_N08.npy",
    r"D:\Bud Thermography Data\round_6\CON6.0_B01_N03.npy",
    r"D:\Bud Thermography Data\round_6\CON6.1F_B33_N03.npy",
    r"D:\Bud Thermography Data\round_5\CON5.0_B24_N02.npy",
    r"D:\Bud Thermography Data\round_5\CON5.1F_B38_N02.npy", #falling
    r"D:\Bud Thermography Data\round_6\CON6.0_B01_N01.npy",
    r"D:\Bud Thermography Data\round_5\CON5.0_B10_N04.npy",
    r"D:\Bud Thermography Data\round_6\CON6.0_B27_N04.npy", #falling
    r"D:\Bud Thermography Data\round_6\CON6.1F_B36_N03.npy",
    r"D:\Bud Thermography Data\round_5\CON5.0_B05_N06.npy",
    r"D:\Bud Thermography Data\round_6\CON6.0_B01_N02.npy",
    r"D:\Bud Thermography Data\round_6\CON6.0_B16_N04.npy",
    r"D:\Bud Thermography Data\round_6\CON6.1F_B38_N09.npy",
    r"D:\Bud Thermography Data\round_5\CON5.1F_B39_N07.npy",
    r"D:\Bud Thermography Data\round_5\CON5.0_B03_N03.npy",
    r"D:\Bud Thermography Data\round_6\CON6.0_B29_N01.npy",
    r"D:\Bud Thermography Data\round_6\CON6.1F_B39_N02.npy", #falling
    r"D:\Bud Thermography Data\round_6\CON6.0_B25_N01.npy", #falling
    r"D:\Bud Thermography Data\round_5\CON5.0_B19_N09.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0_B02_N01.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0F_B34_N04.npy",
    r"D:\Bud Thermography Data\round_2\Con2.0_B10_N02.npy",
    r"D:\Bud Thermography Data\round_4\con4.0_B03_N07.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0_B01_N09.npy",
    r"D:\Bud Thermography Data\round_2\Con2.0_B01_N06.npy",
    r"D:\Bud Thermography Data\round_3\con3.0_B08_N04.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0_B01_N03.npy", #falling
    r"D:\Bud Thermography Data\round_2\Con2.0_B02_N01.npy", #falling
    r"D:\Bud Thermography Data\round_3\con3.1F_B33_N04.npy", #falling
    r"D:\Bud Thermography Data\round_2\Con_2.1_F_B32_N02.npy",
    r"D:\Bud Thermography Data\round_4\con4.0_B29_N07.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0_B03_N02.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0F_B38_N08.npy",
    r"D:\Bud Thermography Data\round_3\con3.0_B18_N03.npy", #falling
    r"D:\Bud Thermography Data\round_3\con3.1F_B31_N09.npy",
    r"D:\Bud Thermography Data\round_2\Con_2.1_F_B39_N06.npy",
    r"D:\Bud Thermography Data\round_4\con4.0F_B39_N08.npy",
    r"D:\Bud Thermography Data\round_4\con4.0_B01_N06.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0F_B42_N03.npy",
    r"D:\Bud Thermography Data\round_3\con3.0_B05_N03.npy",
    r"D:\Bud Thermography Data\round_3\con3.1F_B34_N02.npy",
    r"D:\Bud Thermography Data\round_3\con3.1F_B32_N05.npy",
    r"D:\Bud Thermography Data\round_4\con4.0_B22_N01.npy",
    r"D:\Bud Thermography Data\round_3\con3.1F_B31_N02.npy", #falling
    r"D:\Bud Thermography Data\round_3\con3.0_B02_N01.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0F_B32_N09.npy", #falling
    r"D:\Bud Thermography Data\round_3\con3.0_B26_N03.npy",
    r"D:\Bud Thermography Data\round_2\Con2.0_B24_N04.npy",
    r"D:\Bud Thermography Data\round_3\con3.1F_B33_N02.npy",
    r"D:\Bud Thermography Data\round_3\con3.0_B14_N06.npy", #falling
    r"D:\Bud Thermography Data\round_2\Con2.0_B06_N07.npy",
    r"D:\Bud Thermography Data\round_4\con4.0_B10_N04.npy",
    r"D:\Bud Thermography Data\round_3\con3.1F_B32_N01.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0_B02_N02.npy",
    r"D:\Bud Thermography Data\round_3\con3.0_B13_N01.npy",
    r"D:\Bud Thermography Data\round_4\con4.0_B03_N05.npy",
    r"D:\Bud Thermography Data\round_4\con4.0_B15_N05.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0_B13_N02.npy", #falling
    r"D:\Bud Thermography Data\round_3\con3.1F_B31_N05.npy",
    r"D:\Bud Thermography Data\round_4\con4.0_B01_N05.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0_B01_N04.npy", #falling
    r"D:\Bud Thermography Data\round_2\Con2.0_B04_N01.npy", #falling
    r"D:\Bud Thermography Data\round_2\Con2.0_B08_N04.npy",
    r"D:\Bud Thermography Data\round_2\Con2.0_B01_N01.npy",
    r"D:\Bud Thermography Data\round_3\con3.1F_B40_N03.npy", #falling
    r"D:\Bud Thermography Data\round_2\Con2.0_B04_N03.npy",
    r"D:\Bud Thermography Data\round_4\con4.0_B01_N02.npy",
    r"D:\Bud Thermography Data\round_4\con4.0_B07_N01.npy", #falling
    r"D:\Bud Thermography Data\round_3\con3.0_B15_N02.npy", #falling
    r"D:\Bud Thermography Data\round_3\con3.1F_B32_N02.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0_B17_N04.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0_B01_N01.npy",
    r"D:\Bud Thermography Data\round_3\con3.0_B27_N01.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0_B05_N05.npy",
    r"D:\Bud Thermography Data\round_3\con3.1F_B31_N08.npy",
    r"D:\Bud Thermography Data\round_4\con4.0_B30_N08.npy",
    r"D:\Bud Thermography Data\round_3\con3.1F_B31_N07.npy", #falling
    r"D:\Bud Thermography Data\round_2\Con2.0_B21_N02.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0F_B37_N03.npy",
    r"D:\Bud Thermography Data\round_2\Con_2.1_F_B32_N06.npy",
    r"D:\Bud Thermography Data\round_3\con3.1F_B32_N04.npy",
    r"D:\Bud Thermography Data\round_2\Con2.0_B26_N05.npy", #falling
    r"D:\Bud Thermography Data\round_2\Con_2.1_F_B41_N01.npy", #falling
    r"D:\Bud Thermography Data\round_4\con4.0F_B33_N02.npy",
    r"D:\Bud Thermography Data\round_4\con4.0_B23_N06.npy", #falling
]

abnormal_samples_con_21 = [
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.1F_B42_N04.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CON5.0_B03_N07.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.0_B08_N03.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.1F_B38_N08.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.0_B21_N09.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.0_B30_N07.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.0_B21_N01.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.0_B02_N08.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.0_B01_N03.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.1F_B33_N03.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CON5.0_B24_N02.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CON5.1F_B38_N02.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.0_B01_N01.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CON5.0_B10_N04.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.0_B27_N04.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.1F_B36_N03.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CON5.0_B05_N06.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.0_B01_N02.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.0_B16_N04.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.1F_B38_N09.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CON5.1F_B39_N07.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CON5.0_B03_N03.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.0_B29_N01.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.1F_B39_N02.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]CON6.0_B25_N01.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]CON5.0_B19_N09.npy", #falling
]

abnormal_samples_pn = [
    r"D:\Bud Thermography Data\round_6\PN6.0_B27_N05.npy", #falling
    r"D:\Bud Thermography Data\round_5\PN5.0_B02_N08.npy", #falling
    r"D:\Bud Thermography Data\round_5\PN5.0_B07_N04.npy",
    r"D:\Bud Thermography Data\round_5\PN5.1F_B36_N09.npy", #falling
    r"D:\Bud Thermography Data\round_5\PN5.1F_B32_N01.npy", #falling
    r"D:\Bud Thermography Data\round_5\PN5.0_B03_N05.npy",
    r"D:\Bud Thermography Data\round_5\PN5.1F_B34_N05.npy",
    r"D:\Bud Thermography Data\round_5\PN5.1F_B37_N08.npy",
    r"D:\Bud Thermography Data\round_5\PN5.1F_B39_N05.npy", #falling
    r"D:\Bud Thermography Data\round_6\PN6.0_B03_N01.npy",
    r"D:\Bud Thermography Data\round_6\PN6.0_B24_N03.npy",
    r"D:\Bud Thermography Data\round_6\PN6.0_B10_N06.npy", #falling
    r"D:\Bud Thermography Data\round_6\PN6.1F_B40_N06.npy",
    r"D:\Bud Thermography Data\round_5\PN5.0_B20_N06.npy",
    r"D:\Bud Thermography Data\round_5\PN5.0_B25_N09.npy",
    r"D:\Bud Thermography Data\round_6\PN6.0_B04_N06.npy",
    r"D:\Bud Thermography Data\round_6\PN6.0_B13_N03.npy",
    r"D:\Bud Thermography Data\round_5\PN5.0_B16_N05.npy", #falling
    r"D:\Bud Thermography Data\round_5\PN5.1F_B38_N07.npy",
    r"D:\Bud Thermography Data\round_6\PN6.0_B03_N07.npy",
    r"D:\Bud Thermography Data\round_6\PN6.1F_B35_N07.npy", #falling
    r"D:\Bud Thermography Data\round_6\PN6.0_B23_N05.npy",
    r"D:\Bud Thermography Data\round_5\PN5.1F_B40_N04.npy", #falling
    r"D:\Bud Thermography Data\round_6\PN6.1F_B31_N02.npy", #falling
    r"D:\Bud Thermography Data\round_4\PN4.0_B09_N03.npy",
    r"D:\Bud Thermography Data\round_2\PN_2.0_B24_N03.npy", #falling
    r"D:\Bud Thermography Data\round_3\PN3.0_B19_N09.npy", #falling
    r"D:\Bud Thermography Data\round_3\PN3.0_B13_N02.npy",
    r"D:\Bud Thermography Data\round_2\PN_2.0_B06_N04.npy",
    r"D:\Bud Thermography Data\round_4\PN4.0_B01_N03.npy",
    r"D:\Bud Thermography Data\round_4\PN4.0_B04_N04.npy",
    r"D:\Bud Thermography Data\round_4\PN4.0_B20_N05.npy",
    r"D:\Bud Thermography Data\round_3\PN3.0_B10_N02.npy", #falling
    r"D:\Bud Thermography Data\round_4\PN4.0_B05_N06.npy",
    r"D:\Bud Thermography Data\round_4\PN4.0_B02_N02.npy",
    r"D:\Bud Thermography Data\round_3\PN3.0_B29_N07.npy", #falling
    r"D:\Bud Thermography Data\round_3\PN3.0_B05_N06.npy",
    r"D:\Bud Thermography Data\round_4\PN4.0_B27_N04.npy",
    r"D:\Bud Thermography Data\round_3\PN3.1F_B36_N01.npy",
    r"D:\Bud Thermography Data\round_3\PN3.1F_B34_N07.npy", #falling
    r"D:\Bud Thermography Data\round_3\PN3.0_B24_N07.npy",
    r"D:\Bud Thermography Data\round_3\PN3.0_B25_N09.npy", #falling
    r"D:\Bud Thermography Data\round_2\PN_2.0_B28_N01.npy", #falling
    r"D:\Bud Thermography Data\round_3\PN3.1F_B32_N01.npy",
    r"D:\Bud Thermography Data\round_4\PN4.1F_B38_N02.npy", #falling
    r"D:\Bud Thermography Data\round_3\PN3.1F_B36_N05.npy",
    r"D:\Bud Thermography Data\round_4\PN4.0_B01_N02.npy",
    r"D:\Bud Thermography Data\round_4\PN4.0_B19_N07.npy", #falling
    r"D:\Bud Thermography Data\round_2\PN_2.1_F_B30_N07.npy",
    r"D:\Bud Thermography Data\round_2\PN_2.0_B15_N06.npy",
    r"D:\Bud Thermography Data\round_2\PN_2.0_B11_N06.npy",
    r"D:\Bud Thermography Data\round_4\PN4.0_B13_N01.npy", #falling
    r"D:\Bud Thermography Data\round_3\PN3.1F_B42_N02.npy",
    r"D:\Bud Thermography Data\round_3\PN3.1F_B40_N05.npy",
    r"D:\Bud Thermography Data\round_3\PN3.0_B01_N08.npy",
    r"D:\Bud Thermography Data\round_2\PN_2.0_B17_N09.npy",
    r"D:\Bud Thermography Data\round_3\PN3.1F_B31_N02.npy",
    r"D:\Bud Thermography Data\round_2\PN_2.1_F_B39_N04.npy", #falling
    r"D:\Bud Thermography Data\round_3\PN3.0_B04_N07.npy", #falling
    r"D:\Bud Thermography Data\round_4\PN4.0_B18_N06.npy", #falling
    r"D:\Bud Thermography Data\round_3\PN3.1F_B33_N06.npy",
    r"D:\Bud Thermography Data\round_3\PN3.1F_B32_N07.npy",
    r"D:\Bud Thermography Data\round_4\PN4.0_B01_N06.npy", #falling
    r"D:\Bud Thermography Data\round_3\PN3.1F_B37_N05.npy",
    r"D:\Bud Thermography Data\round_2\PN_2.1_F_B43_N03.npy", #falling
    r"D:\Bud Thermography Data\round_3\PN3.0_B26_N04.npy",
    r"D:\Bud Thermography Data\round_4\PN4.0_B04_N01.npy", #falling
    r"D:\Bud Thermography Data\round_2\PN_2.1_F_B41_N05.npy",
    r"D:\Bud Thermography Data\round_4\PN4.0_B13_N05.npy",
    r"D:\Bud Thermography Data\round_4\PN4.0_B01_N05.npy",
    r"D:\Bud Thermography Data\round_3\PN3.1F_B31_N03.npy",
    r"D:\Bud Thermography Data\round_4\PN4.1F_B31_N04.npy", #falling
    r"D:\Bud Thermography Data\round_4\PN4.0_B04_N09.npy",
    r"D:\Bud Thermography Data\round_4\PN4.0_B03_N02.npy", #falling
    r"D:\Bud Thermography Data\round_3\PN3.0_B07_N05.npy", #falling
    r"D:\Bud Thermography Data\round_2\PN_2.1_F_B44_N01.npy", #falling
    r"D:\Bud Thermography Data\round_3\PN3.1F_B32_N09.npy",
    r"D:\Bud Thermography Data\round_2\PN_2.0_B13_N07.npy",
    r"D:\Bud Thermography Data\round_3\PN3.1F_B31_N08.npy",
    r"D:\Bud Thermography Data\round_4\PN4.0_B21_N03.npy",
    r"D:\Bud Thermography Data\round_2\PN_2.1_F_B40_N03.npy", #falling
    r"D:\Bud Thermography Data\round_3\PN3.1F_B41_N03.npy", #falling
    r"D:\Bud Thermography Data\round_3\PN3.0_B23_N04.npy", #falling
    r"D:\Bud Thermography Data\round_4\PN4.1F_B44_N05.npy", #falling
    r"D:\Bud Thermography Data\round_4\PN4.0_B02_N03.npy",
    r"D:\Bud Thermography Data\round_2\PN_2.0_B11_N01.npy",
    r"D:\Bud Thermography Data\round_4\PN4.1F_B37_N04.npy", #falling
    r"D:\Bud Thermography Data\round_4\PN4.1F_B31_N06.npy", #falling
    r"D:\Bud Thermography Data\round_4\PN4.0_B15_N01.npy",
    r"D:\Bud Thermography Data\round_3\PN3.1F_B31_N04.npy",
    r"D:\Bud Thermography Data\round_4\PN4.1F_B38_N04.npy", #falling
]

abnormal_samples_pn_21 = [
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]PN6.0_B27_N05.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]PN5.0_B02_N08.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]PN5.0_B07_N04.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]PN5.1F_B36_N09.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]PN5.1F_B32_N01.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]PN5.0_B03_N05.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]PN5.1F_B34_N05.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]PN5.1F_B37_N08.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]PN5.1F_B39_N05.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]PN6.0_B03_N01.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]PN6.0_B24_N03.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]PN6.0_B10_N06.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]PN6.1F_B40_N06.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]PN5.0_B20_N06.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]PN5.0_B25_N09.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]PN6.0_B04_N06.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]PN6.0_B13_N03.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]PN5.0_B16_N05.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]PN5.1F_B38_N07.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]PN6.0_B03_N07.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]PN6.1F_B35_N07.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]PN6.0_B23_N05.npy",
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_5\[ROI]PN5.1F_B40_N04.npy", #falling
    r"C:\Users\Mud\Desktop\2ndPaper\roi_data\size-21\round_6\[ROI]PN6.1F_B31_N02.npy", #falling
]

# combine all abnormal samples
abnormal_samples = abnormal_samples_ries + abnormal_samples_cf + abnormal_samples_con + abnormal_samples_pn
abnormal_samples_set = set(abnormal_samples)
abnormal_samples_21 = abnormal_samples_ries_21 + abnormal_samples_cf_21 + abnormal_samples_con_21 + abnormal_samples_pn_21
abnormal_samples_21_set = set(abnormal_samples_21)



def parse_round_num(file_path: str) -> int:
    """
    Extract the round number from the file path.
    """
    match = re.search(r'round_(\d+)', file_path)
    if match:
        return int(match.group(1))
    raise ValueError(f"Round number not found in {file_path}")

def get_abnormal_samples_count(file_paths: list) -> dict:
    result = [0, 0, 0, 0, 0] # round 2, 3, 4, 5, 6
    for file_path in file_paths:
        round_num = parse_round_num(file_path)
        result[round_num - 2] += 1
    return result  

# print the number of abnormal samples of each round, each cultivar
samples_count_ries = get_abnormal_samples_count(abnormal_samples_ries)
samples_count_cf = get_abnormal_samples_count(abnormal_samples_cf)
samples_count_con = get_abnormal_samples_count(abnormal_samples_con)
samples_count_pn = get_abnormal_samples_count(abnormal_samples_pn)
print("Number of abnormal samples of each round, each cultivar")
print(f"Riesling: round 2: {samples_count_ries[0]}, round 3: {samples_count_ries[1]}, round 4: {samples_count_ries[2]}, round 5: {samples_count_ries[3]}, round 6: {samples_count_ries[4]}")
print(f"Cabernet Franc: round 2: {samples_count_cf[0]}, round 3: {samples_count_cf[1]}, round 4: {samples_count_cf[2]}, round 5: {samples_count_cf[3]}, round 6: {samples_count_cf[4]}")
print(f"Concord: round 2: {samples_count_con[0]}, round 3: {samples_count_con[1]}, round 4: {samples_count_con[2]}, round 5: {samples_count_con[3]}, round 6: {samples_count_con[4]}")
print(f"Pinot Noir: round 2: {samples_count_pn[0]}, round 3: {samples_count_pn[1]}, round 4: {samples_count_pn[2]}, round 5: {samples_count_pn[3]}, round 6: {samples_count_pn[4]}")
