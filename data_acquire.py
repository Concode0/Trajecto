"""
Running with Firmware/Trajecto-D

Pre-Process the Data ( GT - 3D Spline Interpolation / Gaussian Process Regression for Uncertainty
                       SB - Scale the data...?
                       Sync Task - DTW ( iPad 2D Accel with Savitzky-Golay - IMU Rigid Body Transformation ))

Save the Data in h5 -> GT / SB ( Raw / Pre-Process )
"""

import bleak
