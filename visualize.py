"""
The Dashboard for Analyzing the Data.

Select Pre-Process Type.
- LPF in IMU, iPad
LPF (IMU, iPAD)
Raw_Data Explorer
- x=time, y=[IMU, iPAD + velocity, acceleration]
Pre-Process Explorer
- DTW / Rigid Body Transformation -> It updates data.
Custom Algorithm Proofer
- 3D Plot | Two 2D Plot ( Data can be selected ) - Animation and time move by user.
3D Visualizer
- iPad Data show up
2D Visualizer
- Orientation Show up
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

true_distances = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

measured_z_offsets = np.array([0.0, 0.04, 0.09, 0.16, 0.24, 0.30, 0.40, 0.47, 0.56, 0.67, 0.75, 0.83, 0.96])

def exponential_model(x, a, b):
    return a * np.power(x, b)

try:
    popt, pcov = curve_fit(exponential_model, measured_z_offsets[1:], true_distances[1:], p0=[20, 1.5])
    
    a_opt, b_opt = popt
    print(f"Optimized Parameters: A = {a_opt:.4f}, p = {b_opt:.4f}")
    
    residuals = true_distances[1:] - exponential_model(measured_z_offsets[1:], *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((true_distances[1:] - np.mean(true_distances[1:]))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R-squared: {r_squared:.4f}")

except Exception as e:
    print(f"Fitting failed: {e}")
    a_opt, b_opt = 1, 1 # Fallback
    r_squared = 0

# 4. 시각화
plt.figure(figsize=(10, 6))

# 산점도 (실제 측정 데이터)
plt.scatter(measured_z_offsets, true_distances, color='blue', label='Measured Data (1mm steps)', s=50, zorder=5)

# 피팅된 곡선
x_fit = np.linspace(0, max(measured_z_offsets), 100)
y_fit = exponential_model(x_fit, a_opt, b_opt)
plt.plot(x_fit, y_fit, color='red', linestyle='--', linewidth=2, label=f'Fitted Model ($y = {a_opt:.2f} \cdot x^{{{b_opt:.2f}}}$)')

# 그래프 꾸미기
plt.title('Physical Verification of iPad Hovering Data (Z-Axis)', fontsize=16, fontweight='bold')
plt.xlabel('iPad zOffset (Normalized Sensor Value)', fontsize=14)
plt.ylabel('Physical Distance (mm)', fontsize=14)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=12)

# 텍스트 정보 추가 (R^2)
plt.text(0.05, 10, f'$R^2 = {r_squared:.4f}$', fontsize=14, color='darkred', bbox=dict(facecolor='white', alpha=0.8))

plt.xlim(0, 1)
plt.ylim(-0.5, 13)

# 저장 또는 보여주기
plt.tight_layout()
plt.show()