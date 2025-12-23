# TrajectoLab Deveolopment Plan

## Layered Abstraction

### Data Perception Layer (Input Plugins):
- Receive Raw-Stream of IMU, Pressure, Tilt and convert it to standard stream.

### Estimation Core Layer (Algorithm Interface):
- Can choice Model ( In this projects ESKF-TCN, AEKF-TCN, Only-TCN)
- Using AbstractEstimator Type, Replace New Prediction Model in 'Plug and Play'

### Application Layer (Output Plugins):
- Using Predicted 3D Trajectory, Handwriting Recognition or Render Digital Calligraphy effects.


## Plugin Develop Plan

1. APS ( Attitude based Plane Segmentation)

2. Pressure-based Pen-State Analysis