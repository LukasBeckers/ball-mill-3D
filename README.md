# BallMill3D
3D Triangulation of Ball Movement Inside a Mixermill

## Architecture
![Architecture](.assets/Architecture.png)


## Current Status: 

```mermaid
graph TD
  A[camera / stereo_camera (implemented & tested)]
  B[calibration_manager (implemented)]
  C[camera_frame_provider (implemented)]
  D[corner_detector (not implemented)]
  E[triangulator (not implemented)]
  F[validator (not implemented)]

  A <-- |dep.| B
  A <--|dep.| C
  B -->|dep.| C
  D -->|dep.| A
  
  
