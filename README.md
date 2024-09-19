# BallMill3D
3D Triangulation of Ball Movement Inside a Mixermill

## Architecture
![Architecture](.assets/Architecture.png)


## Current Status: 

## Current Status

```mermaid
graph TD
  A["camera / stereo_camera (implemented &amp; tested)"]
  B["calibration_manager (implemented)"]
  C["camera_frame_provider (implemented)"]
  D["corner_detector (not implemented)"]
  E["triangulator (not implemented)"]
  F["validator (not implemented)"]

  B -->|dep.| A
  C -->|dep.| A
  B -->|dep.| C
  D -->|dep.| A
  E -->|dep.| A
  F -->|dep.| A
  F -->|dep.| B
  F -->|dep.| D
  F -->|dep.| E

  %% Define classes for styling
  classDef implemented fill:#a8f0a8,stroke:#333,stroke-width:2px;
  classDef notImplemented fill:#f0a8a8,stroke:#333,stroke-width:2px;

  %% Assign classes to nodes
  class A,B,C implemented;
  class D,E,F notImplemented;


  
  
