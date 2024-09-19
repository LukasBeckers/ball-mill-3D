# BallMill3D
3D Triangulation of Ball Movement Inside a Mixermill

## Architecture
![Architecture](.assets/Architecture.png)


## Current Status: 

```mermaid
graph TD
  A[Start] --> B{Decision}
  B -->|Yes| C[Option 1]
  B -->|No| D[Option 2]
  C --> E[End]
  D --> E
