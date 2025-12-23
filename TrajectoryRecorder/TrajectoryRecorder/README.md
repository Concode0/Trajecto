# Trajectory Recorder

Welcome to Trajectory Recorder, a high-fidelity data acquisition application for Apple Pencil Pro. This project focuses on capturing precise motion data for research and analysis, particularly for validating the performance of neural-inertial hybrid reconstruction systems like Trajecto.

Trajectory Recorder is a dedicated data acquisition application developed to validate the performance of Trajecto, a neural-inertial hybrid reconstruction system. By leveraging native iPadOS APIs and the advanced hardware capabilities of the Apple Pencil Pro, it records high-fidelity motion data with microsecond-level precision.

## Key Engineering Highlights

1. High-Frequency Sampling (240Hz+)
- Coalesced Touches: The engine utilizes the coalescedTouches API to capture hardware-level events that occur between screen refresh cycles. This ensures that rapid handwriting dynamics are recorded at 240Hz or higher without data loss.
- Temporal Integrity: Every data point is logged with a high-resolution timestamp, enabling precise temporal alignment via Parabolic Fitting in the post-processing pipeline.

2. Apple Pencil Pro Advanced Integration
- 3D Hover Tracking: Utilizing UIHoverGestureRecognizer and the zOffset property (iOS 17.5+), the app tracks the physical distance between the pencil tip and the screen in real-time.
- 6-DoF Sensor Fusion: The system records Roll Angle, Azimuth, Altitude, and Force (Pressure) simultaneously to support complex 6-Degree-of-Freedom motion analysis.

3. Architecture & SDK Compliance
- Passive Logger Design: To ensure strict adherence to Apple's SDK policies, the app functions as a high-speed "Passive Logger". It records raw signals without unauthorized internal inversion or tampering.
- Decoupled Processing: All 3D coordinate transformations and physics-based modeling are performed in an external offline pipeline (Julia/Python). This separation of concerns ensures both platform compliance and maximum computational efficiency during data acquisition.

## Technical Specifications

- Frameworks: SwiftUI, PencilKit, and UIKit (Custom PKCanvasView Subclassing).

- Data Persistence: Sessions are exported as research-standard CSV files containing: timestamp, x, y, force, azimuth, altitude, hoverDistance, isHovering, rollAngle.
