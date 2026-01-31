// Trajecto: Real-time 3D Trajectory Reconstruction System (Software)
// Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// NOTICE: This software implements the "Hybrid ESKF-Stateful TCN" logic
// protected under ROK Patent Application No. 10-2025-YYYYYYY.
// Commercial use requires a separate license from the author.

import UIKit
import PencilKit

// MARK: - Apple Pencil Pro Features

/// Extension for `EnhancedPKCanvasView` to support Apple Pencil Pro specific features.
extension EnhancedPKCanvasView {
    
    /// Configures the canvas to handle Apple Pencil Pro interactions, such as the squeeze gesture.
    func setupPencilProFeatures() {
        if #available(iOS 18.0, *) {
                let squeezeGesture = UIPencilInteraction()
                squeezeGesture.delegate = self
                
                squeezeGesture.isEnabled = false
                
                addInteraction(squeezeGesture)
            }
    }
}

// MARK: - UIPencilInteractionDelegate

/// Delegate implementation for handling Apple Pencil Pro interactions.
@available(iOS 18.0, *)
extension EnhancedPKCanvasView: UIPencilInteractionDelegate {
    
    /// Called when a tap interaction is detected on the pencil.
    /// - Parameter interaction: The interaction object reporting the event.
    func pencilInteractionDidTap(_ interaction: UIPencilInteraction) {
        // Handle double tap or squeeze actions here
        print("Apple Pencil interaction detected")
    }
}

// MARK: - Data Structures

/// Structure to hold additional sensor data specific to Apple Pencil Pro.
struct ApplePencilProData {
    let squeezeForce: CGFloat
    let rollAngle: CGFloat
    let hoverDistance: CGFloat
    let hapticFeedback: Bool
    
    init(squeezeForce: CGFloat = 0, rollAngle: CGFloat = 0, hoverDistance: CGFloat = 0, hapticFeedback: Bool = false) {
        self.squeezeForce = squeezeForce
        self.rollAngle = rollAngle
        self.hoverDistance = hoverDistance
        self.hapticFeedback = hapticFeedback
    }
}
