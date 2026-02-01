//
// Copyright (C) 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// [PATENT NOTICE]
// This implementation is protected under ROK Patent Applications 10-2025-0201093/092.
// Commercial use without a separate license is strictly prohibited.
//
// Contact: nemonanconcode@gmail.com
//

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
