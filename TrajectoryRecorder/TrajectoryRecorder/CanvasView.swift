//
//  CanvasView.swift
//  TrajectoryRecorder
//
//  Created by haro on 7/17/25.
//
//  NOTE: This file contains legacy or alternative implementations of canvas views.
//  The active implementation is currently EnhancedCanvasView.swift.
//

import SwiftUI
import PencilKit
import UIKit

// MARK: - Legacy Canvas View

/// A basic `PKCanvasView` wrapper.
/// - Note: Currently not used in the main `ContentView`. See `EnhancedCanvasView`.
struct CanvasView: UIViewRepresentable {
    @ObservedObject var dataRecorder: PencilDataRecorder
    @State private var toolPicker = PKToolPicker()
    
    func makeUIView(context: Context) -> PKCanvasView {
        let canvasView = PKCanvasView()
        canvasView.delegate = context.coordinator
        canvasView.drawingPolicy = .anyInput
        canvasView.tool = PKInkingTool(.pen, color: .black, width: 5)
        
        // Enable tool picker
        toolPicker.setVisible(true, forFirstResponder: canvasView)
        toolPicker.addObserver(canvasView)
        canvasView.becomeFirstResponder()
        
        return canvasView
    }
    
    func updateUIView(_ uiView: PKCanvasView, context: Context) {
        // Update canvas if needed
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(dataRecorder: dataRecorder)
    }
    
    class Coordinator: NSObject, PKCanvasViewDelegate {
        let dataRecorder: PencilDataRecorder
        
        init(dataRecorder: PencilDataRecorder) {
            self.dataRecorder = dataRecorder
        }
        
        func canvasViewDrawingDidChange(_ canvasView: PKCanvasView) {
            // Handle drawing changes if needed
        }
    }
}

// MARK: - Legacy Pencil Tracking View

/// A custom view to manually capture touch events for pencil tracking.
/// - Note: This manual tracking approach is an alternative to `PKCanvasView` subclassing.
class PencilTrackingView: UIView {
    var dataRecorder: PencilDataRecorder?
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesBegan(touches, with: event)
        handleTouches(touches, with: event)
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesMoved(touches, with: event)
        handleTouches(touches, with: event)
        
        // Handle coalesced touches for high-frequency data
        if let coalescedTouches = event?.coalescedTouches(for: touches.first!) {
            for touch in coalescedTouches {
                handleTouch(touch)
            }
        }
    }
    
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesEnded(touches, with: event)
        handleTouches(touches, with: event)
    }
    
    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesCancelled(touches, with: event)
        handleTouches(touches, with: event)
    }
    
    private func handleTouches(_ touches: Set<UITouch>, with event: UIEvent?) {
        for touch in touches {
            handleTouch(touch)
        }
    }
    
    private func handleTouch(_ touch: UITouch) {
        let location = touch.preciseLocation(in: self)
        let force = touch.force
        let azimuth = touch.azimuthAngle(in: self)
        let altitude = touch.altitudeAngle
        let timestamp = touch.timestamp
        
        // For Apple Pencil Pro, get roll angle if available
        var rollAngle: CGFloat = touch.rollAngle
        
        let dataPoint = PencilDataPoint(
            x: location.x,
            y: location.y,
            force: force,
            azimuth: azimuth,
            altitude: altitude,
            hoverDistance: 0, // Screen touch, no hover
            timestamp: timestamp,
            isHovering: false,
            rollAngle: rollAngle
        )
        
        dataRecorder?.addDataPoint(dataPoint)
    }
}
