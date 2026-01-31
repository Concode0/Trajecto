// Trajecto: Real-time 3D Trajectory Reconstruction System (Software)
// Copyright 2025-2026 Eunkyum Kim <nemonanconcode@gmail.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
// NOTICE: This software implements the "Hybrid ESKF-Stateful TCN" logic
// protected under ROK Patent Application No. 10-2025-YYYYYYY.
// Commercial use requires a separate license from the author.

import SwiftUI
import PencilKit
import UIKit

// MARK: - SwiftUI Representable

/// A SwiftUI wrapper for `EnhancedPKCanvasView`.
/// Handles binding synchronization and coordinator setup.
struct EnhancedCanvasView: UIViewRepresentable {
    @Binding var canvasView: PKCanvasView
    @ObservedObject var dataRecorder: PencilDataRecorder
    @State private var toolPicker = PKToolPicker()
    
    func makeUIView(context: Context) -> EnhancedPKCanvasView {
        let view = EnhancedPKCanvasView()
        view.delegate = context.coordinator
        view.drawingPolicy = .anyInput
        view.tool = PKInkingTool(.fountainPen, color: .black, width: 8)
        view.dataRecorder = dataRecorder
        
        // Setup Apple Pencil Pro specific features
        view.setupPencilProFeatures()
        
        // Configure tool picker
        // Note: Tool picker logic can be adjusted based on needs.
        // Currently disabling automatic tool picker to rely on simple tool setup.
        // toolPicker.setVisible(true, forFirstResponder: view)
        // toolPicker.addObserver(view)
        // view.becomeFirstResponder()
        
        // Update the binding asynchronously to avoid state modification during view update
        DispatchQueue.main.async {
            self.canvasView = view
        }
        
        return view
    }
    
    func updateUIView(_ uiView: EnhancedPKCanvasView, context: Context) {
        uiView.dataRecorder = dataRecorder
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(dataRecorder: dataRecorder)
    }
    
    // MARK: - Coordinator
    
    class Coordinator: NSObject, PKCanvasViewDelegate {
        let dataRecorder: PencilDataRecorder
        
        init(dataRecorder: PencilDataRecorder) {
            self.dataRecorder = dataRecorder
        }
        
        func canvasViewDrawingDidChange(_ canvasView: PKCanvasView) {
            // Handle drawing changes if necessary
        }
    }
}

// MARK: - Enhanced PKCanvasView

/// A custom `PKCanvasView` subclass that handles high-frequency touch data and hover events.
class EnhancedPKCanvasView: PKCanvasView {
    var dataRecorder: PencilDataRecorder?
    private var hoverGestureRecognizer: UIHoverGestureRecognizer?
    
    // MARK: - Initialization
    
    override func awakeFromNib() {
        super.awakeFromNib()
        setupHoverDetection()
    }
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        setupHoverDetection()
    }
    
    required init?(coder: NSCoder) {
        super.init(coder: coder)
        setupHoverDetection()
    }
    
    // MARK: - Hover Detection
    
    private func setupHoverDetection() {
        if #available(iOS 13.4, *) {
            hoverGestureRecognizer = UIHoverGestureRecognizer(target: self, action: #selector(handleHover(_:)))
            if let hoverGesture = hoverGestureRecognizer {
                addGestureRecognizer(hoverGesture)
            }
        }
    }
    
    @available(iOS 13.4, *)
    @objc private func handleHover(_ gesture: UIHoverGestureRecognizer) {
        let location = gesture.location(in: self)
        let timestamp = ProcessInfo.processInfo.systemUptime
        
        let hoverDistance: CGFloat
        if #available(iOS 17.5, *) {
            hoverDistance = gesture.zOffset
        } else {
            hoverDistance = 12.0 // Fallback/Default value
        }
        
        switch gesture.state {
        case .began, .changed:
            let dataPoint = PencilDataPoint(
                x: location.x,
                y: location.y,
                force: 0,
                azimuth: 0,
                altitude: 0,
                hoverDistance: hoverDistance,
                timestamp: timestamp,
                isHovering: true
            )
            dataRecorder?.addDataPoint(dataPoint)
        default:
            break
        }
    }

    // MARK: - Touch Handling
    
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesBegan(touches, with: event)
        handleTouches(touches, with: event)
    }
    
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        super.touchesMoved(touches, with: event)
        
        // Handle coalesced touches for high-frequency (e.g., 240Hz) data collection
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
        
        var rollAngle: CGFloat = 0
        if #available(iOS 18.0, *) {
            rollAngle = touch.rollAngle
        }
        
        let dataPoint = PencilDataPoint(
            x: location.x,
            y: location.y,
            force: force,
            azimuth: azimuth,
            altitude: altitude,
            hoverDistance: 0,
            timestamp: timestamp,
            isHovering: false,
            rollAngle: rollAngle
        )
        
        dataRecorder?.addDataPoint(dataPoint)
    }
}
