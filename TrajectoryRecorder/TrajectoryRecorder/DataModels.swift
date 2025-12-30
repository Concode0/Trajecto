//
//  DataModels.swift
//  TrajectoryRecorder
//
//  Created by haro on 7/17/25.
//

import Foundation
import CoreGraphics

/// Represents a single point of data recorded from the Apple Pencil.
/// Contains position, force, tilt, and other sensor data.
struct PencilDataPoint {
    let x: CGFloat
    let y: CGFloat
    let force: CGFloat
    let azimuth: CGFloat
    let altitude: CGFloat
    let hoverDistance: CGFloat
    let timestamp: TimeInterval
    let isHovering: Bool
    let rollAngle: CGFloat // For Apple Pencil Pro
    
    init(x: CGFloat, y: CGFloat, force: CGFloat, azimuth: CGFloat, altitude: CGFloat, hoverDistance: CGFloat, timestamp: TimeInterval, isHovering: Bool, rollAngle: CGFloat = 0) {
        self.x = x
        self.y = y
        self.force = force
        self.azimuth = azimuth
        self.altitude = altitude
        self.hoverDistance = hoverDistance
        self.timestamp = timestamp
        self.isHovering = isHovering
        self.rollAngle = rollAngle
    }
}

/// Manages the recording of Apple Pencil data.
/// Handles session state, data accumulation, and file export.
class PencilDataRecorder: ObservableObject {
    
    /// Represents a continuous recording session.
    struct Sample: Identifiable {
        let id = UUID()
        let startTime: TimeInterval
        var dataPoints: [PencilDataPoint] = []
    }

    /// Published state indicating if recording is currently active.
    @Published var isRecording = false
    
    /// The current data sample being recorded or recently finished.
    @Published private(set) var currentSample: Sample?
    
    private var recordingStartTime: TimeInterval = 0

    // MARK: - Recording Control
    
    /// Starts a new recording session.
    func startRecording() {
        guard !isRecording else { return }
        isRecording = true
        
        recordingStartTime = ProcessInfo.processInfo.systemUptime
        currentSample = Sample(startTime: recordingStartTime)
    }
    
    /// Stops the current recording session.
    func stopRecording() {
        guard isRecording else { return }
        isRecording = false
    }
    
    /// Resets all recording data and state.
    func resetData() {
        isRecording = false
        currentSample = nil
        recordingStartTime = 0
    }

    /// Adds a new data point to the current sample.
    /// - Parameter point: The raw data point from the pencil event.
    func addDataPoint(_ point: PencilDataPoint) {
        guard isRecording, var sample = currentSample else { return }

        // Calculate time delta relative to the start of recording
        let deltaFromStart = point.timestamp - recordingStartTime

        let adjustedPoint = PencilDataPoint(
            x: point.x,
            y: point.y,
            force: point.force,
            azimuth: point.azimuth,
            altitude: point.altitude,
            hoverDistance: point.hoverDistance,
            timestamp: deltaFromStart,
            isHovering: point.isHovering,
            rollAngle: point.rollAngle
        )
        
        sample.dataPoints.append(adjustedPoint)
        currentSample = sample
    }
    
    // MARK: - Save to File
    
    /// Saves the current sample to a CSV file in the documents directory.
    /// - Parameter index: The sample index to append to the filename.
    /// - Returns: `true` if saving was successful, `false` otherwise.
    func saveCurrentSampleToFile(index: Int) -> Bool {
        guard let sample = currentSample, !sample.dataPoints.isEmpty else {
            return false
        }
        
        var csv = "timestamp,x,y,force,azimuth,altitude,hoverDistance,isHovering,rollAngle\n"
        
        for point in sample.dataPoints {
            csv += String(format: "%.6f,%.8f,%.8f,%.4f,%.4f,%.4f,%.4f,%d,%.4f\n",
                          point.timestamp,
                          point.x, point.y,
                          point.force,
                          point.azimuth, point.altitude,
                          point.hoverDistance,
                          point.isHovering ? 1 : 0,
                          point.rollAngle)
        }
        
        let fileName = "Sample_\(index).csv"
        
        if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = dir.appendingPathComponent(fileName)
            
            do {
                try csv.write(to: fileURL, atomically: true, encoding: .utf8)
                print("Saved to: \(fileURL.path)")
                
                resetData()
                return true
            } catch {
                print("Error saving file: \(error)")
                return false
            }
        }
        return false
    }
}
