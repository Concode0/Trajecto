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

import SwiftUI
import PencilKit
import UniformTypeIdentifiers

/// The main view of the application.
/// Manages the UI for recording, saving, and resetting pencil data samples.
struct ContentView: View {
    @StateObject private var dataRecorder = PencilDataRecorder()
    @Environment(\.dismiss) private var dismiss
    
    // State to hold the PKCanvasView reference for clearing
    @State private var canvasView = PKCanvasView()
    @State private var canvasID = UUID()
    
    @State private var sampleIndex = 1

    var body: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                // Canvas Area
                ZStack {
                    EnhancedCanvasView(canvasView: $canvasView,
                                       dataRecorder: dataRecorder)
                        .id(canvasID)
                        .frame(height: geometry.size.height * 0.9)
                        .background(Color.white)
                        .cornerRadius(10)
                        .shadow(radius: 2)

                    // Recording Indicator
                    if dataRecorder.isRecording {
                        VStack {
                            HStack {
                                Spacer()
                                HStack {
                                    Circle()
                                        .fill(Color.red)
                                        .frame(width: 12, height: 12)
                                        .scaleEffect(dataRecorder.isRecording ? 1 : 0.5)
                                        .animation(.easeInOut(duration: 0.5).repeatForever(), value: dataRecorder.isRecording)
                                    Text("Recording")
                                        .font(.caption)
                                        .foregroundColor(.red)
                                }
                                .padding(.horizontal, 12)
                                .padding(.vertical, 6)
                                .background(Color.black.opacity(0.1))
                                .cornerRadius(20)
                                .padding()
                            }
                            Spacer()
                        }
                    }
                }
                
                // Controls Area
                VStack(spacing: 16) {
                    HStack {
                        Text("Current Sample: #\(sampleIndex)")
                            .font(.headline)
                            .foregroundColor(.primary)
                        Spacer()
                        Text("Points: \(dataRecorder.currentSample?.dataPoints.count ?? 0)")
                            .font(.caption)
                    }
                    .padding(.horizontal)

                    HStack(spacing: 20) {
                        Button(action: {
                            if dataRecorder.isRecording {
                                dataRecorder.stopRecording()
                                let saved = dataRecorder.saveCurrentSampleToFile(index: sampleIndex)
                                if saved {
                                    sampleIndex += 1
                                    clearCanvas()
                                }
                            } else {
                                dataRecorder.startRecording()
                            }
                        }) {
                            Label(dataRecorder.isRecording ? "Save & Next" : "Start",
                                  systemImage: dataRecorder.isRecording ? "arrow.down.doc.fill" : "play.fill")
                                .padding(.horizontal, 40)
                                .padding(.vertical, 16)
                                .background(dataRecorder.isRecording ? Color.blue : Color.green)
                                .foregroundColor(.white)
                                .font(.system(size: 20, weight: .bold))
                                .cornerRadius(25)
                        }

                        Button(action: {
                            dataRecorder.resetData()
                            clearCanvas()
                        }) {
                            Label("Retry", systemImage: "arrow.clockwise")
                                .padding(.horizontal, 20)
                                .padding(.vertical, 16)
                                .background(Color.orange)
                                .foregroundColor(.white)
                                .font(.system(size: 18, weight: .semibold))
                                .cornerRadius(25)
                        }
                        .disabled(!dataRecorder.isRecording && (dataRecorder.currentSample == nil))
                    }

                    VStack(spacing: 8) {
                        Text("Engineered by Kyum | 240Hz Precision Sampling & Physics-based Z-Estimation Engine for 3D Trajectory Research.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .fontWeight(.semibold)
                            .multilineTextAlignment(.center)
                    }
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .frame(height: geometry.size.height * 0.1)
            }
        }
    }

    /// Clears the canvas by resetting the drawing and regenerating the view ID.
    private func clearCanvas() {
        canvasView.drawing = PKDrawing()
        canvasID = UUID()
    }
}

// MARK: - Legacy / Unused

// CSVDocument was defined here but unused in the current file saving logic.
// It might be useful for SwiftUI's .fileExporter in the future.
/*
struct CSVDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.commaSeparatedText] }
    
    var content: String
    
    init(content: String) {
        self.content = content
    }
    
    init(configuration: ReadConfiguration) throws {
        content = ""
    }
    
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        return FileWrapper(regularFileWithContents: content.data(using: .utf8) ?? Data())
    }
}
*/

#Preview {
    ContentView()
}
