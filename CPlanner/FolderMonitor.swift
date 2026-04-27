//
// FolderMonitor.swift
// Cplanner
//
// Created by iLo on 2026-03-26.
//

import Foundation
import Combine
import os

private let folderMonitorLogger = Logger(subsystem: "com.cplanner", category: "FolderMonitor")

class FolderMonitor: ObservableObject {
    private let folderURL: URL
    private var dispatchSource: DispatchSourceFileSystemObject?

    var folderDidChange: (() -> Void)?

    init(url: URL) {
        self.folderURL = url
    }

    deinit {
        dispatchSource?.cancel()
    }

    func startMonitoring() {
        guard dispatchSource == nil else { return }

        let fileDescriptor = open(folderURL.path, O_EVTONLY)
        guard fileDescriptor != -1 else {
            folderMonitorLogger.error("Failed to open folder for monitoring: \(self.folderURL.path, privacy: .public)")
            return
        }

        let source = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fileDescriptor,
            eventMask: .write,
            queue: DispatchQueue.global(qos: .background)
        )

        source.setEventHandler { [weak self] in
            DispatchQueue.main.async { [weak self] in
                self?.folderDidChange?()
            }
        }

        source.setCancelHandler {
            close(fileDescriptor)
        }

        dispatchSource = source
        source.resume()
    }

    func stopMonitoring() {
        dispatchSource?.cancel()
        dispatchSource = nil
    }
}
