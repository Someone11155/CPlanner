//
// FolderMonitor.swift
// Cplanner
//
// Created by iLo on 2026-03-26.
//

import Foundation
import Combine

class FolderMonitor: ObservableObject {
    private var folderURL: URL
    private var dispatchSource: DispatchSourceFileSystemObject?
    
    var folderDidChange: (() -> Void)?
    
    init(url: URL) {
        self.folderURL = url
    }
    
    func startMonitoring() {
        let fileDescriptor = open(folderURL.path, O_EVTONLY)
        guard fileDescriptor != -1 else { return }
        
        dispatchSource = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fileDescriptor,
            eventMask: .write,
            queue: DispatchQueue.global(qos: .background)
        )
        
        dispatchSource?.setEventHandler { [weak self] in
            DispatchQueue.main.async {
                self?.folderDidChange?()
            }
        }
        
        dispatchSource?.setCancelHandler {
            close(fileDescriptor)
        }
        
        dispatchSource?.resume()
    }
    
    func stopMonitoring() {
        dispatchSource?.cancel()
        dispatchSource = nil
    }
}
