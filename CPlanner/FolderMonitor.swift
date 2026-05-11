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

/// 폴더 변경 감시기. **`ObservableObject`를 의도적으로 채택하지 않음** — Swift 6 strict concurrency가
/// `ObservableObject` 클래스를 암시적 `@MainActor`로 추론하면 내부의 모든 클로저(특히
/// `DispatchSource`에 넘기는 cancel/event handler)가 `@MainActor` 상속을 받아, 그 dispatch source가
/// background queue에서 callout을 호출할 때 `_swift_task_checkIsolatedSwift` 어설션이 발동해서
/// `_dispatch_assert_queue_fail`로 SIGTRAP. (실제로 사용자가 폴더 삭제 시 재현된 크래시.)
/// 이 클래스는 `@Published` 프로퍼티가 없어 ObservableObject가 필요하지 않음.
final class FolderMonitor {
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

        // setEventHandler 클로저는 background-qos 큐에서 발화. MainActor isolated된 `folderDidChange`
        // 콜백을 호출하기 전에 main 스레드로 hop 후 `MainActor.assumeIsolated`로 isolation을 명시.
        // (Task @MainActor literal은 Swift 6에서 비결정적으로 outer 클로저 prologue에 isolation 체크를
        //  삽입해 SIGTRAP 발생 — assumeIsolated는 trampoline 없이 직접 isolation establish.)
        // setCancelHandler는 self를 캡처 안 하므로 @Sendable 명시.
        source.setEventHandler { [weak self] in
            DispatchQueue.main.async {
                MainActor.assumeIsolated {
                    self?.folderDidChange?()
                }
            }
        }

        source.setCancelHandler { @Sendable in
            close(fileDescriptor)
        }

        dispatchSource = source
        source.resume()
    }

    func stopMonitoring() {
        // nil-out first, hold local strong ref for cancel — avoids re-entry / double-cancel races.
        guard let source = dispatchSource else { return }
        dispatchSource = nil
        source.cancel()
    }
}
