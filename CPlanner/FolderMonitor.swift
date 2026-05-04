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

        // 이 클래스가 더 이상 ObservableObject가 아니라 @MainActor 추론을 받지 않으므로
        // 두 클로저 모두 기본 isolation은 nonisolated. setCancelHandler는 self를 캡처 안 하므로
        // @Sendable 명시 가능 (future-proof). setEventHandler는 weak self를 캡처해 main으로 hop하는
        // 패턴이라 @Sendable 명시는 non-Sendable 캡처 경고를 일으키므로 생략 — 어차피 클래스 자체가
        // 더 이상 @MainActor가 아니라 dispatch source의 callout이 background queue에서 안전.
        source.setEventHandler { [weak self] in
            DispatchQueue.main.async {
                self?.folderDidChange?()
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
