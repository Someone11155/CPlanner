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

/// 폴더 변경 감시기.
///
/// **격리 모델 (이 패턴을 깨면 `_dispatch_assert_queue_fail` SIGTRAP 재발):**
/// - 클래스는 `@MainActor` — 상태(`dispatchSource`, `folderDidChange`)는 main에 격리되고 Sendable.
/// - 단, `DispatchSource`에 넘기는 **event/cancel handler는 반드시 `@Sendable`** (nonisolated). 빌드 설정이
///   `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`라 `@Sendable`를 빼면 핸들러가 `@MainActor`로 추론되고,
///   Swift 6가 그 prologue에 executor 체크(`dispatch_assert_queue(main)`)를 넣는다. dispatch source는
///   핸들러를 background-qos 큐에서 callout하므로 단언 실패 → SIGTRAP. (`@Sendable`면 체크 없음.)
/// - main 격리 상태(`folderDidChange`)는 `@Sendable` 핸들러 안에서 `DispatchQueue.main.async` +
///   `MainActor.assumeIsolated`로 hop 후 접근. `@ObservableObject`는 채택 안 함 (`@Published` 불필요).
@MainActor
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

        // **event/cancel handler 둘 다 `@Sendable` 필수.** 빌드 설정이
        // `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`라서, `@Sendable`를 안 붙이면 이 클로저들이
        // 자동으로 `@MainActor`로 추론된다. Swift 6는 `@MainActor` 클로저 prologue에 executor 체크
        // (`dispatch_assert_queue(main)`)를 삽입하는데, dispatch source는 이 핸들러를 background-qos
        // 큐에서 callout → 단언 실패 → `_dispatch_assert_queue_fail` → SIGTRAP. (크래시는 body 진입
        // 전 prologue에서 발생하므로 안쪽 hop으로는 못 막는다.) `@Sendable`로 nonisolated화해서 체크 제거.
        // 그 뒤 main으로 hop + `MainActor.assumeIsolated`로 `folderDidChange`(MainActor) 호출을 establish.
        source.setEventHandler { @Sendable [weak self] in
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
