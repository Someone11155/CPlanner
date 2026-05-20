//
// CplannerApp.swift
// Cplanner
//
// Created by iLo on 2026-03-31.
//

import SwiftUI
import Foundation

@main
struct CplannerApp: App {

    init() {
        // Gemma chunked 엔진(CoreML-LLM)의 prefill chunk를 즉시(foreground) 로드.
        // 기본값(defer)은 decode chunk만 먼저 띄우고 prefill을 백그라운드로 미뤄, 로드 직후
        // ~74s 동안 분류가 느린 decode-loop fallback(~12s/건)으로 동작한다. CPlanner는 분류 속도가
        // 핵심이므로 eager 로드로 전환 — 모델 준비까지 ~80s로 늘지만 첫 분류부터 prefill 경로(~1.7s/건).
        // CoreMLLLM.load() 안에서 ProcessInfo 환경변수를 읽으므로 어떤 로드보다 먼저 설정해야 한다.
        setenv("LLM_DEFER_PREFILL", "0", 1)

        // 모델/토크나이저 설치 여부를 확인. 설치돼 있으면 자동으로 백그라운드 로딩이 트리거되고,
        // 없으면 ContentView에서 다운로드 alert를 띄운다.
        ModelInstaller.shared.checkInstallation()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                // Todomate 디자인은 다크 전용. 시스템 라이트 모드에서도 일관된 비주얼.
                .preferredColorScheme(.dark)
        }
        .defaultSize(width: 1000, height: 750)

        // macOS 표준 Preferences/Settings 윈도우 — 메뉴바 `CPlanner > Settings...` 또는 ⌘,로 접근.
        // ContentView 안의 톱니 아이콘이 띄우는 SettingsView (감시 폴더 + 벤치마크)와는 별개 —
        // 여기는 "앱 자체"에 대한 일반 설정 (TipBar 옵션, 분류 임계값, 앱 정보 등).
        Settings {
            AppSettingsView()
                .preferredColorScheme(.dark)
        }
    }
}
