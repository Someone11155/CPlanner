//
// CplannerApp.swift
// Cplanner
//
// Created by iLo on 2026-03-31.
//

import SwiftUI

@main
struct CplannerApp: App {

    init() {
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
