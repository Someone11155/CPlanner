//
// ContentView.swift
// Cplanner
//
// Created by iLo on 2026-03-26.
//

import SwiftUI
import Combine
import CoreML
import EventKit
import os

private let taskManagerLogger = Logger(subsystem: "com.cplanner", category: "TaskManager")

// --- 1. 모델 정의 ---
struct TaskItem: Identifiable, Codable, Equatable {
    var id = UUID()
    var title: String
    var isCompleted: Bool = false
    var targetFolder: String
    var date: Date
    var eventIdentifier: String? = nil
    /// 분류 신뢰도 0.0~1.0. nil이면 사용자 수동 분류 또는 분류 결과 없음(placeholder).
    /// 1.0 = 정확매치 메모이제이션, 그 외는 알파벳 logits softmax 결과.
    var classificationConfidence: Double? = nil
}

struct Correction: Codable, Equatable, Sendable {
    var taskTitle: String
    var folderName: String
}

struct FolderRule: Identifiable, Codable {
    var id = UUID()
    var folderName: String
    var bookmark: Data

    /// Resolves the security-scoped bookmark.
    /// Caller MUST call `startAccessingSecurityScopedResource()` on the
    /// returned URL before reading the folder, and the matching
    /// `stopAccessingSecurityScopedResource()` once finished.
    func resolveURL() throws -> (url: URL, isStale: Bool) {
        var isStale = false
        let url = try URL(
            resolvingBookmarkData: bookmark,
            options: [.withSecurityScope],
            relativeTo: nil,
            bookmarkDataIsStale: &isStale
        )
        return (url, isStale)
    }
}

// --- 2. 메인 매니저 ---
class TaskManager: ObservableObject {
    private static let tasksKey = "cplanner.tasks"
    private static let folderRulesKey = "cplanner.folderRules"
    private static let correctionsKey = "cplanner.corrections"

    @Published var tasks: [TaskItem] = [] {
        didSet { persist(tasks, forKey: Self.tasksKey) }
    }
    @Published var folderRules: [FolderRule] = [] {
        didSet { persist(folderRules, forKey: Self.folderRulesKey) }
    }
    @Published var corrections: [Correction] = [] {
        didSet { persist(corrections, forKey: Self.correctionsKey) }
    }
    @Published var lastCalendarError: String?
    /// 시스템 공휴일 캘린더(예: 대한민국의 공휴일)에서 가져온 공휴일 정보.
    /// 키는 `Calendar.current.startOfDay(for:)` 정규화된 Date, 값은 이벤트 title(예: "어린이날").
    @Published var holidayNames: [Date: String] = [:]

    private let eventStore = EKEventStore()
    private var monitors: [UUID: (monitor: FolderMonitor, scopedURL: URL)] = [:]
    private var calendarAccessGranted = false
    private var cplannerCalendar: EKCalendar?
    private var isApplyingLocalChange = false
    nonisolated(unsafe) private var calendarChangeObserver: NSObjectProtocol?

    init() {
        // Load persisted state. didSet observers do NOT fire during init,
        // so these assignments don't write back to UserDefaults.
        if let data = UserDefaults.standard.data(forKey: Self.tasksKey),
           let decoded = try? JSONDecoder().decode([TaskItem].self, from: data) {
            self.tasks = decoded
        }
        if let data = UserDefaults.standard.data(forKey: Self.folderRulesKey),
           let decoded = try? JSONDecoder().decode([FolderRule].self, from: data) {
            self.folderRules = decoded
            for rule in self.folderRules {
                startMonitoring(rule: rule)
            }
        }
        if let data = UserDefaults.standard.data(forKey: Self.correctionsKey),
           let decoded = try? JSONDecoder().decode([Correction].self, from: data) {
            self.corrections = decoded
        }
        Task { [weak self] in
            await self?.bootstrapCalendar()
        }
    }

    deinit {
        if let observer = calendarChangeObserver {
            NotificationCenter.default.removeObserver(observer)
        }
        for entry in monitors.values {
            entry.scopedURL.stopAccessingSecurityScopedResource()
        }
    }

    private func persist<T: Encodable>(_ value: T, forKey key: String) {
        do {
            let data = try JSONEncoder().encode(value)
            UserDefaults.standard.set(data, forKey: key)
        } catch {
            taskManagerLogger.error("Failed to persist \(key, privacy: .public): \(error.localizedDescription, privacy: .public)")
        }
    }

    // 할 일 추가 (AI 분류 적용)
    func addTask(title: String, date: Date) {
        // 1) 즉시 placeholder 폴더로 UI 반영
        let placeholder = TaskItem(title: title, targetFolder: "분류 중…", date: date)
        let taskID = placeholder.id
        tasks.append(placeholder)

        // 2) 백그라운드: 분류 + 캘린더 저장 + task 갱신
        let folderNames = self.folderRules.map { $0.folderName }
        let activeCorrections = Array(self.corrections.filter { folderNames.contains($0.folderName) }.suffix(5))
        Task { [weak self] in
            guard let self else { return }
            let (detectedFolder, confidence) = await LocalLLMService.shared.classifyTask(taskTitle: title, availableFolders: folderNames, corrections: activeCorrections)
            if let idx = self.tasks.firstIndex(where: { $0.id == taskID }), self.tasks[idx].targetFolder == "분류 중…" {
                // 사용자가 그동안 폴더를 직접 골랐으면 자동 분류 결과로 덮어쓰지 않음
                self.tasks[idx].targetFolder = detectedFolder
                self.tasks[idx].classificationConfidence = confidence
            }

            if self.calendarAccessGranted, let cal = self.cplannerCalendar {
                let event = EKEvent(eventStore: self.eventStore)
                event.title = title
                let day = Calendar.current.startOfDay(for: date)
                event.startDate = day
                event.endDate = day
                event.isAllDay = true
                event.alarms = [] // 캘린더 source의 기본 alarm이 URL을 포함해 sandbox 경고 발생 — 명시적 클리어
                event.calendar = cal

                self.isApplyingLocalChange = true
                do {
                    try self.eventStore.save(event, span: .thisEvent)
                    if let idx = self.tasks.firstIndex(where: { $0.id == taskID }) {
                        self.tasks[idx].eventIdentifier = event.eventIdentifier
                    }
                    self.lastCalendarError = nil
                } catch {
                    self.lastCalendarError = "캘린더 저장 실패: \(error.localizedDescription)"
                    taskManagerLogger.error("Calendar save failed: \(error.localizedDescription, privacy: .public)")
                }
                DispatchQueue.main.async { [weak self] in
                MainActor.assumeIsolated { self?.isApplyingLocalChange = false }
            }
            }
        }
    }

    /// 사용자가 직접 폴더를 변경 — task에 반영 + 같은 제목의 기존 correction 대체 후 추가.
    /// 다음 분류부터 in-context few-shot 예시로 사용된다.
    func userPickedFolder(taskID: UUID, folder: String) {
        guard let idx = tasks.firstIndex(where: { $0.id == taskID }) else { return }
        tasks[idx].targetFolder = folder
        // 사용자가 직접 고른 결과는 신뢰도 표시에서 제외 (% 표시 안 함)
        tasks[idx].classificationConfidence = nil
        let title = tasks[idx].title
        corrections.removeAll { $0.taskTitle == title }
        corrections.append(Correction(taskTitle: title, folderName: folder))
        // 안전망 — 사용자 도달 비현실적 한계지만 버그/자동누적 폭주 시 차단.
        // 50,000 = 광적 사용 (50건/일 × 27년)에서나 도달. 도달해도 plist ~6MB라 실제 동작 가능.
        if corrections.count > 50_000 {
            corrections.removeFirst(corrections.count - 50_000)
        }
    }

    func deleteTask(id: UUID) {
        guard let idx = tasks.firstIndex(where: { $0.id == id }) else { return }
        let task = tasks[idx]
        if let eid = task.eventIdentifier,
           calendarAccessGranted,
           let event = eventStore.event(withIdentifier: eid) {
            isApplyingLocalChange = true
            do {
                try eventStore.remove(event, span: .thisEvent)
            } catch {
                taskManagerLogger.error("Calendar delete failed: \(error.localizedDescription, privacy: .public)")
            }
            DispatchQueue.main.async { [weak self] in
                MainActor.assumeIsolated { self?.isApplyingLocalChange = false }
            }
        }
        tasks.remove(at: idx)
    }

    // 감시 규칙 추가
    func addRule(name: String, bookmark: Data) {
        let newRule = FolderRule(folderName: name, bookmark: bookmark)
        folderRules.append(newRule)
        startMonitoring(rule: newRule)
    }

    func deleteRule(id: UUID) {
        // Order matters — remove from monitors dict FIRST so any in-flight
        // handleNewFileDetected Task that's about to access monitors[id]?.scopedURL
        // gets nil and bails. Otherwise it can race with stopAccessingSecurityScopedResource()
        // and trigger a dispatch_assert_queue_fail crash in Foundation.
        let entry = monitors.removeValue(forKey: id)
        folderRules.removeAll { $0.id == id }
        if let entry {
            entry.monitor.stopMonitoring()
            entry.scopedURL.stopAccessingSecurityScopedResource()
        }
    }

    // 폴더 감시 시작
    private func startMonitoring(rule: FolderRule) {
        do {
            let (url, isStale) = try rule.resolveURL()
            if isStale {
                taskManagerLogger.warning("Bookmark for \(rule.folderName, privacy: .public) is stale; user should reselect the folder.")
            }
            guard url.startAccessingSecurityScopedResource() else {
                taskManagerLogger.error("Failed to start security-scoped access for \(url.path, privacy: .public)")
                return
            }
            let monitor = FolderMonitor(url: url)
            let ruleID = rule.id
            monitor.folderDidChange = { [weak self] in
                self?.handleNewFileDetected(ruleID: ruleID)
            }
            monitor.startMonitoring()
            monitors[ruleID] = (monitor: monitor, scopedURL: url)
        } catch {
            taskManagerLogger.error("Failed to resolve bookmark for \(rule.folderName, privacy: .public): \(error.localizedDescription, privacy: .public)")
        }
    }

    // 파일 감지 시 AI 검증 로직 실행
    private func handleNewFileDetected(ruleID: UUID) {
        guard let rule = folderRules.first(where: { $0.id == ruleID }),
              let scopedURL = monitors[ruleID]?.scopedURL else { return }

        Task { [weak self] in
            guard let self else { return }
            let contents: [URL]
            do {
                contents = try FileManager.default.contentsOfDirectory(
                    at: scopedURL,
                    includingPropertiesForKeys: [.contentModificationDateKey],
                    options: [.skipsHiddenFiles]
                )
            } catch {
                taskManagerLogger.error("Failed to enumerate folder \(scopedURL.path, privacy: .public): \(error.localizedDescription, privacy: .public)")
                return
            }

            let newest = contents.max { a, b in
                let aDate = (try? a.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let bDate = (try? b.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return aDate < bDate
            }
            guard let fileName = newest?.lastPathComponent else { return }

            let isValid = LocalLLMService.shared.validateFileContext(fileName: fileName, folderName: rule.folderName)
            guard isValid else { return }

            if let idx = self.tasks.firstIndex(where: { $0.targetFolder == rule.folderName && !$0.isCompleted }) {
                self.tasks[idx].isCompleted = true
            }
        }
    }

    private func bootstrapCalendar() async {
        do {
            let granted = try await eventStore.requestFullAccessToEvents()
            calendarAccessGranted = granted
            guard granted else {
                lastCalendarError = "캘린더 접근 권한이 거부되었습니다."
                return
            }
            guard let cal = getOrCreateCPlannerCalendar() else { return }
            cplannerCalendar = cal
            subscribeToCalendarChanges()
            refreshHolidays()
            await syncFromCalendar()
        } catch {
            lastCalendarError = "캘린더 권한 요청 실패: \(error.localizedDescription)"
            taskManagerLogger.error("Calendar access request failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    private func getOrCreateCPlannerCalendar() -> EKCalendar? {
        if let existing = eventStore.calendars(for: .event).first(where: { $0.title == "CPlanner" }) {
            return existing
        }
        let source: EKSource? = eventStore.sources.first(where: { $0.sourceType == .calDAV })
            ?? eventStore.sources.first(where: { $0.sourceType == .local })
        guard let source else {
            lastCalendarError = "캘린더 source를 찾을 수 없습니다."
            taskManagerLogger.error("No suitable EKSource for CPlanner calendar")
            return nil
        }
        let cal = EKCalendar(for: .event, eventStore: eventStore)
        cal.title = "CPlanner"
        cal.source = source
        cal.cgColor = NSColor.systemBlue.cgColor
        do {
            try eventStore.saveCalendar(cal, commit: true)
            return cal
        } catch {
            lastCalendarError = "CPlanner 캘린더 생성 실패: \(error.localizedDescription)"
            taskManagerLogger.error("Failed to create CPlanner calendar: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    private func subscribeToCalendarChanges() {
        calendarChangeObserver = NotificationCenter.default.addObserver(
            forName: .EKEventStoreChanged,
            object: eventStore,
            queue: nil
        ) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                if self.isApplyingLocalChange { return }
                self.refreshHolidays()
                await self.syncFromCalendar()
            }
        }
    }

    /// 시스템 공휴일 캘린더에서 가시 범위(±2년)의 공휴일 날짜를 추출하여 `holidayDates`에 저장.
    /// 매칭은 캘린더 title 키워드 기반: "공휴일", "휴일", "holiday", "holidays".
    /// all-day 이벤트는 endDate가 다음 날 00:00(exclusive end)이라 `< endDay`로 순회.
    func refreshHolidays() {
        let cals = eventStore.calendars(for: .event).filter { cal in
            let lowered = cal.title.lowercased()
            return cal.title.contains("공휴일") || cal.title.contains("휴일")
                || lowered.contains("holiday") || lowered.contains("holidays")
        }
        guard !cals.isEmpty else {
            holidayNames = [:]
            return
        }
        let calendar = Calendar.current
        let now = Date()
        let start = calendar.date(byAdding: .month, value: -6, to: now) ?? now
        let end = calendar.date(byAdding: .month, value: 24, to: now) ?? now
        let predicate = eventStore.predicateForEvents(withStart: start, end: end, calendars: cals)
        let events = eventStore.events(matching: predicate)
        var names = [Date: String]()
        for event in events {
            let title = event.title ?? ""
            let startDay = calendar.startOfDay(for: event.startDate)
            let endDay = calendar.startOfDay(for: event.endDate)
            if startDay >= endDay {
                names[startDay] = title
                continue
            }
            var cursor = startDay
            while cursor < endDay {
                names[cursor] = title
                guard let next = calendar.date(byAdding: .day, value: 1, to: cursor), next > cursor else { break }
                cursor = next
            }
        }
        holidayNames = names
    }

    /// 주어진 날짜가 공휴일이면 true. 비교는 `startOfDay` 기준.
    func isHoliday(_ date: Date) -> Bool {
        holidayName(for: date) != nil
    }

    /// 주어진 날짜가 공휴일이면 이름(예: "어린이날")을 반환, 아니면 nil.
    func holidayName(for date: Date) -> String? {
        let day = Calendar.current.startOfDay(for: date)
        return holidayNames[day]
    }

    private func syncFromCalendar() async {
        guard calendarAccessGranted, let cal = cplannerCalendar else { return }
        let calendar = Calendar.current
        let now = Date()
        let start = calendar.date(byAdding: .month, value: -6, to: now) ?? now
        let end = calendar.date(byAdding: .month, value: 24, to: now) ?? now
        let predicate = eventStore.predicateForEvents(withStart: start, end: end, calendars: [cal])
        let events = eventStore.events(matching: predicate)

        var resultTasks = tasks
        var seenEventIds = Set<String>()
        var taskIdxByEventId: [String: Int] = [:]
        for (i, t) in resultTasks.enumerated() {
            if let eid = t.eventIdentifier { taskIdxByEventId[eid] = i }
        }

        for event in events {
            guard !event.hasRecurrenceRules else {
                taskManagerLogger.warning("Skipping recurring event: \(event.title ?? "(제목 없음)", privacy: .public)")
                continue
            }
            guard let eid = event.eventIdentifier else { continue }
            seenEventIds.insert(eid)
            let normalizedDate = calendar.startOfDay(for: event.startDate)
            let title = event.title ?? "(제목 없음)"

            if let idx = taskIdxByEventId[eid] {
                if resultTasks[idx].title != title { resultTasks[idx].title = title }
                if !calendar.isDate(resultTasks[idx].date, inSameDayAs: normalizedDate) {
                    resultTasks[idx].date = normalizedDate
                }
            } else {
                let folder: String
                let confidence: Double?
                if folderRules.count >= 2 {
                    let names = folderRules.map { $0.folderName }
                    let result = await LocalLLMService.shared.classifyTask(taskTitle: title, availableFolders: names)
                    folder = result.folder
                    confidence = result.confidence
                } else {
                    folder = "(미분류)"
                    confidence = nil
                }
                let newTask = TaskItem(title: title, targetFolder: folder, date: normalizedDate, eventIdentifier: eid, classificationConfidence: confidence)
                resultTasks.append(newTask)
            }
        }

        resultTasks.removeAll { task in
            guard let eid = task.eventIdentifier else { return false }
            return !seenEventIds.contains(eid)
        }

        if resultTasks != tasks {
            tasks = resultTasks
        }
    }
}

// --- 3. 커스텀 캘린더 뷰 ---
struct CustomCalendarView: View {
    @ObservedObject var taskManager: TaskManager
    @Binding var selectedDate: Date

    private let calendar = Calendar.current
    let weekdayNames = ["일", "월", "화", "수", "목", "금", "토"]

    /// 다음 달=trailing(오른쪽에서 들어옴), 이전 달=leading. 월 변경 시 grid가 swipe + fade로 교체.
    @State private var lastMonthDirection: Edge = .trailing

    func generateDaysInMonth() -> [Date] {
        guard let monthRange = calendar.range(of: .day, in: .month, for: selectedDate),
              let firstOfMonth = calendar.date(from: calendar.dateComponents([.year, .month], from: selectedDate)) else { return [] }
        let firstWeekday = calendar.component(.weekday, from: firstOfMonth)
        let daysToPrepend = firstWeekday - 1
        var days: [Date] = []
        for i in 0..<daysToPrepend {
            if let date = calendar.date(byAdding: .day, value: -i-1, to: firstOfMonth) { days.append(date) }
        }
        days.reverse()
        for i in 0..<monthRange.count {
            if let date = calendar.date(byAdding: .day, value: i, to: firstOfMonth) { days.append(date) }
        }
        return days
    }

    var body: some View {
        VStack(spacing: 0) {
            // 월/년 헤더 + prev/next chevrons
            HStack {
                Text(selectedDate, formatter: DateFormatter.monthYearFormatter)
                    .font(DesignFont.calendarMonth())
                    .foregroundColor(.tdmInkPrimary)
                Spacer()
                HStack(spacing: DesignSpacing.sm) {
                    Button(action: { changeMonth(value: -1) }) {
                        Image(systemName: "chevron.left")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundColor(.tdmInkSecondary)
                    }
                    Button(action: { changeMonth(value: 1) }) {
                        Image(systemName: "chevron.right")
                            .font(.system(size: 18, weight: .semibold))
                            .foregroundColor(.tdmInkSecondary)
                    }
                }.buttonStyle(.plain)
            }
            .padding(.horizontal, DesignSpacing.md)
            .padding(.vertical, DesignSpacing.md)

            // 요일 행 — 일=빨강 / 토=파랑 / 평일=흰.
            // 아래 LazyVGrid와 동일한 horizontal padding/컬럼 spec을 줘서 세로 정렬을 맞춤.
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 0), count: 7), spacing: 0) {
                ForEach(weekdayNames, id: \.self) { weekday in
                    Text(weekday)
                        .font(DesignFont.caption(13))
                        .foregroundColor(
                            weekday == "일" ? .tdmDateSunday :
                            weekday == "토" ? .tdmDateSaturday : .tdmInkPrimary
                        )
                        .frame(maxWidth: .infinity)
                        .frame(height: 28)
                }
            }
            .padding(.horizontal, DesignSpacing.xs)

            // 날짜 그리드 — Todomate squircle 셀. 월 변경 시 좌우 슬라이드 + 페이드 transition.
            // .id(monthKey)로 월 바뀔 때 view 재생성을 트리거 → SwiftUI가 transition 적용.
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 0), count: 7), spacing: 6) {
                ForEach(generateDaysInMonth(), id: \.self) { date in
                    dayCell(for: date)
                }
            }
            .padding(.horizontal, DesignSpacing.xs)
            .padding(.top, DesignSpacing.xs)
            .id(monthIdentityKey)
            .transition(monthSlideTransition)

            Spacer()
        }
        .background(Color.tdmCanvas)
    }

    /// 단일 날짜 셀. 4가지 상태 조합:
    /// - 선택: 흰 원 (active inversion) — 가장 강한 emphasis
    /// - 오늘 (선택 아님): 흰 ring (테두리만)
    /// - 이벤트 있음: 노란 squircle 채움
    /// - 기본: 어두운 squircle 채움
    /// 숫자는 셀 안에 — 단, 선택/오늘일 때는 검정/흰 (배경에 따라), 기본일 때는 weekday 색.
    @ViewBuilder
    private func dayCell(for date: Date) -> some View {
        let weekday = calendar.component(.weekday, from: date)
        let day = calendar.component(.day, from: date)
        let isSelected = calendar.isDate(date, inSameDayAs: selectedDate)
        let isToday = calendar.isDateInToday(date)
        let inSameMonth = calendar.isDate(date, equalTo: selectedDate, toGranularity: .month)
        let dots = dotCounts(for: date)
        let totalDots = dots.total
        let completedDots = dots.completed

        let isHoliday = taskManager.isHoliday(date)
        let weekdayColor: Color =
            !inSameMonth ? Color.tdmInkTertiary :
            (weekday == 1 || isHoliday) ? .tdmDateSunday :
            weekday == 7 ? .tdmDateSaturday : .tdmInkPrimary

        VStack(spacing: 3) {
            ZStack {
                // 셀 squircle — 선택일은 흰, 그 외는 어두운 (이벤트 분기 제거됨; 이벤트는 점 row로만 표시)
                if isSelected {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color.tdmCapsuleActive)
                } else {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(inSameMonth ? Color.tdmCapsule : Color.tdmCapsule.opacity(0.4))
                }

                // 오늘 표시 — 선택 아닐 때만 흰 ring
                if isToday && !isSelected {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .stroke(Color.tdmInkPrimary, lineWidth: 1.5)
                }

                // 숫자
                Text("\(day)")
                    .font(DesignFont.calendarDate(13))
                    .foregroundColor(isSelected ? .tdmInkOnLight : weekdayColor)
            }
            .frame(width: 32, height: 32)

            // 이벤트 인디케이터 — 왼쪽부터 체크(완료) → 점(미완료). cap 3.
            HStack(spacing: 3) {
                ForEach(0..<totalDots, id: \.self) { idx in
                    if idx < completedDots {
                        Image(systemName: "checkmark")
                            .font(.system(size: 6, weight: .black))
                            .foregroundColor(inSameMonth ? Color.tdmInkPrimary : Color.tdmInkTertiary)
                            .frame(width: 6, height: 6)
                    } else {
                        Circle()
                            .fill(inSameMonth ? Color.tdmInkPrimary : Color.tdmInkTertiary)
                            .frame(width: 4, height: 4)
                    }
                }
            }
            .frame(height: 6)
        }
        .frame(maxWidth: .infinity)
        .contentShape(Rectangle())
        .onTapGesture {
            withAnimation(.easeInOut(duration: 0.15)) {
                self.selectedDate = date
            }
        }
    }

    func changeMonth(value: Int) {
        lastMonthDirection = value > 0 ? .trailing : .leading
        if let newDate = calendar.date(byAdding: .month, value: value, to: selectedDate) {
            withAnimation(.easeInOut(duration: 0.32)) {
                selectedDate = newDate
            }
        }
    }

    /// 현재 표시 중인 월의 고유 key. 월/년 바뀌면 이 값도 바뀌어 .id로 view 재생성 트리거.
    private var monthIdentityKey: String {
        let comps = calendar.dateComponents([.year, .month], from: selectedDate)
        return "\(comps.year ?? 0)-\(comps.month ?? 0)"
    }

    /// 좌우 슬라이드 + 페이드 — `lastMonthDirection`에 따라 들어오는 방향, 빠지는 방향 반대로.
    private var monthSlideTransition: AnyTransition {
        let inEdge = lastMonthDirection
        let outEdge: Edge = (lastMonthDirection == .trailing) ? .leading : .trailing
        return .asymmetric(
            insertion: .move(edge: inEdge).combined(with: .opacity),
            removal:   .move(edge: outEdge).combined(with: .opacity)
        )
    }
    func hasTasks(date: Date) -> Bool { taskManager.tasks.contains { calendar.isDate($0.date, inSameDayAs: date) } }
    func taskCount(for date: Date) -> Int { taskManager.tasks.lazy.filter { calendar.isDate($0.date, inSameDayAs: date) }.count }
    func completedCount(for date: Date) -> Int {
        taskManager.tasks.lazy.filter { calendar.isDate($0.date, inSameDayAs: date) && $0.isCompleted }.count
    }

    /// 셀 아래 점/체크 개수 결정.
    /// - 할 일 ≤ 2개: 1:1 매핑 (할 일 1개 + 완료 1개 = 체크 1, 할 일 2개 + 완료 1개 = 체크 1 / 점 1)
    /// - 할 일 ≥ 3개: 비율 기반 — `ceil(completed/total × 3)`. completed 1+면 자동으로 첫 체크 표시 (ceil 효과),
    ///   비율이 1/3 / 2/3 / 1.0 임계값을 넘을 때마다 2번째 / 3번째 체크로 전환.
    /// 정수 ceiling: `ceil(a×c/b) = (a×c + b − 1) / b`.
    func dotCounts(for date: Date) -> (total: Int, completed: Int) {
        let total = taskCount(for: date)
        let completed = completedCount(for: date)
        let totalDots = min(total, 3)
        let completedDots: Int
        if total == 0 {
            completedDots = 0
        } else if total >= 3 {
            completedDots = min((completed * 3 + total - 1) / total, totalDots)
        } else {
            completedDots = min(completed, totalDots)
        }
        return (totalDots, completedDots)
    }
}

// --- 3-1. 하단 팁 바 ---
struct TipBar: View {
    static let tips: [String] = [
        "분류 결과를 수정하려면 폴더 아이콘을 클릭하세요",
        "수동으로 분류하면 다음 자동 분류의 학습 예시로 사용돼요",
        "캘린더에서 추가한 일정도 자동으로 동기화돼요",
        "AI 분류는 인터넷 없이 기기 안에서 동작해요",
        "설정에서 폴더 감시 규칙을 추가할 수 있어요",
        "감시 폴더에 파일이 들어오면 할 일이 자동으로 완료돼요",
        "달력의 작은 점은 그날 할 일이 있다는 뜻이에요",
        "할 일을 우클릭하면 삭제할 수 있어요",
        "폴더 옆 % 숫자는 AI의 자신감 — 낮으면 직접 확인해보세요",
        "신뢰도 75% 미만이면 자동으로 '일반' 폴더로 분류돼요"
    ]

    @AppStorage("cplanner.app.tip.intervalSeconds") private var tipIntervalSeconds: Double = 7
    @State private var currentTip: String = TipBar.tips.randomElement() ?? ""
    @State private var tipOpacity: Double = 1.0

    /// fade out (350ms easeIn) → 텍스트 swap → fade in (450ms easeOut). 시퀀셜 cross-fade라
    /// 단순 `.transition(.opacity)`보다 좀 더 부드럽게 인지됨.
    private let fadeOutDuration: Double = 0.35
    private let fadeInDuration: Double = 0.45

    var body: some View {
        HStack(spacing: DesignSpacing.xs) {
            Text("💡")
            Text(currentTip)
                .font(DesignFont.bodySmall())
                .foregroundColor(.tdmInkBio)
            Spacer()
        }
        .opacity(tipOpacity)
        .padding(.horizontal, DesignSpacing.md)
        .padding(.vertical, DesignSpacing.xs)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.tdmCapsule)
        // AppSettings에서 주기 변경 시 즉시 반영되도록 .id로 timer 재구독 강제.
        .modifier(TipTimerModifier(intervalSeconds: tipIntervalSeconds, onTick: rotateTip))
    }

    private func rotateTip() {
        guard TipBar.tips.count > 1 else { return }
        var next = TipBar.tips.randomElement() ?? currentTip
        while next == currentTip {
            next = TipBar.tips.randomElement() ?? currentTip
        }
        withAnimation(.easeIn(duration: fadeOutDuration)) {
            tipOpacity = 0
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + fadeOutDuration) {
            MainActor.assumeIsolated {
                currentTip = next
                withAnimation(.easeOut(duration: fadeInDuration)) {
                    tipOpacity = 1
                }
            }
        }
    }
}

/// 주기를 동적으로 변경 가능한 timer modifier — AppStorage 변경 시 .id로 재구독.
private struct TipTimerModifier: ViewModifier {
    let intervalSeconds: Double
    let onTick: () -> Void

    func body(content: Content) -> some View {
        content.onReceive(
            Timer.publish(every: intervalSeconds, on: .main, in: .common).autoconnect()
        ) { _ in onTick() }
        .id(intervalSeconds)
    }
}

// --- 4. 메인 UI 화면 ---
struct ContentView: View {
    @StateObject private var taskManager = TaskManager()
    @StateObject private var modelInstaller = ModelInstaller.shared
    @State private var selectedDate = Date()
    @State private var isAddingTask = false
    /// 외부 패널(isAddingTask)과 분리해 inner content 애니메이션 타이밍을 제어.
    /// close 시: showAddContent 먼저(120ms easeOut) → 80ms 뒤 isAddingTask false(350ms easeInOut).
    /// open 시: isAddingTask 먼저 → 150ms 뒤 showAddContent.
    @State private var showAddContent = false
    @State private var showSettings = false
    @State private var newTaskTitle = ""
    /// 앱 자체 Settings (⌘,)에서 토글되는 TipBar 표시 여부. AppStorage로 영속.
    @AppStorage("cplanner.app.tip.enabled") private var tipEnabled: Bool = true

    private func presentAddTask() {
        withAnimation(.easeInOut(duration: 0.3)) {
            isAddingTask = true
        }
        withAnimation(.easeIn(duration: 0.2).delay(0.15)) {
            showAddContent = true
        }
    }

    private func dismissAddTask() {
        withAnimation(.easeOut(duration: 0.12)) {
            showAddContent = false
        }
        withAnimation(.easeInOut(duration: 0.35).delay(0.08)) {
            isAddingTask = false
        }
    }

    private func toggleAddTask() {
        if isAddingTask { dismissAddTask() } else { presentAddTask() }
    }

    private func submitAddTask() {
        guard !newTaskTitle.isEmpty else { return }
        taskManager.addTask(title: newTaskTitle, date: selectedDate)
        newTaskTitle = ""
        dismissAddTask()
    }

    private func toggleSettings() {
        withAnimation(.easeInOut(duration: 0.2)) {
            showSettings.toggle()
        }
    }

    private func dismissSettings() {
        withAnimation(.easeInOut(duration: 0.2)) {
            showSettings = false
        }
    }

    /// List 안 task row — 삭제 시 페이드아웃 transition + withAnimation으로 부드럽게 사라짐.
    @ViewBuilder
    private func taskRow(task: TaskItem, index: Int) -> some View {
        HStack(spacing: DesignSpacing.sm) {
            ZStack {
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(Color.tdmCapsuleSoft)
                    .frame(width: 24, height: 24)
                if taskManager.tasks[index].isCompleted {
                    Image(systemName: "checkmark")
                        .font(.system(size: 13, weight: .bold))
                        .foregroundColor(.tdmInkPrimary)
                }
            }
            .contentShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            .onTapGesture {
                withAnimation(.easeInOut(duration: 0.18)) {
                    taskManager.tasks[index].isCompleted.toggle()
                }
            }
            Text(task.title)
                .foregroundColor(taskManager.tasks[index].isCompleted ? .tdmInkSecondary : .tdmInkPrimary)
                .font(DesignFont.bodyMedium())
                .lineLimit(1)
            Spacer()
            Text(task.classificationConfidence.map { "\(task.targetFolder) (\(Int($0 * 100))%)" } ?? task.targetFolder)
                .font(DesignFont.bodySmall())
                .foregroundColor(.tdmInkBio)
                .lineLimit(1)
            Menu {
                ForEach(taskManager.folderRules) { rule in
                    Button(rule.folderName) { taskManager.userPickedFolder(taskID: task.id, folder: rule.folderName) }
                }
                if taskManager.folderRules.isEmpty {
                    Text("폴더 규칙이 없습니다 — 설정에서 추가").foregroundColor(.tdmInkBio)
                }
            } label: {
                Image(systemName: "folder")
                    .font(.title3)
                    .foregroundColor(.tdmInkBio)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help("분류 폴더 변경")
        }
        .padding(.vertical, DesignSpacing.xs)
        .listRowBackground(Color.clear)
        .listRowSeparator(.hidden)
        .transition(.opacity)
        .contextMenu {
            Button(role: .destructive) {
                withAnimation(.easeInOut(duration: 0.3)) {
                    taskManager.deleteTask(id: task.id)
                }
            } label: {
                Label("삭제", systemImage: "trash")
            }
        }
    }

    var body: some View {
        ZStack {
        VStack(spacing: 0) {
        HStack(spacing: 0) {
            // 좌측 캘린더
            CustomCalendarView(taskManager: taskManager, selectedDate: $selectedDate)
                .frame(width: 300)

            Divider()

            // 우측 할 일 목록
            VStack(alignment: .leading, spacing: 0) {
                VStack(alignment: .leading, spacing: 4) {
                    // 첫 줄: "5월 7일 (목)" — 아래 "과제 목록"(28pt)의 2/3 크기.
                    // + 공휴일이면 우측에 이름 (날짜의 절반 크기, 회색).
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(selectedDate, formatter: DateFormatter.koreanHeaderFormatter)
                            .font(.system(size: 28.0 * 2.0 / 3.0, weight: .bold))
                            .foregroundColor(.tdmInkPrimary)
                        if let holidayName = taskManager.holidayName(for: selectedDate) {
                            // 날짜(28*2/3)의 2/3 = 28*4/9 ≈ 12.44pt.
                            Text(holidayName)
                                .font(.system(size: 28.0 * 4.0 / 9.0, weight: .medium))
                                .foregroundColor(.tdmInkTertiary)
                        }
                    }
                    // 둘째 줄: "과제 목록" + 같은 줄 우측에 설정/추가 아이콘 (수직 중앙 정렬).
                    HStack(alignment: .center) {
                        Text("과제 목록")
                            .font(DesignFont.heading1())
                            .foregroundColor(.tdmInkPrimary)
                        Spacer()
                        Button(action: { toggleSettings() }) {
                            Image(systemName: "gearshape.fill")
                                .font(.title2)
                                .foregroundColor(.tdmInkSecondary)
                        }
                        .buttonStyle(.plain)
                        .padding(.trailing, DesignSpacing.sm)
                        Button(action: { toggleAddTask() }) {
                            Image(systemName: isAddingTask ? "xmark.circle.fill" : "plus.circle.fill")
                                .font(.title)
                                .foregroundColor(isAddingTask ? .tdmInkTertiary : .tdmInkPrimary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.horizontal, DesignSpacing.md)
                .padding(.top, DesignSpacing.lg)
                .padding(.bottom, DesignSpacing.md)

                if let calendarError = taskManager.lastCalendarError {
                    Text("⚠️ \(calendarError)")
                        .font(.caption)
                        .foregroundColor(.orange)
                        .padding(.horizontal)
                        .padding(.bottom, 5)
                }

                if isAddingTask {
                    VStack(spacing: DesignSpacing.sm) {
                        if showAddContent {
                            TextField("할 일 제목 (예: 운영체제 과제)", text: $newTaskTitle)
                                .textFieldStyle(.plain)
                                .font(DesignFont.body())
                                .foregroundColor(.tdmInkPrimary)
                                .padding(.vertical, 10)
                                .padding(.horizontal, 14)
                                .background(
                                    RoundedRectangle(cornerRadius: DesignRadius.sm, style: .continuous)
                                        .fill(Color.tdmCapsuleSoft)
                                )
                                .onSubmit { submitAddTask() }
                            HStack {
                                Text("💡 AI가 맥락을 분석하여 폴더를 자동 지정합니다.")
                                    .font(DesignFont.bodySmall())
                                    .foregroundColor(.tdmInkBio)
                                Spacer()
                                Button(action: { submitAddTask() }) {
                                    Text("완료").primaryPill()
                                }
                                .buttonStyle(.plain)
                            }
                        }
                    }
                    .padding(DesignSpacing.md)
                    .background(
                        RoundedRectangle(cornerRadius: DesignRadius.lg, style: .continuous)
                            .fill(Color.tdmCapsule)
                    )
                    .padding(.horizontal, DesignSpacing.md)
                    .padding(.bottom, DesignSpacing.sm)
                }

                List {
                    ForEach(taskManager.tasks.filter { Calendar.current.isDate($0.date, inSameDayAs: selectedDate) }) { task in
                        if let index = taskManager.tasks.firstIndex(where: { $0.id == task.id }) {
                            taskRow(task: task, index: index)
                        }
                    }
                }
                .listStyle(.plain)
                .scrollContentBackground(.hidden)
                .background(Color.tdmCanvas)
                // 다른 날짜 선택 시 List 전체가 페이드되며 새 데이터로 교체.
                .id(selectedDate)
                .transition(.opacity)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.tdmCanvas)
        }
        if tipEnabled {
            Divider().background(Color.tdmCapsule)
            TipBar()
        }
        }

        // 설정 in-window overlay — sheet 대신 ZStack overlay로 띄움:
        // (1) 메인 윈도우 안에 들어가서 윈도우보다 커지지 않음
        // (2) 외부(scrim) 클릭 시 자동 dismiss
        // (3) 패널이 메인 콘텐츠 위에 dim+scale-in 트랜지션으로 등장
        if showSettings {
            Color.black.opacity(0.55)
                .ignoresSafeArea()
                .contentShape(Rectangle())
                .onTapGesture { dismissSettings() }
                .transition(.opacity)
                .zIndex(1)

            SettingsView(taskManager: taskManager, onDismiss: dismissSettings)
                .padding(40)
                .transition(.opacity.combined(with: .scale(scale: 0.96)))
                .zIndex(2)
        }
        }
        .background(Color.tdmCanvas.ignoresSafeArea())
        .sheet(isPresented: needsModelPickerBinding) {
            ModelPickerSheet().interactiveDismissDisabled()
        }
        .alert(needsDownloadAlertTitle, isPresented: needsDownloadBinding) {
            Button("다운로드 (~ \(needsDownloadSizeText))") { modelInstaller.startDownload() }
            Button("나중에", role: .cancel) { modelInstaller.skip() }
        } message: {
            Text(needsDownloadMessage)
        }
        .sheet(isPresented: isDownloadingBinding) {
            DownloadProgressView().interactiveDismissDisabled()
        }
        .alert("다운로드 실패", isPresented: hasFailureBinding) {
            Button("재시도") { modelInstaller.startDownload() }
            Button("닫기", role: .cancel) { modelInstaller.skip() }
        } message: {
            Text(failureMessage)
        }
    }

    private var needsModelPickerBinding: Binding<Bool> {
        Binding(
            get: { if case .awaitingSelection = modelInstaller.state { return true }; return false },
            set: { _ in }
        )
    }
    private var needsDownloadAlertTitle: String {
        let name = modelInstaller.selectedKind?.displayName ?? "모델"
        return "\(name) 모델 다운로드 필요"
    }
    private var needsDownloadSizeText: String {
        guard let kind = modelInstaller.selectedKind else { return "~ GB" }
        return String(format: "%.1f GB", kind.sizeGB)
    }
    private var needsDownloadMessage: String {
        let name = modelInstaller.selectedKind?.displayName ?? "선택된 모델"
        let size = modelInstaller.selectedKind.map { String(format: "%.1f", $0.sizeGB) } ?? "?"
        return "폴더 자동 분류용 \(name) (CoreML) 모델이 설치돼 있지 않습니다.\nHuggingFace에서 약 \(size)GB를 받습니다. 첫 실행 시 ANE 컴파일이 1~2분 추가로 걸릴 수 있습니다."
    }

    private var needsDownloadBinding: Binding<Bool> {
        Binding(
            get: { if case .needsDownload = modelInstaller.state { return true }; return false },
            set: { _ in }
        )
    }
    private var isDownloadingBinding: Binding<Bool> {
        Binding(
            get: {
                switch modelInstaller.state {
                case .downloading, .compiling: return true
                default: return false
                }
            },
            set: { _ in }
        )
    }
    private var hasFailureBinding: Binding<Bool> {
        Binding(
            get: { if case .failed = modelInstaller.state { return true }; return false },
            set: { _ in }
        )
    }
    private var failureMessage: String {
        if case .failed(let msg) = modelInstaller.state { return msg }
        return ""
    }
}

// --- 4-1. 모델 다운로드/컴파일 진행 시트 ---
struct DownloadProgressView: View {
    @ObservedObject private var installer = ModelInstaller.shared

    private var currentModelName: String {
        installer.selectedKind?.displayName ?? "모델"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: DesignSpacing.md) {
            switch installer.state {
            case .downloading(let progress, let status):
                Text("\(currentModelName) 모델 다운로드 중")
                    .font(DesignFont.heading2())
                    .foregroundColor(.tdmInkPrimary)
                ProgressView(value: progress)
                    .progressViewStyle(.linear)
                    .tint(.tdmYellow)
                Text(status)
                    .font(DesignFont.bodySmall())
                    .foregroundColor(.tdmInkBio)
                    .lineLimit(2)
                Text("\(Int(progress * 100))% — 창을 닫지 마세요.")
                    .font(DesignFont.bodySmall())
                    .foregroundColor(.tdmInkTertiary)
            case .compiling:
                Text("ANE 컴파일 중")
                    .font(DesignFont.heading2())
                    .foregroundColor(.tdmInkPrimary)
                ProgressView()
                    .progressViewStyle(.linear)
                    .tint(.tdmYellow)
                Text("CoreML이 chunked decode 모델을 Apple Neural Engine에 컴파일하고 있습니다 (1~2분 소요, 결과 캐시됨).")
                    .font(DesignFont.bodySmall())
                    .foregroundColor(.tdmInkBio)
            default:
                ProgressView().progressViewStyle(.linear).tint(.tdmYellow)
            }
        }
        .padding(DesignSpacing.lg)
        .frame(width: 480)
        .background(Color.tdmBgMenu)
    }
}

// --- 4-2. 모델 선택 picker (first-run + 설정 변경 공용) ---
/// First-run: `interactiveDismissDisabled()` 와 함께 띄워 강제 선택.
/// 설정 변경: `showsCancel: true`로 띄우면 우측 상단 ✕ 표시.
struct ModelPickerSheet: View {
    @ObservedObject private var installer = ModelInstaller.shared
    @Environment(\.dismiss) private var dismiss
    /// `true`면 우측 상단에 ✕(cancel) 버튼 표시 — 설정에서 띄울 때 사용.
    var showsCancel: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: DesignSpacing.md) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("사용할 모델을 선택하세요")
                        .font(DesignFont.heading2())
                        .foregroundColor(.tdmInkPrimary)
                    Text("폴더 자동 분류용 LLM. 다운로드 후에도 설정에서 변경할 수 있어요.")
                        .font(DesignFont.bodySmall())
                        .foregroundColor(.tdmInkBio)
                }
                Spacer()
                if showsCancel {
                    Button(action: { dismiss() }) {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title2)
                            .foregroundColor(.tdmInkTertiary)
                    }
                    .buttonStyle(.plain)
                }
            }

            VStack(spacing: DesignSpacing.sm) {
                ForEach(ModelKind.allCases, id: \.self) { kind in
                    ModelPickerCard(
                        kind: kind,
                        isCurrent: installer.selectedKind == kind,
                        onSelect: {
                            installer.selectModel(kind)
                            if showsCancel { dismiss() }
                        }
                    )
                }
            }
        }
        .padding(DesignSpacing.lg)
        .frame(width: 540)
        .background(Color.tdmBgMenu)
    }
}

private struct ModelPickerCard: View {
    let kind: ModelKind
    let isCurrent: Bool
    let onSelect: () -> Void

    var body: some View {
        Button(action: onSelect) {
            HStack(alignment: .top, spacing: DesignSpacing.md) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 6) {
                        Text(kind.displayName)
                            .font(DesignFont.heading2())
                            .foregroundColor(.tdmInkPrimary)
                        if isCurrent {
                            Text("현재")
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.tdmYellow.opacity(0.25))
                                .foregroundColor(.tdmYellow)
                                .clipShape(Capsule())
                        }
                        if !kind.isAvailable {
                            Text("곧 추가")
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.tdmInkTertiary.opacity(0.2))
                                .foregroundColor(.tdmInkTertiary)
                                .clipShape(Capsule())
                        }
                    }
                    Text(kind.summary)
                        .font(DesignFont.bodySmall())
                        .foregroundColor(.tdmInkSecondary)
                    Text(kind.detail)
                        .font(.caption)
                        .foregroundColor(.tdmInkBio)
                        .lineLimit(3)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Spacer()
                Image(systemName: isCurrent ? "checkmark.circle.fill" : "circle")
                    .font(.title2)
                    .foregroundColor(isCurrent ? .tdmYellow : .tdmInkTertiary)
            }
            .padding(DesignSpacing.md)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .fill(Color.tdmCapsule)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12, style: .continuous)
                    .stroke(isCurrent ? Color.tdmYellow : Color.clear, lineWidth: 1.5)
            )
            .opacity(kind.isAvailable ? 1.0 : 0.5)
        }
        .buttonStyle(.plain)
        .disabled(!kind.isAvailable)
    }
}

// --- 5. 설정 뷰 ---
struct SettingsView: View {
    @ObservedObject var taskManager: TaskManager
    /// In-window overlay에서 띄우므로 sheet의 `@Environment(\.dismiss)` 대신 closure로 받음.
    /// ContentView가 ZStack overlay로 넣으면서 동시에 외부 클릭 dismiss를 묶어 처리.
    let onDismiss: () -> Void
    @State private var newFolderName = ""
    @State private var selectedURL: URL? = nil
    @State private var addRuleError: String? = nil
    @State private var benchmarkRunning = false
    @State private var benchmarkProgress = 0
    @State private var benchmarkResult: String? = nil
    @State private var benchmarkTask: Task<Void, Never>? = nil
    // 세션 한정 — UserDefaults persistence 제거됨. 앱 재시작 시 .cpuAndNeuralEngine로 시작.
    // Qwen3.5 2B (CoreML)는 ANE 친화적으로 설계됨 (~91% ANE residency 보고됨).
    @State private var computeUnits: MLComputeUnits = .cpuAndNeuralEngine

    var body: some View {
        ScrollView {
        VStack(alignment: .leading, spacing: DesignSpacing.md) {
            // 드래그 핸들 (Todomate 패턴 — 시각적 anchor)
            HStack {
                Spacer()
                Capsule()
                    .fill(Color.tdmInkTertiary)
                    .frame(width: 36, height: 4)
                Spacer()
            }
            .padding(.top, DesignSpacing.xs)
            .padding(.bottom, DesignSpacing.xs)

            Text("⚙️ 감시 규칙 및 실제 폴더 연결")
                .font(DesignFont.heading2())
                .foregroundColor(.tdmInkPrimary)

            // 폴더 규칙 리스트
            VStack(spacing: DesignSpacing.xs) {
                ForEach(taskManager.folderRules) { rule in
                    HStack(spacing: DesignSpacing.sm) {
                        VStack(alignment: .leading, spacing: DesignSpacing.xxs) {
                            Text("📁 \(rule.folderName)")
                                .font(DesignFont.bodyMedium())
                                .foregroundColor(.tdmInkPrimary)
                            if let resolved = try? rule.resolveURL() {
                                Text(resolved.url.lastPathComponent)
                                    .font(DesignFont.bodySmall())
                                    .foregroundColor(.tdmInkBio)
                            } else {
                                Text("(경로 해석 실패 — 폴더를 다시 선택하세요)")
                                    .font(DesignFont.bodySmall())
                                    .foregroundColor(.tdmDateSunday)
                            }
                        }
                        Spacer()
                        Button("삭제") { taskManager.deleteRule(id: rule.id) }
                            .buttonStyle(.plain)
                            .font(DesignFont.button(13))
                            .foregroundColor(.tdmDateSunday)
                    }
                    .padding(.vertical, DesignSpacing.xs)
                    .padding(.horizontal, DesignSpacing.sm)
                    .background(
                        RoundedRectangle(cornerRadius: DesignRadius.sm, style: .continuous)
                            .fill(Color.tdmCapsuleSoft)
                    )
                }
                if taskManager.folderRules.isEmpty {
                    Text("등록된 폴더가 없어요")
                        .font(DesignFont.bodySmall())
                        .foregroundColor(.tdmInkBio)
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, DesignSpacing.md)
                }
            }

            Divider().background(Color.tdmCapsuleSoft)

            // 새 감시 폴더 추가
            Text("새 감시 폴더 추가")
                .font(DesignFont.heading3())
                .foregroundColor(.tdmInkPrimary)
            VStack(spacing: DesignSpacing.sm) {
                HStack {
                    Text("분류할 별명:")
                        .font(DesignFont.bodySmall())
                        .foregroundColor(.tdmInkBio)
                    TextField("예: 운영체제", text: $newFolderName)
                        .textFieldStyle(.plain)
                        .font(DesignFont.body())
                        .foregroundColor(.tdmInkPrimary)
                        .padding(.vertical, 8)
                        .padding(.horizontal, 12)
                        .background(
                            RoundedRectangle(cornerRadius: DesignRadius.sm, style: .continuous)
                                .fill(Color.tdmCapsule)
                        )
                }
                HStack {
                    Text("실제 폴더:")
                        .font(DesignFont.bodySmall())
                        .foregroundColor(.tdmInkBio)
                    Text(selectedURL?.lastPathComponent ?? "선택 안 됨")
                        .font(DesignFont.bodySmall())
                        .foregroundColor(selectedURL == nil ? .tdmDateSunday : .tdmDateSaturday)
                    Spacer()
                    Button(action: { selectFolderFromMac() }) {
                        Text("폴더 찾기").secondaryPill()
                    }
                    .buttonStyle(.plain)
                }
                if let addRuleError {
                    Text("⚠️ \(addRuleError)")
                        .font(DesignFont.bodySmall())
                        .foregroundColor(.tdmIconRepeatTomorrow)
                }
                HStack {
                    Spacer()
                    Button(action: {
                        if !newFolderName.isEmpty, let url = selectedURL {
                            do {
                                let bookmark = try url.bookmarkData(
                                    options: .withSecurityScope,
                                    includingResourceValuesForKeys: nil,
                                    relativeTo: nil
                                )
                                taskManager.addRule(name: newFolderName, bookmark: bookmark)
                                newFolderName = ""; selectedURL = nil; addRuleError = nil
                            } catch {
                                addRuleError = "북마크 생성 실패: \(error.localizedDescription)"
                            }
                        }
                    }) {
                        Text("추가하고 감시 시작하기").primaryPill()
                    }
                    .buttonStyle(.plain)
                    .disabled(newFolderName.isEmpty || selectedURL == nil)
                    .opacity((newFolderName.isEmpty || selectedURL == nil) ? 0.4 : 1.0)
                }
            }
            .padding(DesignSpacing.md)
            .background(
                RoundedRectangle(cornerRadius: DesignRadius.md, style: .continuous)
                    .fill(Color.tdmCapsule)
            )

            Divider().background(Color.tdmCapsuleSoft)

            // 속도 벤치마크
            Text("속도 벤치마크")
                .font(DesignFont.heading3())
                .foregroundColor(.tdmInkPrimary)
            VStack(alignment: .leading, spacing: DesignSpacing.xs) {
                Picker("Compute Units:", selection: $computeUnits) {
                    Text("CPU + Neural Engine (권장)").tag(MLComputeUnits.cpuAndNeuralEngine)
                    Text("CPU + GPU").tag(MLComputeUnits.cpuAndGPU)
                    Text("ANE + GPU + CPU (.all)").tag(MLComputeUnits.all)
                }
                .pickerStyle(.menu)
                .disabled(benchmarkRunning)
                .onChange(of: computeUnits) { _, new in
                    Task { await LocalLLMService.shared.setComputeUnits(new) }
                }
                Text("바꾸면 모델 재로드 필요 — 첫 분류 전 잠시 대기")
                    .font(DesignFont.bodySmall())
                    .foregroundColor(.tdmInkTertiary)
                Text("Gemma 4 E2B는 ANE 친화적 — `.cpuAndNeuralEngine` 모드가 가장 효율적 (~91% ANE residency)")
                    .font(DesignFont.bodySmall())
                    .foregroundColor(.tdmInkTertiary)

                if taskManager.folderRules.count < 2 {
                    Text("폴더 규칙이 2개 이상 있어야 분류기가 동작해요. 위에서 폴더를 추가해 주세요.")
                        .font(DesignFont.bodySmall())
                        .foregroundColor(.tdmIconRepeatTomorrow)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    HStack {
                        Button(action: {
                            if benchmarkRunning { benchmarkTask?.cancel() }
                            else { runBenchmark() }
                        }) {
                            Text(benchmarkRunning ? "중단" : "벤치마크 실행 (\(Self.benchmarkIterations)회 랜덤)")
                                .primaryPill()
                        }
                        .buttonStyle(.plain)
                        Text("프리셋 \(Self.benchmarkPresets.count)개에서 \(Self.benchmarkIterations)개 랜덤 추출")
                            .font(DesignFont.bodySmall())
                            .foregroundColor(.tdmInkTertiary)
                        Spacer()
                    }
                    if benchmarkRunning {
                        HStack(spacing: DesignSpacing.xs) {
                            ProgressView().controlSize(.small).tint(.tdmYellow)
                            Text("진행 \(benchmarkProgress)/\(Self.benchmarkIterations)…")
                                .font(DesignFont.bodySmall())
                                .foregroundColor(.tdmInkBio)
                        }
                    }
                    if let result = benchmarkResult {
                        Text(result)
                            .font(DesignFont.bodySmall())
                            .foregroundColor(.tdmInkBio)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
            .padding(DesignSpacing.md)
            .background(
                RoundedRectangle(cornerRadius: DesignRadius.md, style: .continuous)
                    .fill(Color.tdmCapsule)
            )

            HStack {
                Spacer()
                Button(action: { onDismiss() }) {
                    Text("닫기").secondaryPill()
                }
                .buttonStyle(.plain)
                .keyboardShortcut(.escape)
            }
        }
        .padding(DesignSpacing.lg)
        }
        .frame(maxWidth: 500, maxHeight: 600)
        .background(Color.tdmBgMenu)
        .clipShape(RoundedRectangle(cornerRadius: DesignRadius.lg, style: .continuous))
        .shadow(color: .black.opacity(0.5), radius: 40, x: 0, y: 12)
        .contentShape(RoundedRectangle(cornerRadius: DesignRadius.lg, style: .continuous))
    }

    private func selectFolderFromMac() {
        let panel = NSOpenPanel(); panel.canChooseFiles = false; panel.canChooseDirectories = true
        if panel.runModal() == .OK { self.selectedURL = panel.url }
    }

    private static let benchmarkIterations = 5
    /// 벤치마크용 프리셋 task 풀. 사용자의 실제 폴더 카테고리(희곡교육론 / SW프로그래밍의기초 / 국어교과교육론)에서
    /// 자연스럽게 나올 만한 할 일 15개. classifyTask가 실제로 LLM 추론 경로(F1.1 fast logits, F1.3 조건부 시드)를
    /// 거치는지 확인용 — corrections에 없는 항목이라 F1.2 정확매치 단축은 거의 발동 안 함.
    private static let benchmarkPresets: [String] = [
        // 희곡교육론
        "셰익스피어 작품 분석 과제",
        "교실 연극 시나리오 정리",
        "현대 희곡 감상문 작성",
        "낭독극 대본 준비",
        "연극 교육 사례 조사",
        // SW프로그래밍의기초
        "C언어 포인터 실습",
        "for 반복문 연습 문제",
        "재귀 함수 과제 풀이",
        "배열 정렬 코드 작성",
        "코딩 테스트 한 문제 풀기",
        // 국어교과교육론
        "국어 교과서 단원 분석",
        "수업 지도안 작성",
        "문법 단원 자료 정리",
        "교생실습 일지 작성",
        "국어과 교육과정 비교 정리"
    ]

    private func runBenchmark() {
        let corrections = taskManager.corrections
        let folders = taskManager.folderRules.map { $0.folderName }
        guard folders.count >= 2 else { return }
        // 매 실행마다 프리셋에서 5개 랜덤 추출 (no replacement)
        let iterations = Array(Self.benchmarkPresets.shuffled().prefix(Self.benchmarkIterations))

        benchmarkRunning = true
        benchmarkProgress = 0
        benchmarkResult = nil

        benchmarkTask = Task { @MainActor in
            var total = 0
            var totalTime: Double = 0
            let startWall = Date()
            for title in iterations {
                if Task.isCancelled { break }
                // 실제 addTask 경로와 동일하게 corrections.suffix(5) 필터 적용 — 현실적인 측정
                let context = Array(corrections.filter { folders.contains($0.folderName) }.suffix(5))
                let t0 = Date()
                let _ = await LocalLLMService.shared.classifyTask(
                    taskTitle: title,
                    availableFolders: folders,
                    corrections: context
                )
                let dt = Date().timeIntervalSince(t0)
                totalTime += dt
                total += 1
                benchmarkProgress = total
            }
            let cancelled = Task.isCancelled
            let avg = total > 0 ? totalTime / Double(total) : 0
            let wall = Date().timeIntervalSince(startWall)
            let units = await LocalLLMService.shared.currentComputeUnits.label
            benchmarkResult = cancelled
                ? String(format: "[%@] 중단됨 (진행 %d/%d) · 평균 %.2fs · 누적 %.1fs",
                         units, total, Self.benchmarkIterations, avg, wall)
                : String(format: "[%@] %d회 평균 추론 시간 %.2fs · 총 %.1fs",
                         units, total, avg, wall)
            benchmarkRunning = false
        }
    }
}

extension DateFormatter {
    static let monthYearFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "YYYY년 MM월"
        return formatter
    }()

    /// 헤더용 한국어 날짜 + 약어 요일 (예: "5월 7일 (목)").
    static let koreanHeaderFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "ko_KR")
        formatter.dateFormat = "M월 d일 (E)"
        return formatter
    }()
}
