//
// ContentView.swift
// Cplanner
//
// Created by iLo on 2026-03-26.
//

import SwiftUI
import Combine
import EventKit

// --- 1. 모델 정의 ---
struct TaskItem: Identifiable {
    let id = UUID()
    var title: String
    var isCompleted: Bool = false
    var targetFolder: String
    var date: Date
}

struct FolderRule: Identifiable, Codable {
    var id = UUID()
    var folderName: String
    var keywords: [String]
    var folderURLPath: String
    var url: URL { URL(fileURLWithPath: folderURLPath) }
}

// --- 2. 메인 매니저 ---
class TaskManager: ObservableObject {
    @Published var tasks: [TaskItem] = []
    @Published var folderRules: [FolderRule] = []
    
    let eventStore = EKEventStore()
    private var monitors: [UUID: FolderMonitor] = [:]
    
    // 할 일 추가 (AI 분류 적용)
    func addTask(title: String, date: Date) {
        Task {
            let folders = folderRules.map { $0.folderName }
            // AI에게 맥락 분류 요청
            let detectedFolder = await LocalLLMService.shared.classifyTask(taskTitle: title, availableFolders: folders)
            
            DispatchQueue.main.async {
                let newTask = TaskItem(title: title, targetFolder: detectedFolder, date: date)
                self.tasks.append(newTask)
                self.saveToMacCalendar(title: title, date: date)
            }
        }
    }
    
    func deleteTask(id: UUID) { tasks.removeAll { $0.id == id } }
    
    // 감시 규칙 추가
    func addRule(name: String, keywords: [String], url: URL) {
        let newRule = FolderRule(folderName: name, keywords: keywords, folderURLPath: url.path)
        folderRules.append(newRule)
        startMonitoring(rule: newRule)
    }
    
    func deleteRule(id: UUID) {
        folderRules.removeAll { $0.id == id }
        monitors[id]?.stopMonitoring()
        monitors.removeValue(forKey: id)
    }
    
    // 폴더 감시 시작
    private func startMonitoring(rule: FolderRule) {
        let monitor = FolderMonitor(url: rule.url)
        monitor.folderDidChange = { [weak self] in self?.handleNewFileDetected(in: rule) }
        monitor.startMonitoring()
        monitors[rule.id] = monitor
    }
    
    // 파일 감지 시 AI 검증 로직 실행
    private func handleNewFileDetected(in rule: FolderRule) {
        let newFileName = "과제제출본.pdf" // 실제로는 FileManager를 통해 최신 파일명을 가져옵니다.
        
        Task {
            let isValid = await LocalLLMService.shared.validateFileContext(fileName: newFileName, folderName: rule.folderName)
            if isValid {
                DispatchQueue.main.async {
                    if let idx = self.tasks.firstIndex(where: { $0.targetFolder == rule.folderName && !$0.isCompleted }) {
                        self.tasks[idx].isCompleted = true
                    }
                }
            }
        }
    }
    
    private func saveToMacCalendar(title: String, date: Date) {
        let event = EKEvent(eventStore: eventStore)
        event.title = "[Cplanner] \(title)"
        event.startDate = date; event.endDate = date; event.isAllDay = true
        event.calendar = eventStore.defaultCalendarForNewEvents
        try? eventStore.save(event, span: .thisEvent)
    }
}

// --- 3. 커스텀 캘린더 뷰 ---
struct CustomCalendarView: View {
    @ObservedObject var taskManager: TaskManager
    @Binding var selectedDate: Date
    
    private let calendarHeaderColor = Color(NSColor.windowBackgroundColor)
    private let selectionColor = Color.blue
    let weekdayNames = ["일", "월", "화", "수", "목", "금", "토"]
    
    func generateDaysInMonth() -> [Date] {
        let calendar = Calendar.current
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
            HStack {
                Text(selectedDate, formatter: DateFormatter.monthYearFormatter).font(.title3).fontWeight(.bold)
                Spacer()
                HStack(spacing: 5) {
                    Button(action: { changeMonth(value: -1) }) { Image(systemName: "chevron.left") }
                    Button(action: { changeMonth(value: 1) }) { Image(systemName: "chevron.right") }
                }.buttonStyle(.plain).font(.title3)
            }.padding().background(calendarHeaderColor)
            
            Divider()
            Grid(horizontalSpacing: 0, verticalSpacing: 0) {
                GridRow {
                    ForEach(weekdayNames, id: \.self) { weekday in
                        Text(weekday).font(.caption).fontWeight(.medium)
                            .foregroundColor(weekday == "일" ? .red : (weekday == "토" ? .blue : .gray))
                            .frame(maxWidth: .infinity).frame(height: 30)
                    }
                }.background(calendarHeaderColor)
            }
            Divider()
            
            LazyVGrid(columns: Array(repeating: GridItem(.flexible(), spacing: 0), count: 7), spacing: 0) {
                ForEach(generateDaysInMonth(), id: \.self) { date in
                    let isWeekend = Calendar.current.component(.weekday, from: date) == 1 || Calendar.current.component(.weekday, from: date) == 7
                    VStack {
                        ZStack {
                            if Calendar.current.isDate(date, inSameDayAs: selectedDate) {
                                Circle().fill(selectionColor).frame(width: 30, height: 30)
                            }
                            Text("\(Calendar.current.component(.day, from: date))")
                                .font(.system(size: 14))
                                .fontWeight(Calendar.current.isDate(date, inSameDayAs: selectedDate) ? .bold : .medium)
                                .foregroundColor(isSameMonth(date1: date, date2: selectedDate) ? (Calendar.current.isDate(date, inSameDayAs: selectedDate) ? .white : .primary) : .secondary.opacity(0.5))
                        }
                        if hasTasks(date: date) {
                            Circle().fill(isSameMonth(date1: date, date2: selectedDate) ? (Calendar.current.isDate(date, inSameDayAs: selectedDate) ? .white : .blue) : .secondary.opacity(0.5)).frame(width: 4, height: 4)
                        }
                    }
                    .frame(height: 50).frame(maxWidth: .infinity)
                    .background(isWeekend ? Color.gray.opacity(0.06) : Color.clear)
                    .overlay(Rectangle().stroke(Color.gray.opacity(0.1), lineWidth: 0.5))
                    .contentShape(Rectangle())
                    .onTapGesture { self.selectedDate = date }
                }
            }.background(Color(NSColor.controlBackgroundColor))
            
            Divider()
            Spacer()
        }.background(Color(NSColor.controlBackgroundColor))
    }
    
    func changeMonth(value: Int) {
        if let newDate = Calendar.current.date(byAdding: .month, value: value, to: selectedDate) { selectedDate = newDate }
    }
    func isSameMonth(date1: Date, date2: Date) -> Bool { Calendar.current.isDate(date1, equalTo: date2, toGranularity: .month) }
    func hasTasks(date: Date) -> Bool { taskManager.tasks.contains { Calendar.current.isDate($0.date, inSameDayAs: date) } }
}

// --- 4. 메인 UI 화면 ---
struct ContentView: View {
    @StateObject private var taskManager = TaskManager()
    @State private var selectedDate = Date()
    @State private var isAddingTask = false
    @State private var showSettings = false
    @State private var newTaskTitle = ""
    
    var body: some View {
        HStack(spacing: 0) {
            // 좌측 캘린더
            CustomCalendarView(taskManager: taskManager, selectedDate: $selectedDate)
                .frame(width: 300)
            
            Divider()
            
            // 우측 할 일 목록
            VStack(alignment: .leading, spacing: 0) {
                HStack {
                    Text("\(selectedDate.formatted(.dateTime.month().day())) 과제 목록").font(.title).fontWeight(.heavy)
                    Spacer()
                    Button(action: { showSettings.toggle() }) { Image(systemName: "gearshape.fill").font(.title2).foregroundColor(.secondary) }.buttonStyle(.plain).padding(.trailing, 10)
                    Button(action: { withAnimation { isAddingTask.toggle() } }) { Image(systemName: isAddingTask ? "xmark.circle.fill" : "plus.circle.fill").font(.title).foregroundColor(isAddingTask ? .gray : .blue) }.buttonStyle(.plain)
                }.padding(.horizontal).padding(.top, 25).padding(.bottom, 15)
                
                if isAddingTask {
                    VStack(spacing: 12) {
                        TextField("할 일 제목 (예: 운영체제 과제)", text: $newTaskTitle)
                            .textFieldStyle(.roundedBorder)
                            .onSubmit {
                                if !newTaskTitle.isEmpty { taskManager.addTask(title: newTaskTitle, date: selectedDate); newTaskTitle = ""; withAnimation { isAddingTask = false } }
                            }
                        HStack {
                            Text("💡 AI가 맥락을 분석하여 폴더를 자동 지정합니다.").font(.caption).foregroundColor(.secondary)
                            Spacer()
                            Button("완료") {
                                if !newTaskTitle.isEmpty { taskManager.addTask(title: newTaskTitle, date: selectedDate); newTaskTitle = ""; withAnimation { isAddingTask = false } }
                            }.buttonStyle(.borderedProminent)
                        }
                    }.padding().background(Color(NSColor.windowBackgroundColor)).cornerRadius(10).padding(.horizontal).padding(.bottom, 10)
                }
                
                List {
                    ForEach(taskManager.tasks.filter { Calendar.current.isDate($0.date, inSameDayAs: selectedDate) }) { task in
                        let index = taskManager.tasks.firstIndex(where: { $0.id == task.id })!
                        HStack {
                            Image(systemName: taskManager.tasks[index].isCompleted ? "checkmark.circle.fill" : "circle")
                                .foregroundColor(taskManager.tasks[index].isCompleted ? .blue : .gray).font(.title3)
                                .onTapGesture { taskManager.tasks[index].isCompleted.toggle() }
                            VStack(alignment: .leading) {
                                Text(task.title).strikethrough(taskManager.tasks[index].isCompleted).foregroundColor(taskManager.tasks[index].isCompleted ? .gray : .primary).font(.headline)
                                Text("📁 자동 분류됨: \(task.targetFolder)").font(.caption).foregroundColor(.secondary)
                            }
                            Spacer()
                        }
                        .padding(.vertical, 6)
                        .contextMenu { Button(role: .destructive) { taskManager.deleteTask(id: task.id) } label: { Label("삭제", systemImage: "trash") } }
                    }
                }.listStyle(.inset)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity).background(Color(NSColor.textBackgroundColor))
        }
        .sheet(isPresented: $showSettings) { SettingsView(taskManager: taskManager) }
    }
}

// --- 5. 설정 뷰 ---
struct SettingsView: View {
    @ObservedObject var taskManager: TaskManager
    @Environment(\.dismiss) var dismiss
    @State private var newFolderName = ""
    @State private var selectedURL: URL? = nil
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("⚙️ 감시 규칙 및 실제 폴더 연결").font(.title2).fontWeight(.bold)
            
            List {
                ForEach(taskManager.folderRules) { rule in
                    HStack {
                        VStack(alignment: .leading) {
                            Text("📁 \(rule.folderName) (경로: \(rule.url.lastPathComponent))").font(.headline)
                        }
                        Spacer()
                        Button("삭제") { taskManager.deleteRule(id: rule.id) }.buttonStyle(.plain).foregroundColor(.red)
                    }.padding(.vertical, 4)
                }
            }.listStyle(.bordered).frame(height: 150)
            
            Divider()
            Text("새 감시 폴더 추가").font(.headline)
            VStack(spacing: 10) {
                HStack { Text("분류할 별명:"); TextField("예: 운영체제", text: $newFolderName).textFieldStyle(.roundedBorder) }
                HStack {
                    Text("실제 폴더:"); Text(selectedURL?.lastPathComponent ?? "선택 안 됨").foregroundColor(selectedURL == nil ? .red : .blue)
                    Spacer()
                    Button("폴더 찾기") { selectFolderFromMac() }
                }
                Button("추가하고 감시 시작하기") {
                    if !newFolderName.isEmpty, let url = selectedURL {
                        taskManager.addRule(name: newFolderName, keywords: [newFolderName], url: url) // 키워드 입력 생략, AI가 이름 기반으로 분석
                        newFolderName = ""; selectedURL = nil
                    }
                }.buttonStyle(.borderedProminent).disabled(newFolderName.isEmpty || selectedURL == nil)
            }.padding().background(Color(NSColor.controlBackgroundColor)).cornerRadius(8)
            
            HStack { Spacer(); Button("닫기") { dismiss() }.keyboardShortcut(.escape) }
        }.padding().frame(width: 500, height: 450)
    }
    
    private func selectFolderFromMac() {
        let panel = NSOpenPanel(); panel.canChooseFiles = false; panel.canChooseDirectories = true
        if panel.runModal() == .OK { self.selectedURL = panel.url }
    }
}

extension DateFormatter {
    static var monthYearFormatter: DateFormatter {
        let formatter = DateFormatter(); formatter.dateFormat = "YYYY년 MM월"; return formatter
    }
}
