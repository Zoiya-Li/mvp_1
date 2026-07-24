import SwiftUI
import UIKit

@MainActor
final class LibraryViewModel: ObservableObject {
    @Published var projects: [PortraitProject] = []
    @Published var covers: [String: UIImage] = [:]
    @Published var isLoading = false
    @Published var error: String?

    func load(session: AppSession) async {
        isLoading = true
        do {
            let token = try await session.ensureGuest()
            projects = try await APIClient.shared.projects(token: token)
            error = nil
            isLoading = false
            await loadCovers(token: token)
        } catch {
            isLoading = false
            self.error = error.localizedDescription
        }
    }

    private func loadCovers(token: String) async {
        for project in projects.prefix(12)
        where project.status == .previewReady || project.status == .delivered {
            guard covers[project.projectId] == nil,
                  let data = try? await APIClient.shared.heroData(
                    projectID: project.projectId,
                    token: token
                  ),
                  let image = UIImage(data: data) else { continue }
            covers[project.projectId] = image
        }
    }
}

struct LibraryView: View {
    @EnvironmentObject private var session: AppSession
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @StateObject private var model = LibraryViewModel()
    var onStartCreating: () -> Void = {}

    var body: some View {
        Group {
            if model.isLoading && model.projects.isEmpty {
                ProgressView()
            } else if model.projects.isEmpty {
                ContentUnavailableView {
                    Label("你的写真集从这里开始", systemImage: "photo.stack")
                } description: {
                    Text(model.error == nil
                        ? "每一张首图，以及每一套你选择留下的写真，都会收藏在这里。"
                        : "刚才没能刷新写真集，但你的照片仍然安全。")
                } actions: {
                    if model.error == nil {
                        Button("挑选第一套写真", action: onStartCreating)
                            .buttonStyle(.borderedProminent)
                    } else {
                        Button("重新加载") { Task { await model.load(session: session) } }
                            .buttonStyle(.borderedProminent)
                    }
                }
                .flashShotEntrance()
            } else {
                List {
                    Section {
                        ForEach(Array(model.projects.enumerated()), id: \.element.id) { index, project in
                            NavigationLink(value: project) {
                                HStack(spacing: 13) {
                                    projectCover(project)
                                    VStack(alignment: .leading, spacing: 6) {
                                        Text(projectTitle(project))
                                            .font(.headline)
                                        Text(project.source == .privateInspiration
                                            ? "来自你的灵感图"
                                            : "FlashShot 主题创作")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                        StatusPill(status: project.status)
                                    }
                                }
                                .padding(.vertical, 4)
                            }
                            .flashShotEntrance(order: min(index, 5))
                        }
                    } header: {
                        Text("那些你选择留下的自己")
                    }
                }
                .listStyle(.insetGrouped)
            }
        }
        .navigationTitle("我的写真集")
        .navigationDestination(for: PortraitProject.self) { ProjectDetailView(project: $0) }
        .task(id: session.token) { await model.load(session: session) }
        .refreshable { await model.load(session: session) }
        .overlay(alignment: .bottom) {
            if model.error != nil && !model.projects.isEmpty {
                Label("写真集暂时无法刷新", systemImage: "wifi.exclamationmark")
                    .font(.footnote)
                    .padding(10)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
                    .padding()
            }
        }
    }

    private func projectCover(_ project: PortraitProject) -> some View {
        Group {
            if let image = model.covers[project.projectId] {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                Image(systemName: project.status == .delivered ? "photo.stack.fill" : "camera.aperture")
                    .font(.title2)
                    .foregroundStyle(project.status == .delivered ? FlashShotStyle.jade : FlashShotStyle.accent)
            }
        }
        .frame(width: 58, height: 72)
        .background(FlashShotStyle.secondaryPaper)
        .clipped()
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .contentTransition(.opacity)
        .animation(
            FlashShotMotion.quick(reduceMotion: reduceMotion),
            value: model.covers[project.projectId] != nil
        )
    }

    private func projectTitle(_ project: PortraitProject) -> String {
        switch project.status {
        case .delivered: "一套你选择留下的写真"
        case .previewReady: "第一张写真"
        case .previewGenerating, .setGenerating: "正在成形的写真"
        case .failed: "需要重新处理的写真"
        default: project.source == .privateInspiration ? "我的灵感创作" : "一套新的写真"
        }
    }
}
