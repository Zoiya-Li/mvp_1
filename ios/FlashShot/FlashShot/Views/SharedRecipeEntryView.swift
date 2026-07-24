import SwiftUI

@MainActor
final class SharedRecipeEntryViewModel: ObservableObject {
    @Published var recipe: SharedRecipe?
    @Published var isLoading = false
    @Published var errorMessage: String?

    func load(token: String) async {
        isLoading = true
        defer { isLoading = false }
        do {
            recipe = try await APIClient.shared.sharedRecipe(token)
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

struct SharedRecipeEntryView: View {
    @Environment(\.dismiss) private var dismiss
    let shareToken: String
    @StateObject private var model = SharedRecipeEntryViewModel()
    @State private var creating = false

    var body: some View {
        Group {
            if model.isLoading && model.recipe == nil {
                ProgressView()
            } else if let recipe = model.recipe {
                VStack(alignment: .leading, spacing: 18) {
                    Image(systemName: "square.and.arrow.down.on.square.fill")
                        .font(.system(size: 42))
                        .foregroundStyle(FlashShotStyle.accent)
                    Text(recipe.title)
                        .font(.largeTitle.bold())
                    Text("有人分享了自己喜欢的写真配方。用你的照片，把它变成属于你的作品。")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("拍摄同款写真") { creating = true }
                        .buttonStyle(PrimaryActionButtonStyle())
                }
                .padding(20)
            } else {
                ContentUnavailableView {
                    Label("这份写真配方暂时不可用", systemImage: "link.badge.plus")
                } description: {
                    Text(model.errorMessage ?? "这份分享的写真配方已不可用。")
                } actions: {
                    Button("重新加载") { Task { await model.load(token: shareToken) } }
                }
            }
        }
        .navigationTitle("同款写真")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("关闭") { dismiss() }
            }
        }
        .task { await model.load(token: shareToken) }
        .sheet(isPresented: $creating) {
            NavigationStack {
                CreateFlowView(
                    theme: nil,
                    source: .sharedRecipe,
                    sharedRecipeID: shareToken
                )
            }
        }
    }
}
