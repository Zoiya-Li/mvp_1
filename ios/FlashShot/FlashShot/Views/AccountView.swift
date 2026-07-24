import SwiftUI

struct AccountView: View {
    @EnvironmentObject private var session: AppSession
    @EnvironmentObject private var store: StoreKitManager
    @State private var confirmDelete = false
    @State private var errorMessage: String?
    var onShowGuide: () -> Void = {}

    var body: some View {
        List {
            Section {
                if session.isAppleAccount {
                    Label("已通过 Apple 登录", systemImage: "checkmark.seal.fill")
                        .foregroundStyle(FlashShotStyle.jade)
                    if let userID = session.userID {
                        Text(userID)
                            .font(.caption.monospaced())
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                } else {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("在不同设备间保留购买记录和写真")
                            .font(.headline)
                        AppleAccountButton()
                    }
                    .padding(.vertical, 8)
                }
            }

            Section("开始使用") {
                Button(action: onShowGuide) {
                    Label("FlashShot 使用指南", systemImage: "questionmark.circle")
                }
            }

            Section("隐私") {
                Link(destination: URL(string: "https://flashshot.top/privacy")!) {
                    Label("隐私政策", systemImage: "hand.raised")
                }
                Link(destination: URL(string: "https://flashshot.top/terms")!) {
                    Label("服务条款", systemImage: "doc.text")
                }
                Button(role: .destructive) { confirmDelete = true } label: {
                    Label("删除私人创作空间", systemImage: "trash")
                }
                .disabled(store.pendingProjectID != nil)
                if store.pendingProjectID != nil {
                    Text("有一笔购买正在等待 Apple 批准。在完成或取消前，请保留这个创作空间。")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }

            Section("帮助与支持") {
                Link(destination: URL(string: "mailto:support@flashshot.top")!) {
                    Label("联系支持", systemImage: "envelope")
                }
            }

            if let errorMessage {
                Section { Text(errorMessage).foregroundStyle(FlashShotStyle.accent) }
            }
        }
        .navigationTitle("我的")
        .confirmationDialog(
            "删除这个创作空间及其中的写真？",
            isPresented: $confirmDelete,
            titleVisibility: .visible
        ) {
            Button("确认删除", role: .destructive) {
                Task {
                    do { try await session.deleteAccount() }
                    catch { errorMessage = error.localizedDescription }
                }
            }
        }
    }
}
