import SwiftUI

struct RootTabView: View {
    @EnvironmentObject private var session: AppSession
    @AppStorage("flashshot-onboarding-complete") private var onboardingComplete = false
    @State private var selectedTab: AppTab = .discover
    @State private var showsOnboarding = false
    @State private var evaluatedOnboarding = false

    var body: some View {
        TabView(selection: $selectedTab) {
            NavigationStack { DiscoverView() }
                .tabItem { Label("写真", systemImage: "sparkles.rectangle.stack") }
                .tag(AppTab.discover)

            NavigationStack { CustomCreateEntryView() }
                .tabItem { Label("创作", systemImage: "plus.circle") }
                .tag(AppTab.create)

            NavigationStack {
                LibraryView(onStartCreating: { selectedTab = .discover })
            }
                .tabItem { Label("相册", systemImage: "square.grid.2x2") }
                .tag(AppTab.library)

            NavigationStack {
                AccountView(onShowGuide: { showsOnboarding = true })
            }
                .tabItem { Label("我的", systemImage: "person.crop.circle") }
                .tag(AppTab.account)
        }
        .sheet(item: Binding(
            get: { session.incomingShareToken.map(ShareToken.init) },
            set: { if $0 == nil { session.incomingShareToken = nil } }
        )) { item in
            NavigationStack { SharedRecipeEntryView(shareToken: item.value) }
        }
        .fullScreenCover(isPresented: $showsOnboarding) {
            OnboardingView(
                chooseStyle: { finishOnboarding(on: .discover) },
                useInspiration: { finishOnboarding(on: .create) },
                dismiss: { finishOnboarding() }
            )
        }
        .task {
            guard !evaluatedOnboarding else { return }
            evaluatedOnboarding = true

            let environment = ProcessInfo.processInfo.environment
            guard environment["FLASHSHOT_SKIP_ONBOARDING"] != "1",
                  session.incomingShareToken == nil else { return }

            if !onboardingComplete || environment["FLASHSHOT_FORCE_ONBOARDING"] == "1" {
                showsOnboarding = true
            }
        }
        .sensoryFeedback(.selection, trigger: selectedTab)
    }

    private func finishOnboarding(on tab: AppTab? = nil) {
        onboardingComplete = true
        if let tab { selectedTab = tab }
        showsOnboarding = false
    }
}

private enum AppTab: Hashable {
    case discover
    case create
    case library
    case account
}

private struct ShareToken: Identifiable {
    let value: String
    var id: String { value }
}
