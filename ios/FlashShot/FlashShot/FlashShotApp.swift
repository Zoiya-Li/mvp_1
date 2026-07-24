import SwiftUI

@main
struct FlashShotApp: App {
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var session = AppSession()
    @StateObject private var store = StoreKitManager()

    var body: some Scene {
        WindowGroup {
            RootTabView()
                .environmentObject(session)
                .environmentObject(store)
                .tint(FlashShotStyle.accent)
                .preferredColorScheme(debugColorScheme)
                .onOpenURL { session.handle(url: $0) }
                .onChange(of: scenePhase) { _, phase in
                    guard phase == .active, store.product == nil else { return }
                    Task { await store.loadProduct() }
                }
        }
    }

    private var debugColorScheme: ColorScheme? {
        #if DEBUG
        ProcessInfo.processInfo.environment["FLASHSHOT_UI_COLOR_SCHEME"] == "dark" ? .dark : nil
        #else
        nil
        #endif
    }
}
