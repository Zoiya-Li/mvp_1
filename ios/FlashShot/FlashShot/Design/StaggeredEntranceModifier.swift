import SwiftUI

struct StaggeredEntranceModifier: ViewModifier {
    let order: Int
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @State private var isVisible = false

    private var prefersStableLayout: Bool {
        reduceMotion || dynamicTypeSize.isAccessibilitySize
    }

    func body(content: Content) -> some View {
        content
            .opacity(isVisible ? 1 : 0)
            .offset(y: prefersStableLayout || isVisible ? 0 : 16)
            .animation(
                FlashShotMotion.gentle(
                    reduceMotion: prefersStableLayout,
                    delay: prefersStableLayout ? 0 : min(Double(order) * 0.055, 0.22)
                ),
                value: isVisible
            )
            .onAppear { isVisible = true }
    }
}

extension View {
    func flashShotEntrance(order: Int = 0) -> some View {
        modifier(StaggeredEntranceModifier(order: order))
    }
}
