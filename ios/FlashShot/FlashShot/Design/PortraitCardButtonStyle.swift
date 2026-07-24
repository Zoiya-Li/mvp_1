import SwiftUI

struct PortraitCardButtonStyle: ButtonStyle {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed && !reduceMotion ? 0.985 : 1)
            .opacity(configuration.isPressed ? 0.86 : 1)
            .animation(
                FlashShotMotion.quick(reduceMotion: reduceMotion),
                value: configuration.isPressed
            )
    }
}
