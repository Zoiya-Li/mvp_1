import SwiftUI
import UIKit

struct PortraitHeroReveal: View {
    let image: UIImage
    let onDeveloped: () -> Void
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var developed = false

    private static let developDuration: Double = 2.4

    var body: some View {
        Image(uiImage: image)
            .resizable()
            .scaledToFill()
            .aspectRatio(3 / 4, contentMode: .fit)
            .clipped()
            .blur(radius: developed || reduceMotion ? 0 : 22)
            .brightness(developed || reduceMotion ? 0 : -0.07)
            .scaleEffect(developed || reduceMotion ? 1 : 1.035)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .shadow(color: FlashShotStyle.ink.opacity(0.16), radius: 18, y: 10)
            .accessibilityLabel("你的第一张写真")
            .task { await develop() }
    }

    @MainActor
    private func develop() async {
        if reduceMotion {
            try? await Task.sleep(for: .seconds(0.35))
            guard !Task.isCancelled else { return }
            onDeveloped()
            return
        }
        withAnimation(.easeOut(duration: Self.developDuration)) {
            developed = true
        }
        try? await Task.sleep(for: .seconds(Self.developDuration))
        guard !Task.isCancelled else { return }
        onDeveloped()
    }
}
