import SwiftUI

struct StudioProgressIndicator: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var isTurning = false

    var body: some View {
        ZStack {
            Circle()
                .stroke(FlashShotStyle.accent.opacity(0.16), lineWidth: 1)
                .frame(width: 76, height: 76)
            Circle()
                .trim(from: 0.08, to: 0.72)
                .stroke(
                    FlashShotStyle.accent,
                    style: StrokeStyle(lineWidth: 2, lineCap: .round)
                )
                .frame(width: 76, height: 76)
                .rotationEffect(.degrees(isTurning ? 360 : 0))
            Image(systemName: "camera.aperture")
                .font(.system(size: 32, weight: .light))
                .foregroundStyle(FlashShotStyle.accent)
        }
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.linear(duration: 2.4).repeatForever(autoreverses: false)) {
                isTurning = true
            }
        }
        .accessibilityHidden(true)
    }
}
