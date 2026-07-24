import SwiftUI

enum FlashShotStyle {
    static let accent = Color(red: 0.91, green: 0.23, blue: 0.20)
    static let ink = Color(red: 0.08, green: 0.07, blue: 0.065)
    static let jade = Color(red: 0.10, green: 0.45, blue: 0.38)
    static let paper = Color(uiColor: .systemBackground)
    static let secondaryPaper = Color(uiColor: .secondarySystemBackground)
}

private struct FlashShotContentWidth: ViewModifier {
    let maxWidth: CGFloat

    func body(content: Content) -> some View {
        content
            .frame(maxWidth: maxWidth, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .center)
    }
}

extension View {
    func flashShotContentWidth(_ maxWidth: CGFloat = 720) -> some View {
        modifier(FlashShotContentWidth(maxWidth: maxWidth))
    }
}

struct PrimaryActionButtonStyle: ButtonStyle {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.headline)
            .foregroundStyle(.white)
            .frame(maxWidth: .infinity, minHeight: 52)
            .background(configuration.isPressed ? FlashShotStyle.accent.opacity(0.72) : FlashShotStyle.accent)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .scaleEffect(configuration.isPressed && !reduceMotion ? 0.985 : 1)
            .animation(
                FlashShotMotion.quick(reduceMotion: reduceMotion),
                value: configuration.isPressed
            )
    }
}

struct StatusPill: View {
    let status: ProjectStatus

    var body: some View {
        Label(label, systemImage: symbol)
            .font(.caption.weight(.semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(color.opacity(0.1), in: Capsule())
    }

    private var label: String {
        switch status {
        case .delivered: "写真已完成"
        case .failed: "需要重新看看"
        case .previewReady: "首张写真已完成"
        case .setGenerating: "正在完成整套写真"
        case .previewGenerating: "正在拍摄首张写真"
        default: "正在准备"
        }
    }

    private var symbol: String {
        switch status {
        case .delivered: "checkmark.circle.fill"
        case .failed: "exclamationmark.triangle.fill"
        default: "clock.fill"
        }
    }

    private var color: Color {
        switch status {
        case .delivered: FlashShotStyle.jade
        case .failed: FlashShotStyle.accent
        default: .secondary
        }
    }
}
