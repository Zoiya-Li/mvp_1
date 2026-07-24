import SwiftUI

struct PortraitIntentPicker: View {
    @Binding var selection: PortraitIntent
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Namespace private var selectionNamespace

    var body: some View {
        Group {
            if dynamicTypeSize.isAccessibilitySize {
                VStack(spacing: 8) {
                    ForEach(PortraitIntent.allCases) { intent in
                        choice(intent, expanded: true)
                    }
                }
            } else {
                HStack(spacing: 8) {
                    ForEach(PortraitIntent.allCases) { intent in
                        choice(intent, expanded: false)
                    }
                }
            }
        }
        .accessibilityElement(children: .contain)
        .accessibilityLabel("写真想表达的感觉")
        .sensoryFeedback(.selection, trigger: selection)
    }

    private func choice(_ intent: PortraitIntent, expanded: Bool) -> some View {
        Button { select(intent) } label: {
            Group {
                if expanded {
                    HStack(spacing: 12) {
                        symbol(intent)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(intent.title).font(.headline)
                            Text(intent.description)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                    }
                } else {
                    VStack(spacing: 7) {
                        symbol(intent)
                        Text(intent.title)
                            .font(.caption.weight(.semibold))
                            .multilineTextAlignment(.center)
                            .lineLimit(2)
                    }
                    .frame(maxWidth: .infinity)
                }
            }
            .frame(maxWidth: .infinity, minHeight: expanded ? 62 : 74)
            .padding(.horizontal, expanded ? 12 : 6)
            .background {
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(FlashShotStyle.secondaryPaper)
                    if selection == intent {
                        RoundedRectangle(cornerRadius: 8)
                            .fill(FlashShotStyle.accent.opacity(0.09))
                            .matchedGeometryEffect(id: "portrait-intent", in: selectionNamespace)
                    }
                }
            }
            .overlay {
                RoundedRectangle(cornerRadius: 8)
                    .stroke(
                        selection == intent ? FlashShotStyle.accent.opacity(0.55) : .clear,
                        lineWidth: 1
                    )
            }
            .scaleEffect(selection == intent || reduceMotion ? 1 : 0.985)
        }
        .buttonStyle(.plain)
        .accessibilityLabel(intent.title)
        .accessibilityValue(selection == intent ? "已选择" : intent.description)
    }

    private func symbol(_ intent: PortraitIntent) -> some View {
        Image(systemName: intent.systemImage)
            .font(.title3.weight(.semibold))
            .foregroundStyle(selection == intent ? FlashShotStyle.accent : .primary)
            .frame(width: 28, height: 28)
    }

    private func select(_ intent: PortraitIntent) {
        guard selection != intent else { return }
        withAnimation(FlashShotMotion.quick(reduceMotion: reduceMotion)) {
            selection = intent
        }
    }
}
