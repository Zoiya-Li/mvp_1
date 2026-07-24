import SwiftUI
import UIKit

struct ReferencePhotoThumbnail: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let image: UIImage
    let cue: ReferencePhotoCue?
    let serverFeedback: ReferenceRoleFeedback?
    @State private var scanPosition: CGFloat = -1

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            ZStack {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(maxWidth: .infinity)
                    .aspectRatio(0.82, contentMode: .fit)
                    .clipped()

                if serverFeedback == nil && cue?.tone == .reading {
                    GeometryReader { geometry in
                        Rectangle()
                            .fill(
                                LinearGradient(
                                    colors: [.clear, .white.opacity(0.72), .clear],
                                    startPoint: .top,
                                    endPoint: .bottom
                                )
                            )
                            .frame(height: 30)
                            .offset(y: scanPosition * geometry.size.height)
                    }
                    .allowsHitTesting(false)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay {
                RoundedRectangle(cornerRadius: 6)
                    .stroke(borderColor, lineWidth: serverFeedback == nil ? 1.5 : 2.5)
            }
            .overlay(alignment: .topTrailing) {
                Image(systemName: statusSymbol)
                    .font(.body.weight(.semibold))
                    .foregroundStyle(statusColor)
                    .symbolEffect(.bounce, value: statusSymbol)
                    .background(.white, in: Circle())
                    .padding(7)
            }

            Label(message, systemImage: statusSymbol)
                .font(.caption2)
                .foregroundStyle(messageColor)
                .lineLimit(3)
                .fixedSize(horizontal: false, vertical: true)
        }
        .onAppear {
            guard !reduceMotion, serverFeedback == nil, cue?.tone == .reading else { return }
            withAnimation(.easeInOut(duration: 1.15).repeatCount(2, autoreverses: false)) {
                scanPosition = 1
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var message: String {
        if let serverFeedback {
            return serverFeedback.pass
                ? "可以用于写真创作"
                : serverFeedback.nextStep
        }
        return cue?.message ?? "准备进一步检查"
    }

    private var statusSymbol: String {
        if let serverFeedback {
            return serverFeedback.pass ? "checkmark.circle.fill" : "arrow.clockwise.circle.fill"
        }
        return cue?.systemImage ?? "viewfinder"
    }

    private var statusColor: Color {
        if let serverFeedback {
            return serverFeedback.pass ? FlashShotStyle.jade : FlashShotStyle.accent
        }
        switch cue?.tone {
        case .encouraging: return FlashShotStyle.jade
        case .gentleFix: return FlashShotStyle.accent
        default: return .white
        }
    }

    private var borderColor: Color {
        if let serverFeedback {
            return serverFeedback.pass ? FlashShotStyle.jade : FlashShotStyle.accent
        }
        switch cue?.tone {
        case .encouraging: return FlashShotStyle.jade.opacity(0.85)
        case .gentleFix: return FlashShotStyle.accent.opacity(0.85)
        default: return .white.opacity(0.75)
        }
    }

    private var messageColor: Color {
        if serverFeedback?.pass == false || cue?.tone == .gentleFix {
            return FlashShotStyle.accent
        }
        return .secondary
    }
}
