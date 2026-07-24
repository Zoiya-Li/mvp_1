import SwiftUI

enum FlashShotMotion {
    static func quick(reduceMotion: Bool) -> Animation {
        reduceMotion ? .easeOut(duration: 0.16) : .spring(duration: 0.32, bounce: 0.16)
    }

    static func gentle(reduceMotion: Bool, delay: Double = 0) -> Animation {
        if reduceMotion {
            return .easeOut(duration: 0.2)
        }
        return .spring(duration: 0.56, bounce: 0.12).delay(delay)
    }

    static func stageTransition(reduceMotion: Bool) -> AnyTransition {
        if reduceMotion { return .opacity }
        return .asymmetric(
            insertion: .opacity.combined(with: .move(edge: .trailing)),
            removal: .opacity.combined(with: .scale(scale: 0.985))
        )
    }
}
