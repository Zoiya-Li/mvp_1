import Foundation

struct GuidedCapturePrompt: Identifiable, Equatable {
    let id: String
    let title: String
    let instruction: String
    let systemImage: String
}

enum GuidedCapturePlan {
    static let identitySession: [GuidedCapturePrompt] = [
        GuidedCapturePrompt(
            id: "front",
            title: "看向镜头",
            instruction: "看着镜头，轻轻呼气，让表情放松下来。请把完整面部留在取景框内。",
            systemImage: "person.crop.square"
        ),
        GuidedCapturePrompt(
            id: "turn_left",
            title: "向左迎着光",
            instruction: "头稍微向左转，直到脸上的光影发生变化。保持双眼可见，肩膀放松。",
            systemImage: "arrow.turn.up.left"
        ),
        GuidedCapturePrompt(
            id: "turn_right",
            title: "向右看远一点",
            instruction: "头稍微向右转，视线越过手机。保持这个角度，不必刻意维持表情。",
            systemImage: "arrow.turn.up.right"
        ),
        GuidedCapturePrompt(
            id: "expression",
            title: "让表情自然发生",
            instruction: "先看向别处，再回到镜头。细微变化的神情，比一直保持微笑更真实。",
            systemImage: "face.smiling"
        ),
    ]
}
