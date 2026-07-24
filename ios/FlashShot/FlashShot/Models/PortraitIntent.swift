import Foundation

enum PortraitIntent: String, CaseIterable, Identifiable {
    case authentic
    case confident
    case cinematic

    var id: String { rawValue }

    var title: String {
        switch self {
        case .authentic: "还是我"
        case .confident: "从容有力量"
        case .cinematic: "走进故事里"
        }
    }

    var description: String {
        switch self {
        case .authentic: "温暖、自然，一眼就认得出是你。"
        case .confident: "克制、精致，把你的自信轻轻带出来。"
        case .cinematic: "去到一个未曾经历，却仿佛属于你的故事。"
        }
    }

    var systemImage: String {
        switch self {
        case .authentic: "person.crop.circle"
        case .confident: "sparkle"
        case .cinematic: "film.stack"
        }
    }

    var previewLine: String {
        switch self {
        case .authentic: "熟悉、真实，仍然完完全全是你。"
        case .confident: "还是同一个你，只是让自信更靠前一点。"
        case .cinematic: "以你的面孔为中心，为你搭起一个世界。"
        }
    }

    func ranks(_ theme: PortraitTheme) -> Int {
        let preferredSlugs: [String]
        switch self {
        case .authentic:
            preferredSlugs = [
                "lifestyle-portrait", "japanese-korean-portrait", "intellectual-artistic"
            ]
        case .confident:
            preferredSlugs = [
                "urban-professional", "fashion-editorial", "chinese-traditional"
            ]
        case .cinematic:
            preferredSlugs = [
                "cinematic-mood", "creative-special", "chinese-traditional"
            ]
        }
        return preferredSlugs.firstIndex(of: theme.slug) ?? preferredSlugs.count + 1
    }
}
