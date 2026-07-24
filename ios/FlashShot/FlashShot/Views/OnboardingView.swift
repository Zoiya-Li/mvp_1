import SwiftUI

struct OnboardingView: View {
    let chooseStyle: () -> Void
    let useInspiration: () -> Void
    let dismiss: () -> Void

    var body: some View {
        GeometryReader { geometry in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    ZStack(alignment: .topTrailing) {
                    RemotePortraitImage(
                        path: "/api/v2/catalog-images/jp_f_fresh.jpg?v=3"
                    )
                    .frame(maxWidth: .infinity)
                    .frame(height: 330)
                    .clipped()

                    Button(action: dismiss) {
                        Image(systemName: "xmark")
                            .font(.headline)
                            .frame(width: 44, height: 44)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                    .foregroundStyle(FlashShotStyle.accent)
                    .accessibilityLabel("暂时跳过")
                    .padding(.top, geometry.safeAreaInsets.top + 10)
                    .padding(.trailing, 14)
                    }
                    .flashShotEntrance(order: 0)

                    VStack(alignment: .leading, spacing: 24) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("FLASHSHOT")
                            .font(.caption.weight(.bold))
                            .foregroundStyle(FlashShotStyle.accent)
                        Text("把一场写真，装进手机里。")
                            .font(.largeTitle.bold())
                            .fixedSize(horizontal: false, vertical: true)
                        Text("一束光，一个场景，六张属于同一段时光的照片。")
                            .font(.title3)
                            .foregroundStyle(.secondary)
                    }
                    .flashShotEntrance(order: 1)

                    VStack(spacing: 0) {
                        GuideStep(
                            number: 1,
                            title: "选择一套完整写真",
                            detail: "开始前，先看看整套照片的光线、场景和镜头安排。"
                        )
                        Divider().padding(.leading, 58)
                        GuideStep(
                            number: 2,
                            title: "拍下四个真实角度",
                            detail: "跟随三分钟拍摄引导，留下你此刻自然的角度和神情。"
                        )
                        Divider().padding(.leading, 58)
                        GuideStep(
                            number: 3,
                            title: "先确认这就是你",
                            detail: "免费查看一张经过质检的成片；如果不像，可以重新拍。"
                        )
                    }
                    .flashShotEntrance(order: 2)

                    VStack(alignment: .leading, spacing: 6) {
                        Label("这是一间私人影棚，不是公开社区", systemImage: "door.left.hand.closed")
                            .font(.subheadline.weight(.semibold))
                        Text("你的原始照片只用于本次写真创作，并会在 7 天内删除。")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 6)
                    .flashShotEntrance(order: 3)
                    }
                    .padding(.horizontal, 20)
                    .padding(.top, 24)
                    .padding(.bottom, 150)
                    .flashShotContentWidth(620)
                }
            }
            .ignoresSafeArea(edges: .top)
            .safeAreaInset(edge: .bottom) {
                VStack(spacing: 10) {
                    Button("挑选写真主题", action: chooseStyle)
                        .buttonStyle(PrimaryActionButtonStyle())

                    Button("我有想参考的照片", action: useInspiration)
                        .font(.headline)
                        .frame(maxWidth: .infinity, minHeight: 50)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.secondary.opacity(0.35), lineWidth: 1)
                        )
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)
                .padding(.bottom, 8)
                .background(.bar)
            }
        }
    }
}

private struct GuideStep: View {
    let number: Int
    let title: String
    let detail: String

    var body: some View {
        HStack(alignment: .top, spacing: 14) {
            Text(String(number))
                .font(.headline.monospacedDigit())
                .foregroundStyle(.white)
                .frame(width: 34, height: 34)
                .background(FlashShotStyle.ink, in: Circle())

            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                Text(detail)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 15)
        .accessibilityElement(children: .combine)
    }
}
