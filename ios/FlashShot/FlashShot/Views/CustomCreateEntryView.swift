import SwiftUI

struct CustomCreateEntryView: View {
    @State private var creating = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                Image(systemName: "photo.on.rectangle.angled")
                    .font(.system(size: 44))
                    .foregroundStyle(FlashShotStyle.accent)
                    .frame(width: 76, height: 76)
                    .background(FlashShotStyle.accent.opacity(0.1), in: Circle())
                    .flashShotEntrance(order: 0)
                Text("从一张灵感图，创作你的写真")
                    .font(.largeTitle.bold())
                    .flashShotEntrance(order: 1)
                Text("上传你喜欢的光线、服装、场景或构图。照片中的人物不会成为身份参考，你的脸只来自你自己的照片。")
                    .font(.title3)
                    .foregroundStyle(.secondary)
                    .flashShotEntrance(order: 2)

                Button("上传我的灵感图") { creating = true }
                    .buttonStyle(PrimaryActionButtonStyle())
                    .flashShotEntrance(order: 3)
            }
            .padding(20)
            .flashShotContentWidth()
        }
        .navigationTitle("自由创作")
        .sheet(isPresented: $creating) {
            NavigationStack {
                CreateFlowView(
                    theme: nil,
                    source: .privateInspiration,
                    intent: .authentic
                )
            }
        }
        .sensoryFeedback(.impact(weight: .medium), trigger: creating) { _, newValue in
            newValue
        }
    }
}
