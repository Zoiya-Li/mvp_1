import XCTest

final class FlashShotUITests: XCTestCase {
    private var app: XCUIApplication!

    override func setUpWithError() throws {
        continueAfterFailure = false
        app = XCUIApplication()
        app.launchEnvironment["FLASHSHOT_API_BASE_URL"] = uiTestAPIBaseURL
        app.launchEnvironment["FLASHSHOT_SKIP_ONBOARDING"] = "1"
        app.launch()
    }

    func testCorePortraitFlowVisualStates() throws {
        let headline = app.staticTexts["今天，想拍怎样的自己？"]
        XCTAssertTrue(headline.waitForExistence(timeout: 12))
        XCTAssertFalse(app.activityIndicators.firstMatch.exists)
        waitForImages()
        XCTAssertFalse(app.descendants(matching: .any)["portrait-image-load-failure"].exists)
        attachScreenshot(named: "01-discover")

        let firstTheme = app.buttons["theme-card-white-cotton-open-shade"]
        XCTAssertTrue(firstTheme.waitForExistence(timeout: 8))
        XCTAssertTrue(firstTheme.isHittable)
        firstTheme.tap()

        let previewButton = app.buttons["开始拍这套写真"]
        XCTAssertTrue(previewButton.waitForExistence(timeout: 10))
        XCTAssertTrue(previewButton.isHittable)
        waitForImages()
        attachScreenshot(named: "02-theme-detail")
        previewButton.tap()

        XCTAssertTrue(app.staticTexts["留下真实的你"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["开始引导拍摄"].exists)
        XCTAssertTrue(app.buttons["检查我的照片"].isHittable)
        attachScreenshot(named: "03-reference-flow")
        app.buttons["关闭"].tap()

        let backButton = app.navigationBars.buttons.firstMatch
        XCTAssertTrue(backButton.waitForExistence(timeout: 5))
        backButton.tap()
        XCTAssertTrue(tabButton("创作").waitForExistence(timeout: 5))

        tabButton("创作").tap()
        XCTAssertTrue(app.staticTexts["从一张灵感图，创作你的写真"].waitForExistence(timeout: 5))
        attachScreenshot(named: "04-custom-inspiration")

        tabButton("相册").tap()
        XCTAssertTrue(app.staticTexts["你的写真集从这里开始"].waitForExistence(timeout: 8))
        attachScreenshot(named: "05-library-empty")

        tabButton("我的").tap()
        XCTAssertTrue(app.staticTexts["在不同设备间保留购买记录和写真"].waitForExistence(timeout: 5))
        attachScreenshot(named: "06-account")
    }

    func testDarkModeVisualStates() throws {
        relaunch(environment: ["FLASHSHOT_UI_COLOR_SCHEME": "dark"])

        XCTAssertTrue(app.staticTexts["今天，想拍怎样的自己？"].waitForExistence(timeout: 12))
        waitForImages()
        attachScreenshot(named: "dark-01-discover")

        tabButton("创作").tap()
        XCTAssertTrue(app.buttons["上传我的灵感图"].waitForExistence(timeout: 5))
        attachScreenshot(named: "dark-02-custom-inspiration")

        tabButton("我的").tap()
        XCTAssertTrue(app.staticTexts["在不同设备间保留购买记录和写真"].waitForExistence(timeout: 5))
        attachScreenshot(named: "dark-03-account")
    }

    func testDiscoverThemeImagesLoadAcrossGrid() throws {
        let headline = app.staticTexts["今天，想拍怎样的自己？"]
        XCTAssertTrue(headline.waitForExistence(timeout: 12))
        let finalTheme = app.buttons["theme-card-black-cloth-hard-light"]

        for _ in 0..<8 {
            waitForImages()
            XCTAssertFalse(app.descendants(matching: .any)["portrait-image-load-failure"].exists)
            if finalTheme.exists { break }
            app.swipeUp()
        }

        waitForImages()
        XCTAssertFalse(app.descendants(matching: .any)["portrait-image-load-failure"].exists)
        XCTAssertTrue(finalTheme.waitForExistence(timeout: 8))
        attachScreenshot(named: "07-discover-all-images")
    }

    func testAccessibilityTextKeepsPrimaryActionsReachable() throws {
        relaunch(arguments: [
            "-UIPreferredContentSizeCategoryName",
            "UICTContentSizeCategoryAccessibilityExtraExtraExtraLarge"
        ])

        XCTAssertTrue(app.staticTexts["今天，想拍怎样的自己？"].waitForExistence(timeout: 12))
        attachScreenshot(named: "a11y-01-discover")

        tabButton("创作").tap()
        let inspirationButton = app.buttons["上传我的灵感图"]
        XCTAssertTrue(inspirationButton.waitForExistence(timeout: 5))
        inspirationButton.tap()
        XCTAssertTrue(app.staticTexts["你想进入的世界"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.buttons["选择一张灵感图"].waitForExistence(timeout: 5))
        attachScreenshot(named: "a11y-02-reference-flow")
    }

    func testFirstLaunchExplainsWorkflowAndKeepsHelpAvailable() throws {
        relaunch(
            environment: ["FLASHSHOT_FORCE_ONBOARDING": "1"],
            skipOnboarding: false
        )

        XCTAssertTrue(app.staticTexts["把一场写真，装进手机里。"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["选择一套完整写真"].exists)
        XCTAssertTrue(app.staticTexts["拍下四个真实角度"].exists)
        XCTAssertTrue(app.staticTexts["先确认这就是你"].exists)
        XCTAssertTrue(app.staticTexts["这是一间私人影棚，不是公开社区"].exists)
        XCTAssertTrue(app.buttons["挑选写真主题"].exists)
        XCTAssertTrue(app.buttons["我有想参考的照片"].exists)
        waitForImages()
        XCTAssertFalse(app.descendants(matching: .any)["portrait-image-load-failure"].exists)
        attachScreenshot(named: "08-first-launch-guide")

        app.buttons["挑选写真主题"].tap()
        XCTAssertTrue(app.staticTexts["今天，想拍怎样的自己？"].waitForExistence(timeout: 5))
        XCTAssertTrue(app.staticTexts["白棉布与晴日"].waitForExistence(timeout: 8))

        tabButton("我的").tap()
        let helpButton = app.buttons["FlashShot 使用指南"]
        XCTAssertTrue(helpButton.waitForExistence(timeout: 5))
        helpButton.tap()
        XCTAssertTrue(app.staticTexts["把一场写真，装进手机里。"].waitForExistence(timeout: 5))
    }

    private func attachScreenshot(named name: String) {
        let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    private func waitForImages() {
        let spinner = app.activityIndicators.firstMatch
        if spinner.exists {
            XCTAssertTrue(spinner.waitForNonExistence(timeout: 10))
        }
    }

    private func relaunch(
        arguments: [String] = [],
        environment: [String: String] = [:],
        skipOnboarding: Bool = true
    ) {
        app.terminate()
        app = XCUIApplication()
        app.launchEnvironment["FLASHSHOT_API_BASE_URL"] = uiTestAPIBaseURL
        if skipOnboarding {
            app.launchEnvironment["FLASHSHOT_SKIP_ONBOARDING"] = "1"
        }
        for (key, value) in environment {
            app.launchEnvironment[key] = value
        }
        app.launchArguments = arguments
        app.launch()
    }

    private func tabButton(_ title: String) -> XCUIElement {
        let compactTab = app.tabBars.buttons[title].firstMatch
        return compactTab.exists ? compactTab : app.buttons[title].firstMatch
    }

    private var uiTestAPIBaseURL: String {
        ProcessInfo.processInfo.environment["FLASHSHOT_UI_TEST_API_BASE_URL"]
            ?? "https://flashshot.top/api"
    }
}
