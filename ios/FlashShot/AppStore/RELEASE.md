# FlashShot App Store Release

## Product Metadata

- Name: FlashShot AI Portraits
- Subtitle: AI portrait stories that look like you
- Bundle ID: `com.flashshot.app`
- Primary category: Photo & Video
- Secondary category: Lifestyle
- Privacy URL: `https://flashshot.top/privacy`
- Support URL: `https://flashshot.top/account`
- Marketing URL: `https://flashshot.top`
- App Store Connect app ID: `6791045259`
- App record SKU: `flashshot-ios-2026`
- In-app purchase: `portrait_set_6` (consumable, Apple ID `6791048790`)
- IAP pricing: U.S. base `$9.99`; China mainland manually set to `CNY 29.00`;
  Apple-adjusted equivalent pricing in the other 174 storefronts

Suggested keywords: `portrait,photo,studio,headshot,ai,style,photoshoot,picture`

## Description

FlashShot turns four to six identity photos into a directed six-photo portrait
story. Choose an official portrait direction or privately upload an inspiration
image for its composition, light, wardrobe, background, lens, and mood. The
person in an inspiration image is never used as your identity.

The first reviewed portrait is free. If it feels right, a one-time in-app
purchase unlocks the remaining five compositions. Finished portraits can be
saved to Photos, and portrait recipes can be shared without publishing your
portrait pixels.

## App Review Notes

1. Launch the app and select any theme in Discover, or choose Create to upload
   a private inspiration image.
2. Select four to six clear photos of the same consenting adult. Confirm face
   processing consent and that the subject is at least 18.
3. FlashShot generates one free portrait. Generation can take several minutes;
   the project remains visible in Library if the app is closed.
4. Sign in with Apple before purchasing. The consumable
   `portrait_set_6` unlocks exactly one project's five remaining portraits.
5. The backend verifies Apple's signed transaction before generation is
   unlocked. Purchases and delivered projects are restored through the user's
   Apple-linked FlashShot account.
6. Private source photos are retained for up to 7 days, generated portraits for
   up to 30 days, and minimal operational metadata for up to 90 days.

No external purchase link appears in the iOS app. The website is used for
public theme browsing, privacy information, and recipe sharing.

## TestFlight Notes

Exercise official-theme and private-inspiration creation, background/foreground
recovery, StoreKit sandbox purchase, project deletion, save to Photos, system
Share Sheet, and a `https://flashshot.top/s/...` Universal Link. Report the
project ID shown in the Library when contacting support; never send source
photos by email.

Also verify an interrupted purchase and Ask to Buy. The app retries product
loading after returning to the foreground and preserves a pending project until
Apple approves or cancels the transaction, preventing accidental workspace or
project deletion while approval is outstanding.

## Release Assets

- App icon: 1024x1024 RGB PNG with no alpha channel.
- iPhone screenshots: four native iPhone 17 Pro Max captures at 1320x2868 in
  `Screenshots/iPhone-6.9/`.
- iPad screenshots: four native iPad Pro 13-inch captures at 2064x2752 in
  `Screenshots/iPad-13/`.
- The screenshot flow covers theme discovery, theme detail, private identity
  photos, and user-provided portrait inspiration. Library-empty and account
  screens are intentionally excluded from the storefront set.

## App Store Connect Checklist

- App ID `com.flashshot.app` is registered with Sign in with Apple and
  Associated Domains enabled.
- Team `NWUL4BPF68` is set in Xcode and in the website runtime environment;
  both AASA domains return the full application identifier.
- Complete availability, localization, and review screenshot for the created
  consumable IAP `portrait_set_6` (Apple ID `6791048790`); its price schedule is
  already configured.
- Configure App Store Server Notifications V2 to
  `https://flashshot.top/api/v2/apple/notifications`.
- Install Apple root certificates on the API host and set the numeric
  `APPLE_APP_ID` plus production IAP environment.
- Complete App Privacy answers to match `PrivacyInfo.xcprivacy` and the website
  policy.
- Upload the reviewed iPhone and iPad screenshots from `Screenshots/`.

## Local Release Audit

- Current iOS suite: 17/17 unit tests and 5/5 UI tests passed on the iPhone 17
  Pro simulator, including portrait-intent ranking, first-launch guidance, the
  persistent help entry, remote-image retry and caching, all eight Discover
  covers, large-photo preparation, pre-generation reference checking, the
  feedback-conditioned preview retry contract, sticky action reachability,
  animated intent reordering, dark mode, and accessibility text coverage.
- The current interaction audit covers staged onboarding and Discover entrance,
  intent-selection feedback, animated theme reordering, sticky creation actions,
  stage transitions, and help re-entry in a 90-second simulator recording. All
  custom motion respects the system Reduce Motion setting.
- Current signed iPhoneOS Release build and device archive: passed with Apple
  Development signing, explicit `com.flashshot.app` provisioning, version
  `1.0.0`, and build `1`.
- Production portrait E2E: exact six unique 896x1200 assets, share recipe, and
  isolated user cleanup passed through OpenRouter.
- App Store icon and all eight storefront screenshots are present and were
  validated for dimensions and alpha channel.

The current Apple Program License Agreement is accepted. Xcode created a local
Apple Development certificate and an explicit managed provisioning profile for
`com.flashshot.app`; the profile includes Sign in with Apple and Associated
Domains, and a signed device archive succeeds. App Store Connect now contains
the `FlashShot AI Portraits` app record and consumable IAP. The numeric app ID is
configured in production, and an App Store distribution IPA was exported to
`/tmp/FlashShotExport/FlashShot.ipa`. TestFlight remains gated by IAP metadata,
the Paid Applications Agreement, server notification configuration, build
upload, and sandbox verification.
