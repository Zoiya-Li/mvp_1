# FlashShot iOS

Native SwiftUI client for the FlashShot portrait platform. The checked-in
Xcode project is generated from `project.yml`.

## Generate and Build

```bash
xcodegen generate
xcodebuild -project FlashShot.xcodeproj -scheme FlashShot \
  -destination 'generic/platform=iOS Simulator' build
```

The default API is `https://flashshot.top/api`. Automated local runs may set
`FLASHSHOT_API_BASE_URL=http://127.0.0.1:8001/api` in the launch environment.
Production builds should not set that override.

The UI test target uses that local override. Start the API before running the
full scheme so Discover and Library exercise a real local contract:

```bash
cd ../../headshot_pipeline
python -m uvicorn server.main:app --host 127.0.0.1 --port 8001
```

## External Release Configuration

- Development Team `NWUL4BPF68` is set; keep bundle ID `com.flashshot.app`.
- Enable Sign in with Apple and Associated Domains for the App ID.
- Create the consumable `portrait_set_6` in App Store Connect.
- Set the website's public `APPLE_TEAM_ID` for Universal Links.
- Configure the API's Apple App ID, production environment, public root
  certificates, and App Store Server Notifications V2 URL.

No signing key, App Store Connect private key, `.env`, source portrait, or
transaction payload belongs in this directory.
