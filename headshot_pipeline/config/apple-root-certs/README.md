# Apple root certificates

These public DER certificates are the trust anchors used by
`AppleIAPVerifier` for StoreKit 2 signed transactions.

Source: `https://www.apple.com/certificateauthority/`

- `AppleRootCA-G2.cer`
  - SHA-256: `C2:B9:B0:42:DD:57:83:0E:7D:11:7D:AC:55:AC:8A:E1:94:07:D3:8E:41:D8:8F:32:15:BC:3A:89:04:44:A0:50`
- `AppleRootCA-G3.cer`
  - SHA-256: `63:34:3A:BF:B8:9A:6A:03:EB:B5:7E:9B:3F:5F:A7:BE:7C:4F:5C:75:6F:30:17:B3:A8:C4:88:C3:65:3E:91:79`

`deploy/overseas-vps/fetch_apple_root_certs.sh` refreshes the same files from
Apple during a clean server setup. Never replace these with certificates from
a third-party mirror.
