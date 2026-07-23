import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export function GET() {
  const teamID = process.env.APPLE_TEAM_ID?.trim();
  const bundleID = process.env.APPLE_BUNDLE_ID?.trim() || "com.flashshot.app";

  if (!teamID) {
    return NextResponse.json(
      { error: "Universal Links are not configured" },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }

  return NextResponse.json(
    {
      applinks: {
        details: [
          {
            appIDs: [`${teamID}.${bundleID}`],
            components: [{ "/": "/s/*", comment: "Shared portrait recipes" }],
          },
        ],
      },
      webcredentials: { apps: [`${teamID}.${bundleID}`] },
    },
    {
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "public, max-age=300, s-maxage=3600",
      },
    },
  );
}
