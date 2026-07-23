import { SharedLook } from "@/components/portrait/SharedLook";

export default async function SharedLookPage({ params }: { params: Promise<{ token: string }> }) {
  const { token } = await params;
  return <SharedLook token={token} />;
}
