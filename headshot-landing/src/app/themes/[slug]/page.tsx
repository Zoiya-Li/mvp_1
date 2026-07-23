import { ThemeExperience } from "@/components/portrait/ThemeExperience";

export default async function ThemePage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  return <ThemeExperience slug={slug} />;
}

