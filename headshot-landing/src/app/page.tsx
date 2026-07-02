import { Nav } from "@/components/Nav";
import { Hero } from "@/components/Hero";
import { Workflow } from "@/components/Workflow";
import { Gallery } from "@/components/Gallery";
import { Pricing } from "@/components/Pricing";
import { Privacy } from "@/components/Privacy";
import { FAQ } from "@/components/FAQ";
import { CTA } from "@/components/CTA";
import { Footer } from "@/components/Footer";

export default function Home() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <Workflow />
        <Gallery />
        <Pricing />
        <Privacy />
        <FAQ />
        <CTA />
      </main>
      <Footer />
    </>
  );
}
