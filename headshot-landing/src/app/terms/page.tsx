export default function TermsPage() {
  return (
    <main className="max-w-3xl mx-auto px-6 py-16">
      <h1 className="text-3xl font-semibold tracking-tight mb-8">Terms of Service</h1>
      <p className="text-stone-500 text-sm mb-8">Last updated: July 1, 2026</p>

      <section className="space-y-6 text-stone-700 leading-relaxed">
        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">1. Acceptance of Terms</h2>
          <p>
            By accessing or using FlashShot (&ldquo;the Service&rdquo;), you agree to be bound by these Terms of Service. If you do not agree to these terms, please do not use the Service.
          </p>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">2. Description of Service</h2>
          <p>
            FlashShot is an AI-powered portrait generation service. Users upload reference photos, select a style, and receive AI-generated portraits. The Service is provided on an &ldquo;as is&rdquo; and &ldquo;as available&rdquo; basis.
          </p>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">3. User Eligibility</h2>
          <ul className="list-disc pl-5 space-y-2">
            <li>You must be at least 18 years old to use the Service.</li>
            <li>You must have the legal right to use and upload the photos you provide.</li>
            <li>You may only upload photos of yourself or photos for which you have explicit permission from the subject.</li>
          </ul>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">4. User Content</h2>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong>Your photos:</strong> You retain all rights to the photos you upload.</li>
            <li><strong>Generated images:</strong> You are granted a personal, non-exclusive license to use the generated portraits for personal or commercial purposes.</li>
            <li><strong>Prohibited content:</strong> You may not upload photos of minors without parental consent, photos containing nudity, violence, hate symbols, or any illegal content.</li>
          </ul>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">5. Prohibited Uses</h2>
          <p>You may not use the Service to:</p>
          <ul className="list-disc pl-5 space-y-2">
            <li>Generate portraits of people without their consent.</li>
            <li>Create deceptive, fraudulent, or misleading content.</li>
            <li>Generate explicit, violent, or hateful imagery.</li>
            <li>Reverse-engineer, scrape, or abuse the Service API.</li>
            <li>Use automated tools to create excessive sessions or drain service resources.</li>
          </ul>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">6. Payments and Refunds</h2>
          <ul className="list-disc pl-5 space-y-2">
            <li>Payments are processed through Paddle. By making a purchase, you agree to Paddle&rsquo;s terms.</li>
            <li>Due to the nature of AI generation (consumable compute resources), <strong>all sales are final</strong> once generation has begun.</li>
            <li>If the Service fails to generate any portraits due to a technical error on our side, you may request a refund within 7 days.</li>
            <li>Refunds are not available for dissatisfaction with style or artistic interpretation.</li>
          </ul>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">7. Intellectual Property</h2>
          <ul className="list-disc pl-5 space-y-2">
            <li>FlashShot retains all rights to the Service, its code, branding, and technology.</li>
            <li>We reserve the right to use anonymized generated portraits for service improvement and marketing, unless you opt out.</li>
            <li>You may not claim that AI-generated portraits were created by human photographers without disclosure.</li>
          </ul>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">8. Limitation of Liability</h2>
          <p>
            To the maximum extent permitted by law, FlashShot and its operators shall not be liable for any indirect, incidental, special, consequential, or punitive damages arising from your use of the Service. Our total liability shall not exceed the amount you paid for the Service in the 12 months preceding the claim.
          </p>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">9. Service Modifications</h2>
          <p>
            We reserve the right to modify, suspend, or discontinue the Service at any time, with or without notice. We are not liable for any modification, suspension, or discontinuation.
          </p>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">10. Governing Law</h2>
          <p>
            These Terms shall be governed by and construed in accordance with the laws of the State of Delaware, United States, without regard to its conflict of law provisions.
          </p>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">11. Contact</h2>
          <p>
            For questions about these Terms, contact us at <a href="mailto:legal@flashshot.ai" className="text-accent underline">legal@flashshot.ai</a>.
          </p>
        </div>
      </section>
    </main>
  );
}
