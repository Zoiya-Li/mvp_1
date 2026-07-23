export default function TermsPage() {
  return (
    <main className="max-w-3xl mx-auto px-6 py-16">
      <h1 className="text-3xl font-semibold tracking-tight mb-8">Terms of Service</h1>
      <p className="text-stone-500 text-sm mb-8">Last updated: July 14, 2026</p>

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
            <li>You may only upload photos of yourself. Private inspiration images must be images you have the right to use as style references.</li>
          </ul>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">4. User Content</h2>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong>Your photos:</strong> You retain all rights to the photos you upload.</li>
            <li><strong>Generated images:</strong> You are granted a personal, non-exclusive license to use the generated portraits for personal or commercial purposes.</li>
            <li><strong>Prohibited content:</strong> You may not upload photos of minors, non-consenting people, nudity, exploitative content, violence, hate symbols, or illegal content.</li>
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
            <li>Purchases made in the iOS app are processed by Apple as in-app purchases and are subject to Apple&rsquo;s applicable media services terms. Website purchases, when available, are processed by Paddle.</li>
            <li>Prices, taxes, and the applicable payment provider are shown before payment. Statutory consumer rights are not waived.</li>
            <li>If the Service fails to deliver the purchased portrait set because of a technical error on our side, contact support within 7 days for remediation or refund review.</li>
            <li>Refund and chargeback handling is subject to the purchasing platform&rsquo;s rules and applicable law.</li>
          </ul>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">7. Intellectual Property</h2>
          <ul className="list-disc pl-5 space-y-2">
            <li>FlashShot retains all rights to the Service, its code, branding, and technology.</li>
            <li>Your portraits remain private unless you explicitly create a public share link. We do not use private portraits for marketing or model training without a separate opt-in.</li>
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
          <h2 className="text-xl font-semibold text-stone-900 mb-3">10. Contact</h2>
          <p>
            For questions about these Terms, contact us at <a href="mailto:support@flashshot.top" className="text-accent underline">support@flashshot.top</a>.
          </p>
        </div>
      </section>
    </main>
  );
}
