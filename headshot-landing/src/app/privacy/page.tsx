export default function PrivacyPage() {
  return (
    <main className="max-w-3xl mx-auto px-6 py-16">
      <h1 className="text-3xl font-semibold tracking-tight mb-8">Privacy Policy</h1>
      <p className="text-stone-500 text-sm mb-8">Last updated: July 1, 2026</p>

      <section className="space-y-6 text-stone-700 leading-relaxed">
        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">1. Overview</h2>
          <p>
            FlashShot (&ldquo;we&rdquo;, &ldquo;us&rdquo;, or &ldquo;our&rdquo;) is an AI portrait generation service. This Privacy Policy explains how we collect, use, store, and protect your information when you use our website and services.
          </p>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">2. Information We Collect</h2>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong>Photos you upload:</strong> Selfie/reference photos you provide for portrait generation.</li>
            <li><strong>Generated images:</strong> AI-generated portraits created from your photos.</li>
            <li><strong>Usage data:</strong> Session metadata, style preferences, and anonymous quality feedback.</li>
            <li><strong>Payment information:</strong> Processed by our payment provider (Paddle). We do not store full payment card details.</li>
          </ul>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">3. How We Use Your Photos</h2>
          <ul className="list-disc pl-5 space-y-2">
            <li>Your photos are used <strong>only</strong> to generate portraits for your current session.</li>
            <li>We perform temporary facial feature analysis to preserve identity in generated images.</li>
            <li>We do <strong>not</strong> use your photos to train AI models.</li>
            <li>We do <strong>not</strong> build a long-term facial recognition database.</li>
            <li>We do <strong>not</strong> perform cross-user face searches or matching.</li>
          </ul>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">4. Data Retention</h2>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong>Uploaded photos:</strong> Deleted within 7 days after your session ends.</li>
            <li><strong>Generated portraits:</strong> Available for download for 30 days, then deleted.</li>
            <li><strong>Session metadata:</strong> Retained for 90 days for customer support and analytics, then anonymized.</li>
            <li><strong>Payment records:</strong> Retained as required by law (typically 7 years).</li>
          </ul>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">5. Third-Party Services</h2>
          <p>We use the following third-party services:</p>
          <ul className="list-disc pl-5 space-y-2">
            <li><strong>OpenRouter:</strong> For AI image generation. Your photos are sent to their API for processing.</li>
            <li><strong>Paddle:</strong> For payment processing. Subject to Paddle&rsquo;s privacy policy.</li>
            <li><strong>Hosting provider:</strong> For server infrastructure and data storage.</li>
          </ul>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">6. Cookies</h2>
          <p>
            We use essential cookies to maintain your session state and authentication. We do not use tracking cookies for advertising purposes.
          </p>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">7. Your Rights</h2>
          <p>You have the right to:</p>
          <ul className="list-disc pl-5 space-y-2">
            <li>Access the personal data we hold about you.</li>
            <li>Request deletion of your data and photos.</li>
            <li>Export your generated portraits.</li>
            <li>Withdraw consent at any time.</li>
          </ul>
          <p className="mt-2">
            To exercise these rights, contact us at <a href="mailto:privacy@flashshot.ai" className="text-accent underline">privacy@flashshot.ai</a>.
          </p>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">8. Children&rsquo;s Privacy</h2>
          <p>
            FlashShot is not intended for users under 13 years of age. We do not knowingly collect data from children. If you believe a child has used our service, please contact us immediately.
          </p>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">9. Changes to This Policy</h2>
          <p>
            We may update this Privacy Policy from time to time. We will notify you of significant changes via email or a prominent notice on our website.
          </p>
        </div>

        <div>
          <h2 className="text-xl font-semibold text-stone-900 mb-3">10. Contact Us</h2>
          <p>
            If you have any questions about this Privacy Policy, please contact us at <a href="mailto:privacy@flashshot.ai" className="text-accent underline">privacy@flashshot.ai</a>.
          </p>
        </div>
      </section>
    </main>
  );
}
