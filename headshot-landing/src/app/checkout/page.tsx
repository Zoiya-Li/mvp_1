import { Suspense } from "react";

import { PortraitCheckout } from "@/components/portrait/PortraitCheckout";

export default function CheckoutPage() {
  return <Suspense fallback={<div className="portal-loading">Preparing checkout...</div>}><PortraitCheckout /></Suspense>;
}
