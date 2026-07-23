import { Suspense } from "react";

import { PortraitCreateStudio } from "@/components/portrait/PortraitCreateStudio";

export default function CreatePage() {
  return (
    <Suspense fallback={<div className="portal-loading">Opening your private studio...</div>}>
      <PortraitCreateStudio />
    </Suspense>
  );
}
