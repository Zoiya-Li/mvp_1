self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  // Private projects, face photos, generated portraits, and API responses are
  // deliberately never cached by the service worker.
  event.respondWith(fetch(event.request));
});
