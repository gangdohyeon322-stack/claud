// LearnBox 서비스 워커 (HTTPS 접속일 때만 등록됨).
// - 앱 화면(같은 출처)은 네트워크 우선, 실패하면 캐시 (오프라인에서도 열림).
// - 온보드 실행 엔진(onnxruntime-web 1.18.0, jsDelivr)은 처음 쓸 때 받아 두고 그다음부터 캐시 우선.
// - /api/ 요청은 절대 건드리지 않는다 (로그인 토큰, 자료, 모델 파일은 캐시하지 않음).
const CACHE = "learnbox-shell-v2";
const ORT_CACHE = "learnbox-ort-1.18.0"; // 버전이 주소에 고정돼 있으므로 앱 캐시를 올려도 다시 받지 않는다
const ORT_PREFIX = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/";
const SHELL = ["/", "/app.js", "/onboard.js", "/style.css", "/manifest.webmanifest", "/icon.svg"];
const KEEP = [CACHE, ORT_CACHE];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) =>
      // 하나가 실패해도 설치는 계속한다 (나머지는 처음 열 때 채워짐)
      Promise.all(SHELL.map((u) => c.add(new Request(u, { cache: "reload" })).catch(() => null)))
    ).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((ks) => Promise.all(ks.filter((k) => k.startsWith("learnbox-") && !KEEP.includes(k)).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("message", (e) => {
  if (e.data === "skipWaiting") self.skipWaiting();
});

function offlineResponse() {
  return new Response("오프라인 상태입니다.", { status: 503, statusText: "Offline", headers: { "Content-Type": "text/plain; charset=utf-8" } });
}

// 캐시 우선: onnxruntime-web 파일 (ort.min.js, *.wasm)
async function cacheFirst(req) {
  const cache = await caches.open(ORT_CACHE);
  const hit = await cache.match(req, { ignoreVary: true });
  if (hit) return hit;
  const res = await fetch(req);
  if (res && (res.ok || res.type === "opaque")) {
    cache.put(req, res.clone()).catch(() => {});
  }
  return res;
}

// 네트워크 우선: 같은 출처의 앱 화면
async function networkFirst(req) {
  try {
    const res = await fetch(req);
    if (res && res.ok && res.type === "basic") {
      const copy = res.clone();
      caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
    }
    return res;
  } catch (err) {
    const cache = await caches.open(CACHE);
    const hit = await cache.match(req, { ignoreSearch: req.mode === "navigate" });
    if (hit) return hit;
    if (req.mode === "navigate") {
      const home = await cache.match("/");
      if (home) return home;
    }
    return offlineResponse();
  }
}

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.href.startsWith(ORT_PREFIX)) {
    e.respondWith(cacheFirst(req));
    return;
  }
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/") || url.pathname === "/api") return; // API는 절대 캐시하지 않음
  if (req.headers.has("Authorization") || req.headers.has("Range")) return;
  e.respondWith(networkFirst(req));
});
