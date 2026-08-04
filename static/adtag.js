/*
 * adtag.js — the snippet a publisher embeds to show ads.
 *
 *   <div data-adplatform-slot
 *        data-publisher="pub_demo"
 *        data-placement="plc_sidebar"
 *        data-key="pk_test_..."></div>
 *   <script src="http://localhost:8000/static/adtag.js" async></script>
 *
 * Every div with data-adplatform-slot is filled independently, so one page can
 * carry several placements.
 *
 * ---------------------------------------------------------------------------
 * THE API KEY IS PUBLIC HERE. THIS IS A REAL PROBLEM.
 * ---------------------------------------------------------------------------
 * A browser tag cannot hold a secret. Anyone who views source has the
 * publisher's key and can mint bid requests attributed to that publisher.
 * Right now the only thing standing between that and fraud is BID_RATE_LIMIT.
 *
 * Real exchanges do not solve this with a browser key. They either (a) have the
 * publisher's own server sign the request, so the browser never sees a
 * credential, or (b) issue short-lived, origin-bound tokens. The gateway's
 * DynamicCORSMiddleware helps a little — it only echoes CORS headers to
 * registered publisher domains — but CORS is enforced by browsers, not by
 * servers, so curl ignores it entirely.
 *
 * Treat a leaked publisher key as low-severity but real: bounded by rate limit,
 * mitigated by invalid-traffic detection, which does not exist yet.
 *
 * ---------------------------------------------------------------------------
 * WHY THE CREATIVE RENDERS IN A SANDBOXED IFRAME
 * ---------------------------------------------------------------------------
 * creative_html is advertiser-supplied and is not sanitised anywhere in the
 * pipeline. Assigning it to innerHTML would execute advertiser JavaScript in
 * the publisher's origin — access to their cookies, their DOM, their session.
 * That is a stored XSS with an upload form attached to it.
 *
 * So the creative goes into an iframe with an explicit sandbox allowlist. The
 * frame gets no same-origin access, cannot navigate the top window on its own,
 * and can only open a new tab when the user actually clicks. Even a fully
 * malicious creative is then contained to its own rectangle.
 *
 * This is defence in depth, not a substitute for creative review. It stops the
 * creative reaching the publisher's page; it does not stop the creative being
 * malware, and it does not stop it lying about what it advertises.
 */
(function () {
  "use strict";

  var ENDPOINT =
    (document.currentScript && new URL(document.currentScript.src).origin) ||
    "http://localhost:8000";

  function deviceType() {
    var ua = navigator.userAgent;
    if (/iPad|Android(?!.*Mobile)|Tablet/i.test(ua)) return "tablet";
    if (/Mobi|Android|iPhone/i.test(ua)) return "mobile";
    return "desktop";
  }

  /*
   * Keywords the page wants the auction to know about. Real integrations pass
   * article tags; here we read the meta keywords tag and fall back to nothing.
   * Never send free page text — it is unbounded, and it is user data you would
   * then be storing under GDPR.
   */
  function pageKeywords() {
    var meta = document.querySelector('meta[name="keywords"]');
    if (!meta || !meta.content) return [];
    return meta.content
      .split(",")
      .map(function (k) { return k.trim().toLowerCase(); })
      .filter(Boolean)
      .slice(0, 10);
  }

  /*
   * A stable-ish id so pair CTR features have something to key on. Not a
   * tracking cookie, not shared across sites, and cleared when the tab closes.
   * A production tag needs a consent check before setting even this.
   */
  function userId() {
    try {
      var k = "_adp_uid";
      var v = sessionStorage.getItem(k);
      if (!v) {
        v = "u_" + Math.random().toString(36).slice(2, 12);
        sessionStorage.setItem(k, v);
      }
      return v;
    } catch (e) {
      return "u_anon";           // storage blocked: still biddable
    }
  }

  function renderEmpty(slot, message) {
    slot.setAttribute("data-adplatform-state", "empty");
    slot.innerHTML =
      '<div style="font:12px system-ui;color:#999;padding:8px">' +
      message +
      "</div>";
  }

  function renderCreative(slot, bid) {
    var frame = document.createElement("iframe");

    // The allowlist, and why each entry is here:
    //   allow-popups .......................... the click link opens a new tab
    //   allow-popups-to-escape-sandbox ........ the advertiser's landing page
    //                                           must load normally, not inherit
    //                                           this sandbox
    //   allow-top-navigation-by-user-activation  a click may navigate the top
    //                                           window; a script alone may not
    // Deliberately absent: allow-same-origin (would defeat the whole point when
    // combined with allow-scripts) and allow-scripts itself. A creative that
    // needs JS to render is a creative that needs review first.
    frame.setAttribute(
      "sandbox",
      "allow-popups allow-popups-to-escape-sandbox allow-top-navigation-by-user-activation"
    );
    frame.setAttribute("referrerpolicy", "no-referrer");
    frame.setAttribute("loading", "lazy");
    frame.style.cssText =
      "width:100%;height:100%;border:0;display:block;background:transparent";

    // target="_blank" so the click leaves the frame rather than navigating
    // inside it, which would leave the publisher with a blank rectangle.
    frame.srcdoc =
      "<!doctype html><meta charset=utf-8>" +
      "<base target=_blank>" +
      "<style>html,body{margin:0;height:100%;font:14px/1.4 system-ui,sans-serif}" +
      "a{display:flex;flex-direction:column;justify-content:center;height:100%;" +
      "padding:12px;box-sizing:border-box;text-decoration:none;color:inherit;" +
      "background:#fff}" +
      ".ad__brand{font-size:11px;text-transform:uppercase;letter-spacing:.06em;" +
      "color:#6b7280;margin-bottom:4px}" +
      ".ad__headline{font-size:15px;font-weight:600;color:#111827}</style>" +
      bid.ad_markup;

    slot.setAttribute("data-adplatform-state", "filled");
    slot.setAttribute("data-adplatform-impression", bid.impression_id);
    slot.innerHTML = "";
    slot.appendChild(frame);
  }

  function fill(slot) {
    var publisher = slot.getAttribute("data-publisher");
    var placement = slot.getAttribute("data-placement");
    var key = slot.getAttribute("data-key");

    if (!publisher || !placement || !key) {
      renderEmpty(slot, "ad slot misconfigured");
      return;
    }

    slot.setAttribute("data-adplatform-state", "loading");

    fetch(ENDPOINT + "/v1/bid", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-API-Key": key },
      body: JSON.stringify({
        publisher_id: publisher,
        placement_id: placement,
        user_id: userId(),
        device_type: deviceType(),
        page_url: location.href,
        page_keywords: pageKeywords(),
        timestamp_ms: Date.now()
      })
    })
      .then(function (r) {
        // 204 is a no-fill: a normal, frequent outcome, not an error. Every ad
        // slot must degrade to empty space rather than a broken layout.
        if (r.status === 204) return null;
        if (!r.ok) throw new Error("bid failed: " + r.status);
        return r.json();
      })
      .then(function (bid) {
        if (!bid) { renderEmpty(slot, ""); return; }
        renderCreative(slot, bid);
      })
      .catch(function (err) {
        // Never let an ad failure break the publisher's page.
        if (window.console) console.warn("[adplatform]", err.message);
        renderEmpty(slot, "");
      });
  }

  function init() {
    var slots = document.querySelectorAll("[data-adplatform-slot]");
    for (var i = 0; i < slots.length; i++) fill(slots[i]);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  // Exposed so the demo page can force a re-fill without reloading.
  window.adplatform = { refresh: init };
})();
