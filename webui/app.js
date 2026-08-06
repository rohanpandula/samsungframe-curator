(function () {
  "use strict";

  var state = {
    entries: [],
    decisions: {},
    selected: null,
    lastRender: null,
    reviewEntries: [],
    reviewFilter: "pending",
    tasteProfile: null,
  };

  // The pair the Taste Deck last fetched via GET /api/taste/pair — echoed back
  // on vote so the server can revalidate it hasn't changed underneath the user.
  var currentTastePair = null;

  var $ = function (id) {
    return document.getElementById(id);
  };

  function setStatus(text, isError) {
    var status = $("status");
    status.textContent = text;
    status.classList.toggle("error", Boolean(isError));
  }

  var toastTimer = null;
  function showToast(text, isError) {
    var toast = $("message");
    toast.textContent = text;
    toast.classList.toggle("error", Boolean(isError));
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      toast.classList.remove("show");
    }, 3500);
  }

  async function fetchJSON(url, options) {
    var resp = await fetch(url, options);
    if (!resp.ok) {
      var detail = "";
      try {
        var body = await resp.json();
        detail = body && body.detail ? " — " + JSON.stringify(body.detail) : "";
      } catch (e) {
        /* non-JSON error body */
      }
      throw new Error(url + " failed (" + resp.status + ")" + detail);
    }
    return resp.json();
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // -- catalog grid ----------------------------------------------------------

  function renderCards() {
    var grid = $("catalog-grid");
    if (!state.entries.length) {
      var empty = el("p", "empty", "No catalog entries.");
      grid.replaceChildren(empty);
      return;
    }
    var cards = state.entries.map(function (entry) {
      var decision = state.decisions[entry.asset_id] || "pending";
      var card = el("button", "card");
      card.type = "button";
      card.setAttribute("tabindex", "0");
      card.setAttribute(
        "aria-pressed",
        state.selected && state.selected.asset_id === entry.asset_id ? "true" : "false"
      );

      var name = el("span", "card-name", entry.asset_id);
      var decisionBadge = el("span", "decision " + decision, decision);
      var sha = el("span", "card-sub", "sha " + (entry.sha256 || "").slice(0, 10) + "…");
      var score = el(
        "span",
        "card-score",
        entry.quality_score != null ? "quality " + Number(entry.quality_score).toFixed(2) : ""
      );

      card.append(decisionBadge, name, sha);
      if (score.textContent) card.append(score);

      card.addEventListener("click", function () {
        select(entry);
      });
      card.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          select(entry);
        }
      });
      if (state.selected && state.selected.asset_id === entry.asset_id) {
        card.classList.add("selected");
      }
      return card;
    });
    grid.replaceChildren.apply(grid, cards);
  }

  function select(entry) {
    state.selected = entry;
    state.lastRender = null;
    $("validate").disabled = true;
    renderCards();
    renderDetails();
    $("details").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // -- details panel ---------------------------------------------------------

  function renderDetails() {
    var entry = state.selected;
    $("empty-details").hidden = true;
    $("detail-body").hidden = false;

    var title = entry.asset_id;
    if (entry.revision) title += " (r" + entry.revision + ")";
    $("detail-title").textContent = title;

    var decision = state.decisions[entry.asset_id] || "pending";
    var badge = $("detail-decision");
    badge.className = "decision " + decision;
    badge.textContent = decision;

    var meta = $("detail-meta");
    var pairs = [
      ["sha256", entry.sha256 || "—"],
      ["quality", entry.quality_score != null ? Number(entry.quality_score).toFixed(2) : "—"],
      ["reason", entry.quality_reason || "—"],
      ["cluster", entry.cluster_id || "—"],
      ["created", entry.created_at || "—"],
    ];
    meta.replaceChildren.apply(
      meta,
      pairs.map(function (p) {
        var div = el("div");
        div.append(el("dt", null, p[0]), el("dd", null, p[1]));
        return div;
      })
    );

    resetOutputs();
  }

  function resetOutputs() {
    $("analysis-summary").textContent = "Not analyzed yet.";
    $("analysis-summary").className = "muted";
    $("analysis-body").replaceChildren();
    $("proposals-summary").textContent = "No proposals yet.";
    $("proposals-summary").className = "muted";
    $("proposals-list").replaceChildren();
    $("render-summary").textContent = "Nothing rendered.";
    $("render-summary").className = "muted";
    $("render-body").replaceChildren();
    $("validate-summary").textContent = "Nothing validated.";
    $("validate-summary").className = "muted";
    $("validate-body").replaceChildren();
  }

  function requireSelection() {
    if (!state.selected) {
      throw new Error("Select a catalog entry first.");
    }
    return state.selected;
  }

  function assertOk(data) {
    if (!data || data.status === "error") {
      throw new Error("Bad response from server.");
    }
  }

  // -- actions ---------------------------------------------------------------

  async function runAction(name, fn, onDone) {
    var label = name.charAt(0).toUpperCase() + name.slice(1);
    try {
      setStatus("Running " + label + "…");
      var data = await fn();
      onDone(data);
      setStatus(label + " complete.");
      showToast(label + " complete.");
    } catch (err) {
      setStatus(label + " failed.", true);
      showToast(err.message, true);
    }
  }

  function bodyFor(extra) {
    return JSON.stringify(Object.assign({ asset: state.selected.sha256 }, extra));
  }

  function post(url, body) {
    return fetchJSON(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: body,
    });
  }

  async function analyze() {
    requireSelection();
    var profile = $("profile").value;
    return runAction(
      "analyze",
      function () {
        return post("/api/analyze", bodyFor({ profile: profile }));
      },
      renderAnalysis
    );
  }

  function fmtMetric(label, value) {
    var div = el("dl", "metric");
    div.append(el("dt", null, label + ":"), el("dd", null, String(value)));
    return div;
  }

  function renderAnalysis(data) {
    assertOk(data);
    var quality = data.quality || {};
    var summary = $("analysis-summary");
    summary.className = "";
    summary.textContent =
      "Profile " + (data.metadata && data.metadata.profile) + " · asset " +
      String(data.asset_id).slice(0, 12) + "…";

    var body = $("analysis-body");
    var metrics = [
      ["technical", round(quality.technical_quality)],
      ["aesthetic", round(quality.aesthetic_quality)],
      ["sharpness", round(quality.sharpness)],
      ["exposure", round(quality.exposure)],
      ["contrast", round(quality.contrast)],
      [
        "resolution ok",
        quality.resolution_sufficient ? "yes" : "no",
      ],
    ];
    body.replaceChildren.apply(
      body,
      metrics.map(function (m) {
        return fmtMetric(m[0], m[1]);
      })
    );
  }

  async function propose() {
    requireSelection();
    var target = $("target").value;
    return runAction(
      "propose",
      function () {
        return post("/api/propose", bodyFor({ target: target }));
      },
      renderProposals
    );
  }

  function round(n) {
    return n === undefined || n === null ? "—" : Number(n).toFixed(3);
  }

  function renderProposals(data) {
    assertOk(data);
    var proposals = Array.isArray(data) ? data : [];
    var summary = $("proposals-summary");
    if (!proposals.length) {
      summary.className = "muted";
      summary.textContent = "No treatments proposed.";
      return;
    }
    summary.className = "";
    summary.textContent = proposals.length + " treatment(s) proposed.";

    var list = $("proposals-list");
    var items = proposals.map(function (p) {
      var li = el("li");
      var head = el("div");
      var treatment = el("span", "treatment", p.treatment);
      var scoreSpan = el(
        "span",
        "score",
        " — score " + round(p.score) + (p.evidence ? " · " + String(p.evidence).slice(0, 60) : "")
      );
      head.append(treatment, scoreSpan);
      li.append(head);

      if (Array.isArray(p.rationale) && p.rationale.length) {
        var ul = el("ul");
        ul.append.apply(
          ul,
          p.rationale.map(function (r) {
            return el("li", null, String(r));
          })
        );
        li.append(ul);
      }
      return li;
    });
    list.replaceChildren.apply(list, items);
  }

  async function renderTo(targetName) {
    requireSelection();
    return runAction(
      "render " + targetName,
      function () {
        return post("/api/render", bodyFor({ target: targetName }));
      },
      function (data) {
        renderRenderResult(data, targetName);
      }
    );
  }

  function renderRenderResult(data, targetName) {
    assertOk(data);
    state.lastRender = { sha: data.sha256, target: targetName, size: data.size_bytes };
    $("validate").disabled = false;

    var summary = $("render-summary");
    summary.className = "";
    summary.textContent =
      "Rendered " + data.treatment + " → " + data.target_width + "×" + data.target_height +
      (data.upscaled_warning ? " (UPSCALED — review quality)" : "");

    var body = $("render-body");
    var rows = [
      ["target", data.target_width + "×" + data.target_height],
      ["treatment", data.treatment],
      ["sha256", data.sha256],
      ["size", humanBytes(data.size_bytes)],
      ["renderer", data.renderer_version],
    ];
    if (data.notes && data.notes.length) {
      rows.push(["notes", data.notes.join("; ")]);
    }
    var grid = el("div", "render-result");
    grid.append.apply(
      grid,
      rows.map(function (r) {
        var div = el("div");
        div.append(el("span", "muted", r[0] + ": "), el("span", null, r[1]));
        return div;
      })
    );
    body.replaceChildren(grid);
  }

  async function validate() {
    requireSelection();
    if (!state.lastRender) {
      setStatus("Render an artifact first.", true);
      return;
    }
    var r = state.lastRender;
    return runAction(
      "validate",
      function () {
        return post("/api/validate", JSON.stringify({
          artifact_sha: r.sha,
          expected_sha: r.sha,
          target: r.target,
        }));
      },
      renderValidation
    );
  }

  function renderValidation(data) {
    assertOk(data);
    var summary = $("validate-summary");
    summary.className = "";
    summary.textContent =
      (data.publishable ? "Publishable." : "NOT publishable.") +
      " valid=" + (data.valid ? "yes" : "no");

    var body = $("validate-body");
    var checks = (data.checks || []).map(function (check) {
      var div = el("div", "check " + (check.passed ? "pass" : "fail"));
      var mark = el("span", "mark", check.passed ? "✓" : "✗");
      var name = el("span", "name", check.name);
      div.append(mark, name);
      if (check.reason) {
        var reason = el("span", "muted", " — " + check.reason);
        div.append(reason);
      }
      return div;
    });
    var report = el("div", "validate-report");
    report.append.apply(report, checks);
    body.replaceChildren(report);
  }

  function reviewAction(action, assetId) {
    return function () {
      if (!assetId) {
        setStatus("Select a catalog entry first.", true);
        return;
      }
      return runAction(
        action,
        function () {
          return post("/api/review/" + action, JSON.stringify({ asset: assetId }));
        },
        function (data) {
          state.decisions[assetId] = data.decision || "pending";
          renderCards();
          renderDetails();
          renderReview();
        }
      );
    };
  }

  function humanBytes(n) {
    n = Number(n) || 0;
    if (n < 1024) return n + " B";
    if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
    return (n / 1048576).toFixed(1) + " MB";
  }

  // -- review queue (M004/S03 T2) ------------------------------------------

  function reviewRow(item) {
    var decision = item.decision || "pending";
    var row = el("li", "review-row");
    row.setAttribute("role", "listitem");

    row.append(el("span", "review-name", item.asset_id));

    var badge = el("span", "decision " + decision, decision);
    row.append(badge);

    var actions = el("div", "review-actions");
    actions.setAttribute("role", "group");
    actions.setAttribute("aria-label", "Review actions for " + item.asset_id);

    var approve = el("button", "btn approve small", "Approve");
    approve.type = "button";
    approve.setAttribute(
      "aria-label",
      "Approve " + item.asset_id
    );
    approve.addEventListener("click", reviewAction("approve", item.asset_id));

    var reject = el("button", "btn reject small", "Reject");
    reject.type = "button";
    reject.setAttribute("aria-label", "Reject " + item.asset_id);
    reject.addEventListener("click", reviewAction("reject", item.asset_id));

    var undo = el("button", "btn small", "Undo");
    undo.type = "button";
    undo.setAttribute(
      "aria-label",
      "Undo decision for " + item.asset_id
    );
    undo.addEventListener("click", reviewAction("undo", item.asset_id));

    actions.append(approve, reject, undo);
    row.append(actions);

    if (Array.isArray(item.history) && item.history.length) {
      var history = el(
        "span",
        "review-history",
        "history: " + item.history.join(" · ")
      );
      row.append(history);
    }
    return row;
  }

  function renderReview() {
    var list = $("review-list");
    var filter = state.reviewFilter;
    var rows = state.reviewEntries.filter(function (item) {
      var d = item.decision || "pending";
      if (filter === "all") return true;
      return d === filter;
    });
    var status = $("review-status");
    status.textContent = rows.length + " entr" + (rows.length === 1 ? "y" : "ies") +
      " (" + (filter === "all" ? "all decisions" : filter) + ").";

    if (!rows.length) {
      var empty = el("li", "empty", "No entries with this decision.");
      empty.setAttribute("role", "listitem");
      list.replaceChildren(empty);
      return;
    }
    var items = rows.map(reviewRow);
    list.replaceChildren.apply(list, items);
  }

  function setFilter(filter) {
    state.reviewFilter = filter;
    ["pending", "approved", "rejected", "all"].forEach(function (key) {
      var btn = $("filter-" + key);
      if (btn) btn.setAttribute("aria-pressed", key === filter ? "true" : "false");
    });
    renderReview();
  }

  // -- taste dialogue --------------------------------------------------------

  function readAsBase64(file) {
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onload = function () {
        // strip the "data:<mime>;base64," prefix the API does not want
        var result = String(reader.result || "");
        resolve(result.slice(result.indexOf(",") + 1));
      };
      reader.onerror = function () {
        reject(new Error("could not read " + file.name));
      };
      reader.readAsDataURL(file);
    });
  }

  async function submitReaction() {
    var files = Array.prototype.slice.call($("taste-files").files || []);
    var note = $("taste-note").value.trim();
    if (!files.length) {
      showToast("Pick at least one image to react to.", true);
      return;
    }
    if (!note) {
      showToast("Say something about it — your words are the point.", true);
      return;
    }
    var button = $("taste-submit");
    button.disabled = true;
    $("taste-room-status").textContent = "Recording your reaction…";
    try {
      var images = await Promise.all(files.map(readAsBase64));
      var turn = await fetchJSON("/api/taste/drop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images: images, note: note, save: $("taste-save").checked }),
      });
      renderTurn(turn);
      $("taste-note").value = "";
      await loadTasteProfile();
    } catch (err) {
      $("taste-room-status").textContent = err.message;
      $("taste-room-status").classList.add("error");
      showToast(err.message, true);
    } finally {
      button.disabled = false;
    }
  }

  function renderTurn(turn) {
    $("taste-room-status").classList.remove("error");
    $("taste-room-status").textContent = "Recorded — kept in your own words.";
    $("taste-turn").hidden = false;
    // At most one probing question per turn; the room never lectures.
    $("taste-question").textContent = turn.question ? turn.question.text : "";
    // No silent learning: every reaction reports what it added.
    $("taste-learned-body").textContent = (turn.learned && turn.learned.summary) || "";
  }

  function claimItem(claim) {
    var item = el("li", "taste-claim");
    item.setAttribute("data-claim-id", claim.id);
    item.appendChild(el("p", "taste-claim-text", claim.text));

    var meta = el("p", "taste-claim-meta");
    meta.textContent =
      claim.status + " · " + claim.provenance + "-provenance · " +
      claim.evidence.length + " piece" + (claim.evidence.length === 1 ? "" : "s") +
      " of evidence";
    item.appendChild(meta);

    // Every claim opens its evidence: the images and the words behind it.
    var evidence = el("ul", "taste-evidence");
    claim.evidence.forEach(function (ref) {
      var line = el("li");
      line.appendChild(el("code", "taste-evidence-sha", ref.image_sha.slice(0, 12)));
      line.appendChild(el("q", "taste-evidence-quote", ref.verbatim));
      line.appendChild(
        el("span", "muted", " (confidence " + Number(ref.confidence).toFixed(2) + ")")
      );
      evidence.appendChild(line);
    });
    item.appendChild(evidence);

    var actions = el("div", "taste-claim-actions");
    [
      { kind: "pin", label: "Pin" },
      { kind: "edit", label: "Edit" },
      { kind: "dispute", label: "Dispute" },
    ].forEach(function (action) {
      var btn = el("button", "btn", action.label);
      btn.type = "button";
      btn.setAttribute("data-action", action.kind);
      btn.setAttribute("aria-label", action.label + " claim: " + claim.text);
      btn.addEventListener("click", function () {
        claimAction(action.kind, claim);
      });
      actions.appendChild(btn);
    });
    item.appendChild(actions);
    return item;
  }

  async function claimAction(kind, claim) {
    var body = { claim_id: claim.id };
    if (kind === "edit") {
      var text = window.prompt("Say it in your own words:", claim.text);
      if (!text) return;
      body.text = text;
    }
    try {
      await fetchJSON("/api/taste/" + kind, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      showToast(
        kind === "dispute"
          ? "Disputed — the claim is gone and its evidence is marked for another look."
          : "Recorded."
      );
      await loadTasteProfile();
    } catch (err) {
      showToast(err.message, true);
    }
  }

  function renderTasteProfile(profile) {
    var vocabulary = $("taste-vocabulary");
    vocabulary.textContent = "";
    var words = Object.keys(profile.vocabulary || {});
    if (!words.length) {
      vocabulary.appendChild(
        el("li", "muted", "Nothing yet — react to some images above.")
      );
    }
    words.forEach(function (word) {
      var entry = profile.vocabulary[word];
      vocabulary.appendChild(
        el(
          "li",
          "taste-word",
          word + " → " + entry.attribute + " (" + entry.usage_count + " uses)"
        )
      );
    });

    [
      { id: "taste-patterns", claims: profile.patterns, empty: "No patterns yet." },
      { id: "taste-tensions", claims: profile.tensions, empty: "No tensions surfaced." },
    ].forEach(function (section) {
      var list = $(section.id);
      list.textContent = "";
      var claims = section.claims || [];
      if (!claims.length) {
        list.appendChild(el("li", "muted", section.empty));
        return;
      }
      claims.forEach(function (claim) {
        list.appendChild(claimItem(claim));
      });
    });

    var evolution = $("taste-evolution");
    evolution.textContent = "";
    (profile.evolution || []).forEach(function (entry) {
      evolution.appendChild(
        el("li", "taste-evolution-entry", (entry.at ? entry.at + ": " : "") + entry.summary)
      );
    });

    var count = (profile.patterns || []).length + (profile.tensions || []).length;
    $("taste-profile-status").textContent =
      "Version " + profile.version + " · " + count + " claim" + (count === 1 ? "" : "s") +
      " · " + words.length + " word" + (words.length === 1 ? "" : "s");
  }

  async function loadTasteProfile() {
    try {
      state.tasteProfile = await fetchJSON("/api/taste/profile");
      renderTasteProfile(state.tasteProfile);
    } catch (err) {
      $("taste-profile-status").textContent = "Could not load the profile.";
      showToast(err.message, true);
    }
  }

  // -- taste deck (M009/S01) --------------------------------------------------

  function candidateLabel(cand) {
    return "entry " + cand.entry_id + " · " + cand.sha256.slice(0, 12);
  }

  function renderTasteDeck(pair) {
    var empty = $("taste-deck-empty");
    var body = $("taste-deck-body");
    if (!pair.available) {
      empty.hidden = false;
      body.hidden = true;
      return;
    }
    empty.hidden = true;
    body.hidden = false;
    $("taste-deck-a").textContent = candidateLabel(pair.a);
    $("taste-deck-b").textContent = candidateLabel(pair.b);
  }

  async function loadTastePair() {
    try {
      var pair = await fetchJSON("/api/taste/pair");
      currentTastePair = pair.available ? pair : null;
      renderTasteDeck(pair);
    } catch (err) {
      currentTastePair = null;
      $("taste-deck-status").textContent = "Could not load the pair.";
      showToast(err.message, true);
    }
  }

  async function submitVote(prefer) {
    if (!currentTastePair) {
      await loadTastePair();
      return;
    }
    var pair = currentTastePair;
    try {
      var result = await fetchJSON("/api/taste/vote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prefer: prefer,
          note: "",
          a_entry_id: pair.a.entry_id,
          b_entry_id: pair.b.entry_id,
        }),
      });
      $("taste-deck-status").classList.remove("error");
      $("taste-deck-status").textContent =
        "Recorded — profile now version " + result.profile_version + ".";
      await loadTastePair();
    } catch (err) {
      if (err.message.indexOf("(409)") !== -1) {
        $("taste-deck-status").textContent = "The pair changed — showing a new one.";
        await loadTastePair();
        return;
      }
      showToast(err.message, true);
    }
  }

  // -- view switching --------------------------------------------------------

  function showView(view) {
    $("catalog-view").hidden = view !== "catalog";
    $("review-view").hidden = view !== "review";
    $("taste-view").hidden = view !== "taste";
    var navs = { catalog: $("nav-catalog"), review: $("nav-review"), taste: $("nav-taste") };
    Object.keys(navs).forEach(function (key) {
      if (key === view) {
        navs[key].setAttribute("aria-current", "page");
      } else {
        navs[key].removeAttribute("aria-current");
      }
    });
    if (view === "review") renderReview();
    if (view === "taste") {
      loadTasteProfile();
      loadTastePair();
    }
  }

  // -- boot ------------------------------------------------------------------

  async function loadReview() {
    var review = await fetchJSON("/api/review");
    state.reviewEntries = review;
    state.decisions = {};
    review.forEach(function (r) {
      state.decisions[r.asset_id] = r.decision || "pending";
    });
  }

  async function loadCatalog() {
    setStatus("Loading catalog…");
    try {
      var entries = await fetchJSON("/catalog");
      await loadReview();
      state.entries = entries;
      state.reviewEntries = state.reviewEntries || [];
      renderCards();
      renderReview();
      setStatus(entries.length + " catalog entr" + (entries.length === 1 ? "y" : "ies") + ".");
    } catch (err) {
      setStatus("Failed to load catalog.", true);
      showToast(err.message, true);
    }
  }

  $("analyze").addEventListener("click", analyze);
  $("propose").addEventListener("click", propose);
  $("render-1080p").addEventListener("click", function () {
    renderTo("1080p");
  });
  $("render-4k").addEventListener("click", function () {
    renderTo("4k");
  });
  $("validate").addEventListener("click", validate);
  $("approve").addEventListener("click", function () {
    return reviewAction("approve", state.selected ? state.selected.asset_id : null)();
  });
  $("reject").addEventListener("click", function () {
    return reviewAction("reject", state.selected ? state.selected.asset_id : null)();
  });
  $("undo").addEventListener("click", function () {
    return reviewAction("undo", state.selected ? state.selected.asset_id : null)();
  });

  $("nav-catalog").addEventListener("click", function (e) {
    e.preventDefault();
    showView("catalog");
  });
  $("nav-review").addEventListener("click", function (e) {
    e.preventDefault();
    showView("review");
  });
  $("nav-taste").addEventListener("click", function (e) {
    e.preventDefault();
    showView("taste");
  });
  $("taste-submit").addEventListener("click", submitReaction);
  $("taste-prefer-a").addEventListener("click", function () {
    submitVote("a");
  });
  $("taste-prefer-b").addEventListener("click", function () {
    submitVote("b");
  });
  ["pending", "approved", "rejected", "all"].forEach(function (key) {
    $("filter-" + key).addEventListener("click", function () {
      setFilter(key);
    });
  });

  loadCatalog();
})();
