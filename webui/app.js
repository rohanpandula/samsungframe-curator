(function () {
  "use strict";

  // One Wall (M011/S02): one page, one flow — Load → Score and pick → Hang.
  // GET /api/wall is the single source of truth for the page; every action
  // that changes state ends by calling refresh().

  var state = {
    wall: null,          // last GET /api/wall payload
    filter: "pending",   // grid filter: pending | approved | rejected | all
    selected: null,      // the entry open in the drawer
    lastRender: null,    // {sha, target, size} for the Validate tool
    tasteProfile: null,
    watching: {},        // job name -> interval id
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

  function post(url, body) {
    return fetchJSON(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: body,
    });
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function plural(n, one, many) {
    return n + " " + (n === 1 ? one : many);
  }

  function decisionWord(decision) {
    return decision === "approved" ? "approved" : decision === "rejected" ? "rejected" : "pending";
  }

  function thumbUrl(entry, w) {
    return entry.thumb + "?w=" + (w || 320);
  }

  function scoreText(entry) {
    if (!entry.scored || entry.score === null) return "not scored yet";
    return "score " + Number(entry.score).toFixed(2);
  }

  // -- the wall --------------------------------------------------------------

  async function refresh() {
    try {
      state.wall = await fetchJSON("/api/wall");
    } catch (err) {
      setStatus("Could not load your photos.", true);
      showToast(err.message, true);
      return;
    }
    renderAll();
  }

  function renderAll() {
    var wall = state.wall;
    renderCounts(wall.counts);
    renderFolders(wall.folders);
    renderGrid();
    renderPicks();
    renderDestinations(wall.destinations);
    renderHangSummary(wall.counts);
    resumeJobs(wall.jobs);
    if (state.selected) {
      var fresh = findEntry(state.selected.entry_id);
      if (fresh) {
        state.selected = fresh;
        renderDrawer(fresh, false);
      }
    }
    var c = wall.counts;
    setStatus(
      c.loaded === 0
        ? "No photos yet — load a folder to begin."
        : plural(c.loaded, "photo", "photos") + " · " + c.scored + " scored · " +
          c.approved + " approved · " + c.hung + " on the Frame."
    );
  }

  function findEntry(entryId) {
    var entries = state.wall ? state.wall.entries : [];
    for (var i = 0; i < entries.length; i++) {
      if (entries[i].entry_id === entryId) return entries[i];
    }
    return null;
  }

  function renderCounts(counts) {
    $("count-loaded").textContent = counts.loaded;
    $("count-scored").textContent = counts.scored;
    $("count-approved").textContent = counts.approved;
    $("count-hung").textContent = counts.hung;
  }

  function renderFolders(folders) {
    var node = $("load-folders");
    if (!folders.length) {
      node.textContent = "Nothing loaded yet.";
      return;
    }
    node.textContent = "Loaded so far: " + folders.join(", ");
    if (!$("load-path").value) $("load-path").placeholder = folders[0];
  }

  // -- stage 2: the grid -----------------------------------------------------

  function visibleEntries() {
    var entries = state.wall ? state.wall.entries : [];
    if (state.filter === "all") return entries;
    return entries.filter(function (entry) {
      return entry.decision === state.filter;
    });
  }

  function renderGrid() {
    var grid = $("catalog-grid");
    var entries = visibleEntries();
    if (!state.wall || !state.wall.entries.length) {
      grid.replaceChildren(el("p", "empty", "No photos loaded yet."));
      return;
    }
    if (!entries.length) {
      grid.replaceChildren(el("p", "empty", "No " + decisionWord(state.filter) + " photos."));
      return;
    }
    grid.replaceChildren.apply(grid, entries.map(card));
  }

  function card(entry) {
    var article = el("article", "card " + entry.decision);
    article.dataset.entry = entry.entry_id;

    var open = el("button", "card-open");
    open.type = "button";
    open.setAttribute("aria-label", "Open " + entry.name);
    open.setAttribute(
      "aria-pressed",
      state.selected && state.selected.entry_id === entry.entry_id ? "true" : "false"
    );
    var img = el("img");
    img.src = thumbUrl(entry, 480);
    img.alt = "";
    img.loading = "lazy";
    img.decoding = "async";
    open.append(img);
    open.append(el("span", "score-badge" + (entry.scored ? "" : " unscored"), scoreText(entry)));
    if (entry.hung && entry.hung.length) {
      open.append(el("span", "hung-badge", "on the Frame"));
    }
    open.addEventListener("click", function () {
      select(entry);
    });

    var foot = el("div", "card-foot");
    foot.append(el("span", "card-name", entry.name));
    var actions = el("div", "card-actions");
    actions.setAttribute("role", "group");
    actions.setAttribute("aria-label", "Decide on " + entry.name);
    var keep = el("button", "btn approve" + (entry.decision === "approved" ? " on" : ""), "✓");
    keep.type = "button";
    keep.setAttribute("aria-label", "Approve " + entry.name);
    keep.setAttribute("aria-pressed", entry.decision === "approved" ? "true" : "false");
    keep.addEventListener("click", function () {
      decide(entry.decision === "approved" ? "undo" : "approve", entry);
    });
    var pass = el("button", "btn reject" + (entry.decision === "rejected" ? " on" : ""), "✕");
    pass.type = "button";
    pass.setAttribute("aria-label", "Reject " + entry.name);
    pass.setAttribute("aria-pressed", entry.decision === "rejected" ? "true" : "false");
    pass.addEventListener("click", function () {
      decide(entry.decision === "rejected" ? "undo" : "reject", entry);
    });
    actions.append(keep, pass);
    foot.append(actions);

    article.append(open, foot);
    return article;
  }

  function setFilter(filter) {
    state.filter = filter;
    ["pending", "approved", "rejected", "all"].forEach(function (key) {
      var btn = $("filter-" + key);
      if (btn) btn.setAttribute("aria-pressed", key === filter ? "true" : "false");
    });
    renderGrid();
  }

  // -- decisions -------------------------------------------------------------

  async function decide(action, entry) {
    try {
      var data = await post(
        "/api/review/" + action,
        JSON.stringify({ asset: entry.asset_id, entry_id: entry.entry_id })
      );
      var word = decisionWord(data.decision || "pending");
      showToast(entry.name + " — " + word + ".");
      await refresh();
    } catch (err) {
      showToast(err.message, true);
    }
  }

  // -- your picks --------------------------------------------------------------

  function renderPicks() {
    var list = $("review-list");
    var kept = (state.wall ? state.wall.entries : []).filter(function (entry) {
      return entry.decision === "approved";
    });
    $("review-status").textContent = kept.length
      ? plural(kept.length, "photo", "photos") + " approved, in score order."
      : "Nothing approved yet. Tap ✓ on a photo above.";
    if (!kept.length) {
      list.replaceChildren();
      return;
    }
    list.replaceChildren.apply(
      list,
      kept.map(function (entry) {
        var row = el("li", "review-row");
        var open = el("button", "pick-open");
        open.type = "button";
        open.setAttribute("aria-label", "Open " + entry.name);
        var img = el("img");
        img.src = thumbUrl(entry, 240);
        img.alt = "";
        img.loading = "lazy";
        open.append(img);
        open.addEventListener("click", function () {
          select(entry);
        });
        var name = el("span", "review-name", entry.name);
        var undo = el("button", "btn small", "Undo");
        undo.type = "button";
        undo.setAttribute("aria-label", "Undo approving " + entry.name);
        undo.addEventListener("click", function () {
          decide("undo", entry);
        });
        row.append(open, name, undo);
        return row;
      })
    );
  }

  // -- the drawer --------------------------------------------------------------

  function select(entry) {
    state.selected = entry;
    state.lastRender = null;
    $("validate").disabled = true;
    renderDrawer(entry, true);
    renderGrid();
  }

  function closeDrawer() {
    state.selected = null;
    $("details").hidden = true;
    renderGrid();
  }

  function renderDrawer(entry, reset) {
    var drawer = $("details");
    drawer.hidden = false;
    $("detail-title").textContent = entry.name;
    var badge = $("detail-decision");
    badge.className = "decision " + entry.decision;
    badge.textContent = decisionWord(entry.decision);
    var img = $("detail-image");
    img.hidden = false;
    img.alt = entry.name;
    var src = thumbUrl(entry, 1200);
    if (img.getAttribute("src") !== src) img.src = src;
    $("detail-score").textContent = entry.scored
      ? "Score " + Number(entry.score).toFixed(2) + " · technical " +
        Number(entry.technical).toFixed(2) + (entry.hung.length ? " · on the Frame" : "")
      : "Not scored yet — run “Score all photos”.";

    var meta = $("detail-meta");
    var pairs = [
      ["file", entry.asset_id],
      ["sha256", entry.sha256],
      ["decision", entry.decision],
    ];
    entry.hung.forEach(function (h) {
      pairs.push(["hung", h.artifact_id + " · " + h.adapter_id + " · " + h.target]);
    });
    meta.replaceChildren.apply(
      meta,
      pairs.map(function (p) {
        var div = el("div");
        div.append(el("dt", null, p[0]), el("dd", null, p[1]));
        return div;
      })
    );
    if (reset) {
      resetOutputs();
      $("detail-layout").textContent = "Working out how it should hang…";
      suggestLayout(entry);
      drawer.scrollTop = 0;
    }
  }

  var layoutRequest = 0;
  async function suggestLayout(entry) {
    var ticket = ++layoutRequest;
    try {
      var proposals = await post(
        "/api/propose",
        JSON.stringify({ asset: entry.sha256, target: $("target").value })
      );
      if (ticket !== layoutRequest) return;
      if (!Array.isArray(proposals) || !proposals.length) {
        $("detail-layout").textContent = "No layout suggested for this photo.";
        return;
      }
      var top = proposals[0];
      var why = Array.isArray(top.rationale) && top.rationale.length ? top.rationale[0] : "";
      $("detail-layout").textContent =
        "Hangs as " + String(top.treatment).replace(/_/g, " ") + (why ? " — " + why : "");
      renderProposals(proposals);
    } catch (err) {
      if (ticket !== layoutRequest) return;
      $("detail-layout").textContent = "Could not suggest a layout: " + err.message;
    }
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
      throw new Error("Open a photo first.");
    }
    return state.selected;
  }

  function assertOk(data) {
    if (!data || data.status === "error") {
      throw new Error("Bad response from server.");
    }
  }

  // -- jobs (load / score / publish) ------------------------------------------

  function jobUI(name) {
    return {
      load: { progress: $("load-progress"), status: $("load-status"), button: $("load-button") },
      score: { progress: $("score-progress"), status: $("score-status"), button: $("score-button") },
      publish: { progress: $("hang-progress"), status: $("hang-status"), button: $("hang-button") },
    }[name];
  }

  function showJob(name, job) {
    var ui = jobUI(name);
    var running = job.state === "running";
    ui.button.disabled = running;
    ui.progress.hidden = !running;
    if (running) {
      if (job.total > 0) {
        ui.progress.max = job.total;
        ui.progress.value = job.done;
        ui.status.textContent =
          job.done + " of " + job.total + (job.current ? " · " + job.current : "");
      } else {
        ui.progress.removeAttribute("value");
        ui.status.textContent = job.current || "Working…";
      }
    }
    ui.status.classList.toggle("error", job.state === "error");
    if (job.state === "error") ui.status.textContent = job.message || "Failed.";
  }

  function watchJob(name, url, onDone) {
    if (state.watching[name]) return;
    state.watching[name] = setInterval(async function () {
      var job;
      try {
        job = await fetchJSON(url);
      } catch (err) {
        return; // transient; try again next tick
      }
      showJob(name, job);
      if (job.state !== "running") {
        clearInterval(state.watching[name]);
        delete state.watching[name];
        if (job.state === "done") onDone(job);
        else showToast((job.message || name + " failed"), true);
        await refresh();
      }
    }, 1000);
  }

  function resumeJobs(jobs) {
    Object.keys(jobs || {}).forEach(function (name) {
      var job = jobs[name];
      if (job.state === "running") {
        showJob(name, job);
        watchJob(name, jobEndpoint(name), jobDone(name));
      }
    });
  }

  function jobEndpoint(name) {
    return { load: "/api/load", score: "/api/score", publish: "/api/publish" }[name];
  }

  function jobDone(name) {
    return { load: loadDone, score: scoreDone, publish: hangDone }[name];
  }

  async function startJob(name, body) {
    var ui = jobUI(name);
    try {
      var job = await post(jobEndpoint(name), JSON.stringify(body));
      showJob(name, job);
      if (job.state === "running") watchJob(name, jobEndpoint(name), jobDone(name));
      else if (job.state === "done") { jobDone(name)(job); await refresh(); }
      else showToast(job.message || name + " failed", true);
    } catch (err) {
      ui.status.textContent = "";
      showToast(err.message, true);
    }
  }

  // stage 1
  function loadDone(job) {
    var r = job.result || {};
    // The ingest report's counts, in the words a person would use.
    var said = [
      [r.indexed_count, "added"],
      [r.unsupported_count, "unsupported (RAW or unreadable)"],
      [r.corrupt_count, "corrupt"],
      [r.error_count, "failed"],
    ].filter(function (pair) { return typeof pair[0] === "number" && pair[0] > 0; })
     .map(function (pair) { return pair[0] + " " + pair[1]; });
    if (!said.length && typeof r.total_enumerated === "number") {
      said.push(r.total_enumerated === 0 ? "no photos found there" : "nothing new — already loaded");
    }
    $("load-status").textContent = "Loaded " + (r.folder || "") + (said.length ? " — " + said.join(", ") : "") + ".";
    showToast("Folder loaded.");
  }

  // stage 2
  function scoreDone(job) {
    var r = job.result || {};
    $("score-status").textContent = r.total === 0
      ? "Every photo is already scored."
      : "Scored " + r.scored + " of " + r.total + (r.failed ? " · " + r.failed + " could not be read" : "") + ".";
    showToast(r.total === 0 ? "Nothing new to score." : "Scored " + r.scored + " photos.");
  }

  // stage 3
  function renderDestinations(destinations) {
    var select = $("destination");
    var current = select.value;
    select.replaceChildren.apply(
      select,
      destinations.map(function (d) {
        var option = el("option", null, d.label + (d.available ? "" : " — not available"));
        option.value = d.id;
        option.disabled = !d.available;
        return option;
      })
    );
    var ids = destinations.map(function (d) { return d.id; });
    select.value = ids.indexOf(current) !== -1 && current ? current : "folder";
    destinationChanged();
  }

  function destinationChanged() {
    var id = $("destination").value;
    var chosen = (state.wall ? state.wall.destinations : []).filter(function (d) {
      return d.id === id;
    })[0];
    var folderInput = $("destination-folder");
    folderInput.hidden = id !== "folder";
    var note = $("destination-note");
    if (!chosen) {
      note.textContent = "";
    } else if (!chosen.available) {
      note.textContent = chosen.reason;
    } else if (chosen.id === "folder") {
      note.textContent = "Files land in " + (folderInput.value || chosen.location) + ". Copy that folder to a USB stick and the Frame will show it.";
    } else {
      note.textContent = chosen.location === "in-memory"
        ? "A test target: it remembers what was hung until the server restarts."
        : chosen.location;
    }
  }

  function renderHangSummary(counts) {
    var button = $("hang-button");
    button.textContent = counts.approved
      ? "Hang " + plural(counts.approved, "approved photo", "approved photos")
      : "Hang approved photos";
    button.disabled = counts.approved === 0 || Boolean(state.watching.publish);
    $("hang-summary").textContent = counts.approved
      ? ""
      : "Approve at least one photo first.";
  }

  function hangDone(job) {
    var r = job.result || {};
    var status = $("hang-status");
    status.textContent =
      "Hung " + r.hung + " of " + r.approved +
      (r.skipped ? " · " + r.skipped + " already there" : "") +
      (r.failed ? " · " + r.failed + " skipped, see below" : "") +
      (r.location && r.location !== "in-memory" ? " · saved to " + r.location : "") + ".";
    var list = $("hang-results");
    list.replaceChildren.apply(
      list,
      (r.items || []).map(function (item) {
        var li = el("li", "hang-item " + item.status);
        var text;
        if (item.status === "hung") text = item.name + " — hung as " + String(item.treatment).replace(/_/g, " ");
        else if (item.status === "skipped") text = item.name + " — already there";
        else if (item.status === "unpublishable") text = item.name + " — not publishable: " + (item.reasons || []).join("; ");
        else text = item.name + " — " + (item.error || "failed");
        li.textContent = text;
        return li;
      })
    );
    showToast(r.hung ? "Hung " + r.hung + " photos." : "Nothing was hung.", !r.hung);
  }

  // -- more tools (the old bench, unchanged in behavior) ----------------------

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

  function round(n) {
    return n === undefined || n === null ? "—" : Number(n).toFixed(3);
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
      ["resolution ok", quality.resolution_sufficient ? "yes" : "no"],
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
    list.replaceChildren.apply(
      list,
      proposals.map(function (p) {
        var li = el("li");
        var head = el("div");
        head.append(
          el("span", "treatment", p.treatment),
          el("span", "score", " — score " + round(p.score))
        );
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
      })
    );
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

  function humanBytes(n) {
    n = Number(n) || 0;
    if (n < 1024) return n + " B";
    if (n < 1048576) return (n / 1024).toFixed(1) + " KB";
    return (n / 1048576).toFixed(1) + " MB";
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
    if (data.notes && data.notes.length) rows.push(["notes", data.notes.join("; ")]);
    var grid = el("div", "render-result");
    grid.append.apply(
      grid,
      rows.map(function (r) {
        var div = el("div");
        div.append(el("span", "muted", r[0] + ": "), el("span", null, r[1]));
        return div;
      })
    );
    var preview = el("img", "render-preview");
    preview.src = "/api/thumb/" + data.sha256 + "?w=960";
    preview.alt = "Rendered " + data.treatment;
    body.replaceChildren(grid, preview);
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
      div.append(el("span", "mark", check.passed ? "✓" : "✗"), el("span", "name", check.name));
      if (check.reason) div.append(el("span", "muted", " — " + check.reason));
      return div;
    });
    var report = el("div", "validate-report");
    report.append.apply(report, checks);
    body.replaceChildren(report);
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

  // -- taste deck (M009/S01; pictures instead of labels since M011/S02) -------

  function candidateFigure(container, cand, letter) {
    container.replaceChildren();
    var img = el("img");
    img.src = "/api/thumb/" + cand.sha256 + "?w=480";
    img.alt = "Photo " + letter;
    container.append(img, el("span", "pair-letter", letter));
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
    candidateFigure($("taste-deck-a"), pair.a, "A");
    candidateFigure($("taste-deck-b"), pair.b, "B");
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
    // IN-03: disable both buttons for the duration of the request, mirroring
    // submitReaction's existing pattern — a rapid double-click otherwise fires
    // two POST /api/taste/vote requests against the same pair before the first
    // response updates it (harmless — the server-side TOCTOU check 409s the
    // second one — but an avoidable, user-visible "pair changed" flash).
    var buttonA = $("taste-prefer-a");
    var buttonB = $("taste-prefer-b");
    buttonA.disabled = true;
    buttonB.disabled = true;
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
    } finally {
      buttonA.disabled = false;
      buttonB.disabled = false;
    }
  }

  // -- wiring + boot ---------------------------------------------------------

  $("load-form").addEventListener("submit", function (e) {
    e.preventDefault();
    var path = $("load-path").value.trim();
    if (!path) {
      showToast("Type the folder to load first.", true);
      $("load-path").focus();
      return;
    }
    startJob("load", { path: path });
  });
  $("score-button").addEventListener("click", function () {
    startJob("score", {});
  });
  $("hang-button").addEventListener("click", function () {
    var body = { destination: $("destination").value, output: $("target").value };
    var folder = $("destination-folder").value.trim();
    if (body.destination === "folder" && folder) body.folder = folder;
    $("hang-results").replaceChildren();
    startJob("publish", body);
  });
  $("destination").addEventListener("change", destinationChanged);
  $("destination-folder").addEventListener("input", destinationChanged);

  ["pending", "approved", "rejected", "all"].forEach(function (key) {
    $("filter-" + key).addEventListener("click", function () {
      setFilter(key);
    });
  });

  $("detail-close").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && state.selected) closeDrawer();
  });
  $("approve").addEventListener("click", function () {
    if (state.selected) decide("approve", state.selected);
  });
  $("reject").addEventListener("click", function () {
    if (state.selected) decide("reject", state.selected);
  });
  $("undo").addEventListener("click", function () {
    if (state.selected) decide("undo", state.selected);
  });
  $("analyze").addEventListener("click", analyze);
  $("propose").addEventListener("click", propose);
  $("render-1080p").addEventListener("click", function () {
    renderTo("1080p");
  });
  $("render-4k").addEventListener("click", function () {
    renderTo("4k");
  });
  $("validate").addEventListener("click", validate);

  ["catalog", "review", "taste"].forEach(function (key) {
    $("nav-" + key).addEventListener("click", function () {
      ["catalog", "review", "taste"].forEach(function (other) {
        if (other === key) $("nav-" + other).setAttribute("aria-current", "page");
        else $("nav-" + other).removeAttribute("aria-current");
      });
    });
  });

  $("taste-submit").addEventListener("click", submitReaction);
  $("taste-prefer-a").addEventListener("click", function () {
    submitVote("a");
  });
  $("taste-prefer-b").addEventListener("click", function () {
    submitVote("b");
  });
  $("taste-deck-a").addEventListener("click", function () {
    submitVote("a");
  });
  $("taste-deck-b").addEventListener("click", function () {
    submitVote("b");
  });

  refresh();
  loadTasteProfile();
  loadTastePair();
})();
