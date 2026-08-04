function render(candidates) {
  const list = document.getElementById("candidates");
  const empty = document.createElement("li");
  empty.textContent = "No candidates to review.";
  if (!candidates.length) {
    list.replaceChildren(empty);
    return;
  }
  const items = candidates.map((c) => {
    const li = document.createElement("li");
    const decision = c.decision ?? "pending";

    const name = document.createElement("span");
    name.textContent = c.asset_id;

    const state = document.createElement("span");
    state.className = `decision ${decision}`;
    state.textContent = decision === "pending" ? "pending" : decision;

    li.append(state, " — ", name);
    return li;
  });
  list.replaceChildren(...items);
}

async function loadReview() {
  const message = document.getElementById("message");
  message.textContent = "";
  message.className = "";
  const status = document.getElementById("status").value;
  const url = status
    ? `/api/review?status=${encodeURIComponent(status)}`
    : "/api/review";
  try {
    const resp = await fetch(url, { headers: { Accept: "application/json" } });
    if (!resp.ok) {
      throw new Error(`review request failed (${resp.status})`);
    }
    render(await resp.json());
  } catch (err) {
    message.textContent = err.message;
    message.className = "error";
  }
}

document.getElementById("load").addEventListener("click", loadReview);
loadReview();
