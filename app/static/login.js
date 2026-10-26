document.getElementById("f").addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = document.getElementById("msg");
  msg.textContent = "";
  const r = await fetch("/api/login", { method: "POST", body: new FormData(e.target) });
  if (r.ok) { location.href = "/"; return; }
  const j = await r.json().catch(() => ({}));
  msg.textContent = j.detail || "Sign-in failed.";
});
