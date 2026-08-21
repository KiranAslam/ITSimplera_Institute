const POLL_INTERVAL_MS = 2000;

async function refreshStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();

    document.getElementById("stat-people").textContent = data.unique_people_tracked;
    document.getElementById("stat-violations").textContent = data.total_violations_logged;

    const list = document.getElementById("violation-breakdown");
    list.innerHTML = "";
    const entries = Object.entries(data.violations_by_class || {});
    if (entries.length === 0) {
      list.innerHTML = "<li>No violations yet</li>";
    } else {
      entries
        .sort((a, b) => b[1] - a[1])
        .forEach(([cls, count]) => {
          const li = document.createElement("li");
          li.innerHTML = `<span>${cls}</span><span>${count}</span>`;
          list.appendChild(li);
        });
    }
  } catch (err) {
    console.error("stats fetch failed", err);
  }
}

async function refreshLogs() {
  try {
    const res = await fetch("/api/logs");
    const data = await res.json();

    const tbody = document.getElementById("log-body");
    tbody.innerHTML = "";
    data.forEach((entry) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${entry.timestamp}</td>
        <td>${entry.track_id}</td>
        <td>${entry.violation_class}</td>
        <td>${entry.confidence}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("logs fetch failed", err);
  }
}

function refreshAll() {
  refreshStats();
  refreshLogs();
}

refreshAll();
setInterval(refreshAll, POLL_INTERVAL_MS);
