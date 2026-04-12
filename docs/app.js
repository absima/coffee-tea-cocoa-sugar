function FormatValue(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }

  if (typeof value === "number") {
    if (Math.abs(value) >= 1000) {
      return value.toLocaleString();
    }
    return Number(value).toFixed(4).replace(/\.?0+$/, "");
  }

  return String(value);
}

function EscapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function RenderInlineMarkdown(text) {
  return EscapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>");
}

function RenderMarkdown(containerId, markdownText) {
  const container = document.getElementById(containerId);
  const source = (markdownText || "").trim();

  if (!source) {
    container.innerHTML = '<div class="empty-state">Note not available yet.</div>';
    return;
  }

  const lines = source.split(/\r?\n/);
  const html = [];
  let inList = false;
  let paragraph = [];

  const FlushParagraph = () => {
    if (paragraph.length > 0) {
      html.push(`<p>${RenderInlineMarkdown(paragraph.join(" "))}</p>`);
      paragraph = [];
    }
  };

  const CloseList = () => {
    if (inList) {
      html.push("</ul>");
      inList = false;
    }
  };

  lines.forEach((rawLine) => {
    const line = rawLine.trim();

    if (!line) {
      FlushParagraph();
      CloseList();
      return;
    }

    const headingMatch = line.match(/^(#{1,4})\s+(.*)$/);
    if (headingMatch) {
      FlushParagraph();
      CloseList();
      const level = headingMatch[1].length;
      html.push(`<h${level}>${RenderInlineMarkdown(headingMatch[2])}</h${level}>`);
      return;
    }

    const bulletMatch = line.match(/^[-*]\s+(.*)$/);
    if (bulletMatch) {
      FlushParagraph();
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${RenderInlineMarkdown(bulletMatch[1])}</li>`);
      return;
    }

    CloseList();
    paragraph.push(line);
  });

  FlushParagraph();
  CloseList();
  container.innerHTML = html.join("");
}

function RenderTable(containerId, columns, rows) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  if (!rows || rows.length === 0) {
    container.innerHTML = '<div class="empty-state">No rows available yet.</div>';
    return;
  }

  const template = document.getElementById("table-template");
  const table = template.content.firstElementChild.cloneNode(true);
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");

  const headerRow = document.createElement("tr");
  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = column;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((column) => {
      const td = document.createElement("td");
      td.textContent = FormatValue(row[column]);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  container.appendChild(table);
}

function RenderSteps(steps) {
  const grid = document.getElementById("steps-grid");
  grid.innerHTML = "";

  steps.forEach((step, index) => {
    const article = document.createElement("article");
    article.className = "step-card";
    article.innerHTML = `
      <span class="step-number">0${index + 1}</span>
      <h3>${step.title}</h3>
      <p>${step.description}</p>
    `;
    grid.appendChild(article);
  });
}

function RenderMetricCards(metricsSummary) {
  const container = document.getElementById("metrics-band");
  container.innerHTML = "";

  metricsSummary.forEach((metric) => {
    const article = document.createElement("article");
    article.className = "metric-card";
    article.innerHTML = `
      <h3>${metric.asset_name}</h3>
      <strong>RMSE ${FormatValue(metric.rmse)}</strong>
      <span>Directional accuracy ${FormatValue(metric.directional_accuracy * 100)}%</span>
    `;
    container.appendChild(article);
  });
}

function RenderPlots(plots) {
  const grid = document.getElementById("plot-grid");
  grid.innerHTML = "";

  plots.forEach((plot) => {
    const article = document.createElement("article");
    article.className = "plot-card";
    article.innerHTML = `
      <header>
        <h3>${plot.title}</h3>
      </header>
      <div class="plot-media">
        <img src="${plot.path}" alt="${plot.title}" />
      </div>
      <footer>${plot.caption}</footer>
    `;
    grid.appendChild(article);
  });
}

function RenderNotes(notesByAsset, metricsSummary) {
  const tabs = document.getElementById("asset-tabs");
  const noteTitle = document.getElementById("asset-note-title");
  tabs.innerHTML = "";

  const assetNameLookup = {};
  (metricsSummary || []).forEach((metric) => {
    assetNameLookup[metric.asset_key] = metric.asset_name;
  });

  const preferredOrder = ["coffee", "tea", "cocoa", "sugar"];
  const assetKeys = Object.keys(notesByAsset).sort((left, right) => {
    const leftIndex = preferredOrder.indexOf(left);
    const rightIndex = preferredOrder.indexOf(right);
    const safeLeft = leftIndex === -1 ? Number.MAX_SAFE_INTEGER : leftIndex;
    const safeRight = rightIndex === -1 ? Number.MAX_SAFE_INTEGER : rightIndex;
    if (safeLeft !== safeRight) {
      return safeLeft - safeRight;
    }
    return left.localeCompare(right);
  });
  if (assetKeys.length === 0) {
    noteTitle.textContent = "Per-asset note";
    RenderMarkdown("asset-note", "Run the pipeline to generate per-asset notes.");
    return;
  }

  const SetActive = (assetKey) => {
    noteTitle.textContent = `${assetNameLookup[assetKey] || assetKey} note`;
    RenderMarkdown("asset-note", notesByAsset[assetKey] || "Note not available.");
    tabs.querySelectorAll(".tab-button").forEach((button) => {
      button.classList.toggle("active", button.dataset.assetKey === assetKey);
    });
  };

  assetKeys.forEach((assetKey, index) => {
    const button = document.createElement("button");
    button.className = "tab-button";
    button.dataset.assetKey = assetKey;
    button.textContent = assetNameLookup[assetKey] || assetKey;
    button.addEventListener("click", () => SetActive(assetKey));
    tabs.appendChild(button);

    if (index === 0) {
      SetActive(assetKey);
    }
  });
}

async function LoadSiteData() {
  const response = await fetch("./data/site_data.json");
  if (!response.ok) {
    throw new Error("Unable to load `web/data/site_data.json`. Run the pipeline first.");
  }
  return response.json();
}

async function Init() {
  try {
    const data = await LoadSiteData();

    document.getElementById("generated-at").textContent = data.generated_at;
    document.getElementById("asset-count").textContent = `${data.metrics_summary.length} commodities`;
    document.getElementById("raw-title").textContent = data.raw_preview.title;
    document.getElementById("feature-title").textContent = data.feature_preview.title;
    RenderMarkdown("cross-asset-note", data.cross_asset_note || "Cross-asset note not available yet.");

    RenderSteps(data.steps);
    RenderMetricCards(data.metrics_summary);
    RenderTable("raw-table", data.raw_preview.columns, data.raw_preview.rows);
    RenderTable("feature-table", data.feature_preview.columns, data.feature_preview.rows);

    const comparisonColumns = data.cross_asset_table.length > 0
      ? Object.keys(data.cross_asset_table[0])
      : [];
    RenderTable("comparison-table", comparisonColumns, data.cross_asset_table);
    RenderPlots(data.plots);
    RenderNotes(data.notes_by_asset, data.metrics_summary);
  } catch (error) {
    document.querySelector(".page-shell").innerHTML = `
      <section class="stage-panel">
        <div class="section-heading">
          <p class="eyebrow">Web View</p>
          <h2>Artifacts not ready yet</h2>
        </div>
        <div class="empty-state">${error.message}</div>
      </section>
    `;
  }
}

Init();
