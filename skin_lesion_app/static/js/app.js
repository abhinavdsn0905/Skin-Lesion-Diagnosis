/* ============================================================
   DermaAI – Skin Lesion Classifier · app.js
   ============================================================ */

"use strict";

// ── Colour palette for model dots ──────────────────────────────────────────
const MODEL_COLORS = {
  "ResNet-50":       "#3B82F6",
  "EfficientNet-B0": "#10B981",
  "DINOv2":          "#8B5CF6",
};

// ── Risk labels & icons ───────────────────────────────────────────────────
const RISK_META = {
  high:   { label: "High Risk",   icon: "⚠️" },
  medium: { label: "Medium Risk", icon: "🔶" },
  low:    { label: "Low Risk",    icon: "✅" },
};

// ── DOM refs ──────────────────────────────────────────────────────────────
const dropzone        = document.getElementById("dropzone");
const fileInput       = document.getElementById("file-input");
const dropzoneIdle    = document.getElementById("dropzone-idle");
const dropzonePreview = document.getElementById("dropzone-preview");
const previewImg      = document.getElementById("preview-img");
const removeBtn       = document.getElementById("remove-btn");
const analyzeBtn      = document.getElementById("analyze-btn");
const btnText         = document.getElementById("btn-text");
const btnIcon         = document.getElementById("btn-icon");
const errorBox        = document.getElementById("error-box");
const spinner         = document.getElementById("spinner");
const resultsContent  = document.getElementById("results-content");
const resultsPlaceholder = document.getElementById("results-placeholder");
const modelResults    = document.getElementById("model-results");
const resultTemplate  = document.getElementById("result-template");

let currentImageBase64 = null;

// ============================================================
// IMAGE HANDLING
// ============================================================

function showPreview(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Please upload a valid image file (PNG, JPG, JPEG, BMP).");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    currentImageBase64 = e.target.result; // full data URL
    previewImg.src = currentImageBase64;
    dropzoneIdle.classList.add("hidden");
    dropzonePreview.classList.remove("hidden");
    analyzeBtn.disabled = false;
    hideError();
  };
  reader.readAsDataURL(file);
}

function clearImage() {
  currentImageBase64 = null;
  fileInput.value = "";
  previewImg.src = "";
  dropzonePreview.classList.add("hidden");
  dropzoneIdle.classList.remove("hidden");
  analyzeBtn.disabled = true;
  hideError();
  resetResults();
}

// File input change
fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) showPreview(e.target.files[0]);
});

// Click on dropzone → trigger file input
dropzone.addEventListener("click", (e) => {
  if (e.target === removeBtn || removeBtn.contains(e.target)) return;
  if (currentImageBase64) return; // already has image
  fileInput.click();
});

// Remove button
removeBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  clearImage();
});

// Drag & drop
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
["dragleave", "dragend"].forEach((ev) => {
  dropzone.addEventListener(ev, () => dropzone.classList.remove("dragover"));
});
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) showPreview(file);
});

// ============================================================
// HELPERS
// ============================================================

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
}
function hideError() { errorBox.classList.add("hidden"); }

function resetResults() {
  resultsContent.classList.add("hidden");
  resultsPlaceholder.classList.remove("hidden");
  modelResults.innerHTML = "";
  spinner.classList.add("hidden");
}

function setLoading(loading) {
  if (loading) {
    btnText.textContent = "Analyzing…";
    btnIcon.style.display = "none";
    analyzeBtn.classList.add("loading");
    analyzeBtn.disabled = true;
    spinner.classList.remove("hidden");
    resultsPlaceholder.classList.add("hidden");
    resultsContent.classList.add("hidden");
    modelResults.innerHTML = "";
  } else {
    btnText.textContent = "Analyze Image";
    btnIcon.style.display = "";
    analyzeBtn.classList.remove("loading");
    analyzeBtn.disabled = !currentImageBase64;
    spinner.classList.add("hidden");
  }
}

function getSelectedModels() {
  return [...document.querySelectorAll('input[name="models"]:checked')]
    .map((el) => el.value);
}

// ============================================================
// BUILD RESULT CARDS
// ============================================================

function buildResultBlock(result) {
  const tpl = resultTemplate.content.cloneNode(true);
  const block = tpl.querySelector(".model-result-block");

  // Model dot colour
  const dot = block.querySelector(".model-dot");
  dot.style.background = MODEL_COLORS[result.model] || "#64748B";

  // Model name
  block.querySelector(".model-name-text").textContent = result.model;

  // Note badge (demo mode)
  const noteBadge = block.querySelector(".model-note-badge");
  if (result.note) {
    noteBadge.textContent = result.note;
    noteBadge.classList.remove("hidden");
  }

  // Predicted pill
  block.querySelector(".pred-label").textContent = result.predicted_label.toUpperCase();
  block.querySelector(".pred-conf").textContent   = result.confidence;

  // Description
  block.querySelector(".pred-desc").textContent = result.predicted_description;

  // Risk banner
  const riskEl = block.querySelector(".risk-banner");
  const riskMeta = RISK_META[result.predicted_risk] || { label: "Unknown", icon: "❓" };
  riskEl.className = `risk-banner ${result.predicted_risk}`;
  riskEl.textContent = `${riskMeta.icon} ${riskMeta.label}`;

  // Plain Text Review Section
  const reviewContainer = block.querySelector(".model-text-review");
  if (reviewContainer && result.review) {
    reviewContainer.className = `model-text-review risk-${result.predicted_risk}`;
    
    const reviewTitle = reviewContainer.querySelector(".review-title");
    if (reviewTitle) reviewTitle.textContent = result.review.title || "Clinical Assessment";

    const reviewSummary = reviewContainer.querySelector(".review-summary");
    if (reviewSummary) reviewSummary.textContent = result.review.summary || "";

    const reviewAdvice = reviewContainer.querySelector(".review-advice");
    if (reviewAdvice) reviewAdvice.textContent = result.review.clinical_advice || "";

    const reviewWatch = reviewContainer.querySelector(".review-watch");
    if (reviewWatch) reviewWatch.textContent = result.review.watch_signs || "";
  }

  // Score bars
  const barsContainer = block.querySelector(".score-bars");
  result.scores.forEach((s, idx) => {
    const row = document.createElement("div");
    row.className = `score-row${idx === 0 ? " top-class" : ""}`;
    row.innerHTML = `
      <span class="score-row-label">${s.label}</span>
      <div class="score-track">
        <div class="score-fill" style="width: 0%"></div>
      </div>
      <span class="score-pct">${s.percent}</span>`;
    barsContainer.appendChild(row);

    // Animate bar width after a brief paint delay
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        row.querySelector(".score-fill").style.width = `${(s.score * 100).toFixed(2)}%`;
      });
    });
  });

  return block;
}

// ============================================================
// PREDICTION REQUEST
// ============================================================

analyzeBtn.addEventListener("click", async () => {
  hideError();

  if (!currentImageBase64) {
    showError("Please upload an image first.");
    return;
  }

  const selectedModels = getSelectedModels();
  if (!selectedModels.length) {
    showError("Please select at least one model.");
    return;
  }

  setLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: currentImageBase64, models: selectedModels }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || `Server error ${response.status}`);
    }

    const data = await response.json();

    if (!data.results || !data.results.length) {
      throw new Error("No results returned from the server.");
    }

    modelResults.innerHTML = "";
    data.results.forEach((r) => modelResults.appendChild(buildResultBlock(r)));

    resultsPlaceholder.classList.add("hidden");
    resultsContent.classList.remove("hidden");
    resultsContent.scrollIntoView({ behavior: "smooth", block: "start" });

  } catch (err) {
    showError(`Analysis failed: ${err.message}`);
    resetResults();
  } finally {
    setLoading(false);
  }
});
