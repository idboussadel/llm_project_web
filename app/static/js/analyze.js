/**
 * Sentiment Analysis Page Interactivity
 */

// DOM Elements
const textInput = document.getElementById("textInput");
const exampleSelector = document.getElementById("exampleSelector");
const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");
const charCount = document.getElementById("charCount");
const loadingState = document.getElementById("loadingState");
const errorState = document.getElementById("errorState");
const errorMessage = document.getElementById("errorMessage");
const emptyState = document.getElementById("emptyState");
const resultsDisplay = document.getElementById("resultsDisplay");

// State
let currentExample = null;

// Character counter
textInput.addEventListener("input", () => {
  const count = textInput.value.length;
  charCount.textContent = count;

  // Enable/disable analyze button
  analyzeBtn.disabled = count === 0;
});

// Example selector
exampleSelector.addEventListener("change", (e) => {
  const selectedOption = e.target.selectedOptions[0];

  if (selectedOption.value) {
    const text = selectedOption.dataset.text;
    const category = selectedOption.dataset.category;
    const source = selectedOption.dataset.source;

    textInput.value = text;
    textInput.dispatchEvent(new Event("input"));

    currentExample = {
      id: selectedOption.value,
      text,
      category,
      source,
    };
  }
});

// Clear button
clearBtn.addEventListener("click", () => {
  textInput.value = "";
  exampleSelector.value = "";
  charCount.textContent = "0";
  analyzeBtn.disabled = true;
  currentExample = null;

  // Reset results
  showEmptyState();
  hideError();
});

// Analyze button
analyzeBtn.addEventListener("click", async () => {
  const text = textInput.value.trim();

  if (!text) {
    showError("Please enter some text to analyze");
    return;
  }

  await analyzeSentiment(text);
});

// API: Analyze sentiment
async function analyzeSentiment(text) {
  try {
    // Show loading
    showLoading();
    hideError();

    // API call
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: text,
        return_probs: true,
      }),
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || "Analysis failed");
    }

    // Display results
    displayResults(data.result);
  } catch (error) {
    console.error("Analysis error:", error);
    showError(error.message || "Failed to analyze sentiment");
    showEmptyState();
  } finally {
    hideLoading();
  }
}

// Display results
function displayResults(result) {
  // Hide empty state, show results
  emptyState.classList.add("hidden");
  resultsDisplay.classList.remove("hidden");

  const { label, score, confidence, probabilities } = result;

  // Update sentiment badge
  const sentimentBadge = document.getElementById("sentimentBadge");
  sentimentBadge.textContent = label.charAt(0).toUpperCase() + label.slice(1);

  // Set badge color
  sentimentBadge.className = "px-4 py-2 rounded-md font-semibold text-lg";
  if (label === "positive") {
    sentimentBadge.classList.add("bg-positive", "text-white");
  } else if (label === "negative") {
    sentimentBadge.classList.add("bg-negative", "text-white");
  } else {
    sentimentBadge.classList.add("bg-neutral", "text-white");
  }

  // Update confidence
  const confidenceValue = document.getElementById("confidenceValue");
  confidenceValue.textContent = `${(confidence * 100).toFixed(1)}%`;

  // Update score
  const scoreValue = document.getElementById("scoreValue");
  scoreValue.textContent = score.toFixed(3); // Format to 3 decimal places

  // Update score bar
  const scoreBar = document.getElementById("scoreBar");
  const scorePercent = ((score + 1) / 2) * 100; // Map -1..1 to 0..100
  scoreBar.style.width = `${scorePercent}%`;

  if (score > 0) {
    scoreBar.className = "h-full bg-positive transition-all duration-500";
  } else if (score < 0) {
    scoreBar.className = "h-full bg-negative transition-all duration-500";
  } else {
    scoreBar.className = "h-full bg-neutral transition-all duration-500";
  }

  // Update probability bars
  if (probabilities) {
    updateProbabilityBar("positive", probabilities.positive || 0);
    updateProbabilityBar("neutral", probabilities.neutral || 0);
    updateProbabilityBar("negative", probabilities.negative || 0);
  }

  // Update interpretation
  updateInterpretation(label, confidence, score);
}

// Update probability bar
function updateProbabilityBar(sentiment, value) {
  const probElement = document.getElementById(
    `prob${sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}`,
  );
  const barElement = document.getElementById(
    `bar${sentiment.charAt(0).toUpperCase() + sentiment.slice(1)}`,
  );

  probElement.textContent = `${(value * 100).toFixed(1)}%`;
  barElement.style.width = `${value * 100}%`;
}

// Update interpretation text
function updateInterpretation(label, confidence, score) {
  const interpretationText = document.getElementById("interpretationText");

  let interpretation = "";

  if (confidence > 0.9) {
    interpretation = `The model is <strong>highly confident</strong> (${(confidence * 100).toFixed(1)}%) that this text expresses <strong>${label}</strong> sentiment. `;
  } else if (confidence > 0.7) {
    interpretation = `The model is <strong>confident</strong> (${(confidence * 100).toFixed(1)}%) that this text expresses <strong>${label}</strong> sentiment. `;
  } else {
    interpretation = `The model suggests <strong>${label}</strong> sentiment with <strong>moderate confidence</strong> (${(confidence * 100).toFixed(1)}%). `;
  }

  if (label === "positive") {
    interpretation +=
      "This indicates bullish market sentiment and potential upward price movement.";
  } else if (label === "negative") {
    interpretation +=
      "This indicates bearish market sentiment and potential downward price movement.";
  } else {
    interpretation +=
      "This indicates neutral market sentiment with unclear directional bias.";
  }

  interpretationText.innerHTML = interpretation;
}

// UI State Management
function showLoading() {
  loadingState.classList.remove("hidden");
  analyzeBtn.disabled = true;
}

function hideLoading() {
  loadingState.classList.add("hidden");
  analyzeBtn.disabled = false;
}

function showError(message) {
  errorMessage.textContent = message;
  errorState.classList.remove("hidden");
}

function hideError() {
  errorState.classList.add("hidden");
}

function showEmptyState() {
  emptyState.classList.remove("hidden");
  resultsDisplay.classList.add("hidden");
}

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  console.log("Sentiment Analysis page initialized");
  analyzeBtn.disabled = true;
});
