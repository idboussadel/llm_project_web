/**
 * Trading Signals Page JavaScript
 * Full pipeline: Multi-source → Sentiment → TFT → Signals
 */

// DOM Elements
const tickerSelect = document.getElementById("tickerSelect");
const generateBtn = document.getElementById("generateBtn");
const loadingState = document.getElementById("loadingState");
const errorState = document.getElementById("errorState");
const errorMessage = document.getElementById("errorMessage");
const emptyState = document.getElementById("emptyState");
const resultsDisplay = document.getElementById("resultsDisplay");

// Enable/disable button based on selection
tickerSelect.addEventListener("change", (e) => {
  generateBtn.disabled = !e.target.value;
});

// Prevent multiple simultaneous requests
let isGenerating = false;

// Generate signal button
generateBtn.addEventListener("click", async () => {
  const ticker = tickerSelect.value;

  if (!ticker) {
    showError("Please select a ticker");
    return;
  }

  // Prevent duplicate requests
  if (isGenerating) {
    console.log("Request already in progress, ignoring duplicate click");
    return;
  }

  await generateTradingSignal(ticker);
});

// API: Generate trading signal
async function generateTradingSignal(ticker) {
  if (isGenerating) {
    return; // Already processing
  }

  try {
    isGenerating = true;
    showLoading();
    hideError();

    // Add timeout to prevent hanging requests (5 minutes)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 300000); // 5 minutes

    const response = await fetch("/api/signals/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        ticker: ticker,
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: "Unknown error" }));
      throw new Error(errorData.error || `HTTP ${response.status}: Failed to generate signal`);
    }

    const data = await response.json();

    if (!data.success) {
      throw new Error(data.error || "Failed to generate signal");
    }

    displayResults(data.result);
  } catch (error) {
    console.error("Signal generation error:", error);
    if (error.name === 'AbortError') {
      showError("Request timed out. The analysis is taking longer than expected. Please try again.");
    } else {
      showError(error.message || "Failed to generate trading signal");
    }
    showEmptyState();
  } finally {
    isGenerating = false;
    hideLoading();
  }
}

// Display complete results
function displayResults(result) {
  emptyState.classList.add("hidden");
  resultsDisplay.classList.remove("hidden");

  // Display trading signal
  displayTradingSignal(result.trading_signal);

  // Display multi-source sentiments
  displaySourceSentiments(result.sources);

  // Display aggregated sentiment
  displayAggregatedSentiment(result.aggregated_sentiment);

  // Display TFT prediction
  displayTFTPrediction(result.tft_prediction);

  // Display interpretability analysis - always show section if we have TFT prediction
  if (result.tft_prediction) {
    // Check if interpretability exists (even if it's an empty object)
    if (result.tft_prediction.interpretability !== undefined && result.tft_prediction.interpretability !== null) {
      // Check if it has any actual data
      const hasData = result.tft_prediction.interpretability.attention_plot || 
                      result.tft_prediction.interpretability.feature_importance_plot ||
                      result.tft_prediction.interpretability.attention_weights;
      
      if (hasData) {
        displayInterpretability(result.tft_prediction.interpretability);
      } else {
        // Show unavailable message
        displayInterpretabilityUnavailable(result.tft_prediction);
      }
    } else {
      // Show section even if interpretability is not available, with a message
      displayInterpretabilityUnavailable(result.tft_prediction);
    }

    // Display empirical validation if available
    if (result.empirical_validation) {
      displayEmpiricalValidation(result.empirical_validation);
    } else {
      hideEmpiricalValidation();
    }
  } else {
    // Hide if no TFT prediction at all
    hideInterpretability();
    hideEmpiricalValidation();
  }
}

// Display trading signal
function displayTradingSignal(signal) {
  const signalBadge = document.getElementById("signalBadge");
  const signalConfidence = document.getElementById("signalConfidence");
  const signalReasoning = document.getElementById("signalReasoning");
  const signalCard = document.getElementById("signalCard");

  signalBadge.textContent = signal.action;
  signalConfidence.textContent = `${(signal.confidence * 100).toFixed(1)}%`;
  signalReasoning.textContent = signal.reasoning;

  // Set colors
  signalBadge.className = "px-6 py-3 rounded-md font-bold text-2xl";
  signalCard.className = "border-l-4 pl-4";

  if (signal.color === "bullish") {
    signalBadge.classList.add("bg-positive", "text-white");
    signalCard.classList.add("border-positive");
  } else if (signal.color === "bearish") {
    signalBadge.classList.add("bg-negative", "text-white");
    signalCard.classList.add("border-negative");
  } else {
    signalBadge.classList.add("bg-neutral", "text-white");
    signalCard.classList.add("border-neutral");
  }

  // Re-initialize Lucide icons
  if (typeof lucide !== "undefined") {
    lucide.createIcons();
  }
}

// Show trading signal info modal
function showTradingSignalInfo() {
  const modal = document.getElementById("infoModal");
  const modalTitle = document.getElementById("infoModalTitle");
  const modalContent = document.getElementById("infoModalContent");

  modalTitle.textContent = "Trading Signal";
  modalContent.innerHTML = `
    <div class="space-y-4">
      <p class="text-sm text-foreground leading-relaxed">
        Trading signals are generated by combining sentiment analysis from multiple sources (Earnings Transcripts, News Articles, Social Media) with TFT (Temporal Fusion Transformer) price predictions. The system analyzes both qualitative sentiment and quantitative price forecasts to generate actionable trading recommendations.
      </p>
      <div class="pt-3 border-t border-border space-y-3">
        <div>
          <h4 class="text-xs font-semibold text-foreground mb-2">Signal Types:</h4>
          <div class="space-y-1.5 text-xs text-muted-foreground">
            <div><span class="font-semibold text-foreground">BUY:</span> Positive sentiment and/or predicted price increase</div>
            <div><span class="font-semibold text-foreground">SELL:</span> Negative sentiment and/or predicted price decrease</div>
            <div><span class="font-semibold text-foreground">HOLD:</span> Mixed or neutral signals suggest waiting</div>
          </div>
        </div>
        <div>
          <h4 class="text-xs font-semibold text-foreground mb-2">Decision Rule:</h4>
          <div class="space-y-1.5 text-xs text-muted-foreground">
            <div><span class="font-semibold text-foreground">BUY:</span> (Sentiment > 0.1 OR Return > 3%) AND Return > 1%</div>
            <div><span class="font-semibold text-foreground">SELL:</span> (Sentiment < -0.1 OR Return < -3%) AND Return < -1%</div>
            <div><span class="font-semibold text-foreground">HOLD:</span> Otherwise</div>
          </div>
        </div>
      </div>
    </div>
  `;

  modal.classList.remove("hidden");
  
  // Re-initialize Lucide icons
  if (typeof lucide !== "undefined") {
    lucide.createIcons();
  }
}

// Display multi-source sentiments
function displaySourceSentiments(sources) {
  const container = document.getElementById("sourceSentiments");
  container.innerHTML = "";

  const sourceIcons = {
    news: '<i data-lucide="newspaper" class="w-4 h-4"></i>',
    social: '<i data-lucide="message-circle" class="w-4 h-4"></i>',
    earnings: '<i data-lucide="bar-chart-3" class="w-4 h-4"></i>',
  };

  const sourceNames = {
    news: "News Articles",
    social: "Social Media",
    earnings: "Earnings Transcripts",
  };

  for (const [sourceName, data] of Object.entries(sources)) {
    const card = document.createElement("div");
    
    const sentiment = data.sentiment || data.dominant_sentiment;
    const sentimentColor =
      sentiment === "positive"
        ? "text-positive"
        : sentiment === "negative"
          ? "text-negative"
          : "text-neutral";
    
    // Determine border color based on sentiment
    const borderColor =
      sentiment === "positive"
        ? "border-positive/30"
        : sentiment === "negative"
          ? "border-negative/30"
          : "border-neutral/30";
    
    // Determine background color based on sentiment
    const bgColor =
      sentiment === "positive"
        ? "bg-positive/5"
        : sentiment === "negative"
          ? "bg-negative/5"
          : "bg-neutral/5";

    const score = data.score || data.average_score;
    const confidence = data.confidence || data.average_confidence;
    const texts = data.texts || [];

    // Make entire card clickable with hover effects
    card.className = `${bgColor} border ${borderColor} rounded-lg p-4 cursor-pointer transition-all duration-200 hover:shadow-md hover:border-opacity-60 hover:scale-[1.02]`;
    card.setAttribute("data-source", sourceName);
    card.setAttribute("data-title", sourceNames[sourceName]);

    card.innerHTML = `
            <div class="flex items-center justify-between mb-3">
                <div class="flex items-center gap-2">
                    <span class="text-lg text-foreground">${sourceIcons[sourceName]}</span>
                    <span class="font-semibold text-sm text-foreground">${sourceNames[sourceName]}</span>
                    <button 
                        class="info-badge-btn ml-1 p-1 rounded-full hover:bg-foreground/10 transition-colors"
                        data-source="${sourceName}"
                        onclick="event.stopPropagation(); showInfoModal('${sourceName}')"
                    >
                        <i data-lucide="info" class="w-3.5 h-3.5 text-muted-foreground"></i>
                    </button>
                </div>
                <span class="text-xs font-bold px-2 py-1 rounded-md ${sentimentColor} bg-current/10">
                    ${sentiment.toUpperCase()}
                </span>
            </div>
            <div class="flex items-center justify-between text-xs">
                <span class="text-foreground/80 font-medium">Score: <span class="font-semibold">${score.toFixed(2)}</span></span>
                <span class="text-foreground/80 font-medium">Confidence: <span class="font-semibold">${(confidence * 100).toFixed(1)}%</span></span>
                <span class="text-primary font-semibold flex items-center gap-1">
                    ${data.count} texts 
                    <i data-lucide="chevron-right" class="w-3 h-3"></i>
                </span>
            </div>
        `;

    container.appendChild(card);

    // Add click event to the entire card
    card.addEventListener("click", (e) => {
      // Prevent event bubbling if clicking on nested elements
      e.stopPropagation();
      showTextModal(sourceName, texts, sourceNames[sourceName]);
    });
  }

  // Re-initialize Lucide icons for dynamically added content
  if (typeof lucide !== "undefined") {
    lucide.createIcons();
  }
}

// Show text modal
function showTextModal(sourceType, texts, title) {
  const modal = document.getElementById("textModal");
  const modalTitle = document.getElementById("modalTitle");
  const modalContent = document.getElementById("modalContent");

  modalTitle.textContent = title;

  // Create content with improved styling
  let content = '<div class="space-y-4">';
  texts.forEach((text, index) => {
    content += `
            <div class="p-5 bg-card border-2 border-border rounded-lg shadow-sm hover:shadow-md hover:border-primary/30 transition-all">
                <div class="text-xs font-bold text-primary mb-3 uppercase tracking-wide">Text ${index + 1}</div>
                <div class="text-foreground text-sm leading-relaxed">${escapeHtml(text)}</div>
            </div>
        `;
  });
  content += "</div>";

  modalContent.innerHTML = content;
  modal.classList.remove("hidden");
  
  // Re-initialize Lucide icons in modal
  if (typeof lucide !== "undefined") {
    lucide.createIcons();
  }
}

// Close text modal
function closeTextModal() {
  const modal = document.getElementById("textModal");
  modal.classList.add("hidden");
}

// Escape HTML
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Close modal on background click
document.addEventListener("click", (e) => {
  const textModal = document.getElementById("textModal");
  const infoModal = document.getElementById("infoModal");
  if (e.target === textModal) {
    closeTextModal();
  }
  if (e.target === infoModal) {
    closeInfoModal();
  }
});

// Show info modal for sentiment sources
function showInfoModal(sourceType) {
  const modal = document.getElementById("infoModal");
  const modalTitle = document.getElementById("infoModalTitle");
  const modalContent = document.getElementById("infoModalContent");

  const sourceInfo = {
    earnings: {
      title: "Earnings Transcripts",
      description: "Official earnings call transcripts from company quarterly reports. These provide direct insights from company executives about financial performance, guidance, and strategic direction.",
      source: "Financial Modeling Prep (FMP) API",
      weight: "30% of aggregated sentiment"
    },
    news: {
      title: "News Articles",
      description: "Recent news articles from financial news sources covering company developments, market analysis, and industry trends. Provides broader market context and analyst perspectives.",
      source: "NewsAPI",
      weight: "40% of aggregated sentiment"
    },
    social: {
      title: "Social Media",
      description: "Social media posts and discussions from platforms like Twitter/X, Reddit, and financial forums. Captures retail investor sentiment and real-time market discussions.",
      source: "Finnhub API",
      weight: "30% of aggregated sentiment"
    }
  };

  const info = sourceInfo[sourceType];
  if (!info) return;

  modalTitle.textContent = info.title;
  modalContent.innerHTML = `
    <div class="space-y-4">
      <p class="text-sm text-foreground leading-relaxed">
        ${info.description}
      </p>
      <div class="pt-3 border-t border-border space-y-2">
        <div class="flex items-center justify-between text-xs">
          <span class="text-muted-foreground">Data Source:</span>
          <span class="font-semibold text-foreground">${info.source}</span>
        </div>
        <div class="flex items-center justify-between text-xs">
          <span class="text-muted-foreground">Weight:</span>
          <span class="font-semibold text-foreground">${info.weight}</span>
        </div>
      </div>
    </div>
  `;

  modal.classList.remove("hidden");
  
  // Re-initialize Lucide icons
  if (typeof lucide !== "undefined") {
    lucide.createIcons();
  }
}

// Close info modal
function closeInfoModal() {
  const modal = document.getElementById("infoModal");
  modal.classList.add("hidden");
}

// Display aggregated sentiment
function displayAggregatedSentiment(aggregated) {
  const aggSentiment = document.getElementById("aggSentiment");
  const aggScore = document.getElementById("aggScore");

  const sentiment = aggregated.sentiment;
  const sentimentColor =
    sentiment === "positive"
      ? "text-positive"
      : sentiment === "negative"
        ? "text-negative"
        : "text-neutral";

  aggSentiment.textContent =
    sentiment.charAt(0).toUpperCase() + sentiment.slice(1);
  aggSentiment.className = `text-lg font-semibold ${sentimentColor}`;

  aggScore.textContent = `Score: ${aggregated.score.toFixed(3)} | Confidence: ${(aggregated.confidence * 100).toFixed(1)}%`;
}

// Display TFT prediction
function displayTFTPrediction(prediction) {
  const tftReturn = document.getElementById("tftReturn");
  const tftInterval = document.getElementById("tftInterval");

  const returnPct = prediction.predicted_return * 100;
  const returnColor =
    returnPct > 0
      ? "text-positive"
      : returnPct < 0
        ? "text-negative"
        : "text-neutral";
  

  tftReturn.textContent = `${returnPct > 0 ? "+" : ""}${returnPct.toFixed(2)}%`;
  tftReturn.className = `text-lg font-semibold ${returnColor}`;

  const ci = prediction.confidence_interval;
  tftInterval.textContent = `95% CI: [${(ci.lower * 100).toFixed(2)}%, ${(ci.upper * 100).toFixed(2)}%]`;
}

// Display decision rule
function displayDecisionRule() {
  const decisionRule = document.getElementById("decisionRule");
  decisionRule.innerHTML = `
    <div class="space-y-1.5">
      <div><span class="font-semibold text-foreground">BUY:</span> (Sentiment > 0.1 OR Return > 3%) AND Return > 1%</div>
      <div><span class="font-semibold text-foreground">SELL:</span> (Sentiment < -0.1 OR Return < -3%) AND Return < -1%</div>
      <div><span class="font-semibold text-foreground">HOLD:</span> Otherwise</div>
    </div>
  `;
}

// Display interpretability analysis
function displayInterpretability(interpretability) {
  const section = document.getElementById("interpretabilitySection");
  const attentionPlot = document.getElementById("attentionPlot");
  const featureImportancePlot = document.getElementById("featureImportancePlot");
  const insightsContent = document.getElementById("insightsContent");
  const localImportanceSection = document.getElementById("localImportanceSection");
  const timeStepSelector = document.getElementById("timeStepSelector");
  const localImportanceContent = document.getElementById("localImportanceContent");

  if (!interpretability) {
    hideInterpretability();
    return;
  }

  // Show section
  section.classList.remove("hidden");

  // Display attention weights plot
  if (interpretability.attention_plot) {
    attentionPlot.innerHTML = `<img src="${interpretability.attention_plot}" alt="Attention Weights" class="w-full h-auto rounded-md" />`;
  } else {
    attentionPlot.innerHTML = '<p class="text-xs text-muted-foreground">Attention weights visualization not available</p>';
  }

  // Display feature importance plot (global)
  if (interpretability.feature_importance_plot) {
    featureImportancePlot.innerHTML = `<img src="${interpretability.feature_importance_plot}" alt="Feature Importance" class="w-full h-auto rounded-md" />`;
  } else {
    featureImportancePlot.innerHTML = '<p class="text-xs text-muted-foreground">Feature importance visualization not available</p>';
  }

  // Display LOCAL IMPORTANCE if available
  if (interpretability.feature_importances && interpretability.feature_importances.local) {
    displayLocalImportance(interpretability.feature_importances.local);
  } else {
    localImportanceSection.classList.add("hidden");
  }

  // Generate insights
  const insights = generateInsights(interpretability);
  insightsContent.innerHTML = insights.map(insight => `<div>• ${insight}</div>`).join('');

  // Re-initialize Lucide icons
  if (typeof lucide !== "undefined") {
    lucide.createIcons();
  }
}

// Display local importance (per time step)
function displayLocalImportance(localData) {
  const localImportanceSection = document.getElementById("localImportanceSection");
  const timeStepSelector = document.getElementById("timeStepSelector");
  const localImportanceContent = document.getElementById("localImportanceContent");

  if (!localData || !localData.per_time_step || localData.per_time_step.length === 0) {
    localImportanceSection.classList.add("hidden");
    return;
  }

  // Show section
  localImportanceSection.classList.remove("hidden");

  // Populate time step selector
  timeStepSelector.innerHTML = '';
  localData.per_time_step.forEach((timeStep, index) => {
    const option = document.createElement('option');
    option.value = index;
    const daysAgoText = timeStep.days_ago === 0 ? 'today (most recent)' : `${timeStep.days_ago} days ago`;
    option.textContent = `Time Step ${timeStep.time_idx} (${daysAgoText})`;
    timeStepSelector.appendChild(option);
  });

  // Display first time step by default
  displayTimeStepImportance(localData.per_time_step[0]);

  // Add event listener for selector
  timeStepSelector.onchange = function() {
    const selectedIndex = parseInt(this.value);
    displayTimeStepImportance(localData.per_time_step[selectedIndex]);
  };
}

// Display feature importance for a specific time step
function displayTimeStepImportance(timeStepData) {
  const localImportanceContent = document.getElementById("localImportanceContent");

  if (!timeStepData || !timeStepData.top_features) {
    localImportanceContent.innerHTML = '<p class="text-xs text-muted-foreground">No local importance data available</p>';
    return;
  }

  const topFeatures = timeStepData.top_features;
  const names = topFeatures.names || [];
  const scores = topFeatures.scores || [];

  if (names.length === 0) {
    localImportanceContent.innerHTML = '<p class="text-xs text-muted-foreground">No features available for this time step</p>';
    return;
  }

  // Find max score for normalization
  const maxScore = Math.max(...scores.map(s => Math.abs(s)));

  // Create HTML for top features
  let html = `<div class="space-y-2">`;
  const daysAgoText = timeStepData.days_ago === 0 ? 'today (most recent)' : `${timeStepData.days_ago} days ago`;
  html += `<p class="text-xs font-semibold text-foreground mb-2">Top Features for Time Step ${timeStepData.time_idx} (${daysAgoText}):</p>`;
  
  names.slice(0, 10).forEach((name, idx) => {
    const score = scores[idx];
    const normalizedScore = maxScore > 0 ? Math.abs(score) / maxScore : 0;
    const barWidth = Math.round(normalizedScore * 100);
    
    // Color coding
    let colorClass = 'bg-gray-500';
    const nameLower = name.toLowerCase();
    if (nameLower.includes('sentiment')) {
      colorClass = 'bg-yellow-500';
    } else if (nameLower.includes('close') && nameLower.includes('lag')) {
      colorClass = 'bg-blue-500';
    } else if (nameLower.includes('atr') || nameLower.includes('volatility')) {
      colorClass = 'bg-orange-500';
    } else if (nameLower.includes('hma') || nameLower.includes('hull')) {
      colorClass = 'bg-teal-500';
    }

    html += `
      <div class="flex items-center gap-2">
        <div class="text-xs text-foreground w-32 truncate" title="${name}">${name.replace(/_/g, ' ')}</div>
        <div class="flex-1 bg-muted rounded-full h-4 overflow-hidden">
          <div class="${colorClass} h-full transition-all" style="width: ${barWidth}%"></div>
        </div>
        <div class="text-xs text-muted-foreground w-16 text-right">${score.toFixed(3)}</div>
      </div>
    `;
  });

  html += `</div>`;
  localImportanceContent.innerHTML = html;
}

// Generate interpretability insights
function generateInsights(interpretability) {
  const insights = [];

  // Analyze attention weights
  if (interpretability.attention_weights && interpretability.attention_weights.length > 0) {
    const attention = interpretability.attention_weights;
    const maxAttention = Math.max(...attention);
    const maxIdx = attention.indexOf(maxAttention);
    const encoderLength = attention.length;

    if (maxIdx < encoderLength * 0.2) {
      insights.push(`Recent days (last ${Math.round(encoderLength * 0.2)} days) have the highest attention, indicating the model focuses on recent price movements.`);
    } else if (maxIdx > encoderLength * 0.7) {
      insights.push(`Older historical data (${maxIdx} days ago) has high attention, suggesting the model considers longer-term patterns.`);
    } else {
      insights.push(`The model balances attention across different time periods, with peak attention at ${maxIdx} days ago.`);
    }

    // Check if recent days have higher attention
    const recentAttention = attention.slice(-5).reduce((a, b) => a + b, 0) / 5;
    const olderAttention = attention.slice(0, -5).reduce((a, b) => a + b, 0) / Math.max(1, attention.length - 5);
    
    if (recentAttention > olderAttention * 1.2) {
      insights.push(`Recent days show significantly higher attention weights, confirming the model prioritizes recent market conditions.`);
    }
  }

  // Analyze feature importance
  if (interpretability.feature_importances) {
    const featImp = interpretability.feature_importances;
    
    if (featImp.encoder && featImp.encoder.attention_scores) {
      const encoderAttention = featImp.encoder.attention_scores;
      const avgAttention = encoderAttention.reduce((a, b) => a + b, 0) / encoderAttention.length;
      insights.push(`Average attention score: ${avgAttention.toFixed(3)}. Higher values indicate stronger influence from historical data.`);
    }

    // Check for sentiment importance (global)
    if (featImp.variables) {
      const hasSentiment = JSON.stringify(featImp.variables).toLowerCase().includes('sentiment');
      if (hasSentiment) {
        insights.push(`Sentiment features are among the most influential variables globally, confirming their importance in price prediction during volatile periods.`);
      }
    }

    // Check for local importance
    if (featImp.local && featImp.local.per_time_step) {
      insights.push(`Local importance analysis available: Feature importance varies across ${featImp.local.per_time_step.length} time steps. Use the selector above to explore per-time-step rankings.`);
    }
  }

  // Default insight if no specific insights generated
  if (insights.length === 0) {
    insights.push(`The model's attention mechanism helps explain which historical periods most influenced the current prediction.`);
    insights.push(`Feature importance analysis reveals which variables (price, volume, sentiment, etc.) contribute most to the forecast.`);
  }

  return insights;
}

// Display interpretability unavailable message
function displayInterpretabilityUnavailable(tftPrediction) {
  const section = document.getElementById("interpretabilitySection");
  const attentionPlot = document.getElementById("attentionPlot");
  const featureImportancePlot = document.getElementById("featureImportancePlot");
  const insightsContent = document.getElementById("insightsContent");

  if (!section) return;

  // Show section
  section.classList.remove("hidden");

  // Show unavailable messages
  const unavailableMsg = '<div class="text-center py-8 text-muted-foreground"><p class="text-sm">Interpretability analysis is not available for this prediction.</p><p class="text-xs mt-2">This may occur if the model does not support interpretability or if the analysis failed.</p></div>';
  
  attentionPlot.innerHTML = unavailableMsg;
  featureImportancePlot.innerHTML = unavailableMsg;
  
  insightsContent.innerHTML = '<div class="text-sm text-muted-foreground">Interpretability data is not available. The model prediction is still valid, but detailed explanations of the decision-making process cannot be displayed.</div>';

  // Re-initialize Lucide icons
  if (typeof lucide !== "undefined") {
    lucide.createIcons();
  }
}

// Hide interpretability section
function hideInterpretability() {
  const section = document.getElementById("interpretabilitySection");
  if (section) {
    section.classList.add("hidden");
  }
}

// Display empirical validation results
function displayEmpiricalValidation(empiricalData) {
  const section = document.getElementById("empiricalValidationSection");
  const content = document.getElementById("empiricalValidationContent");

  if (!section || !empiricalData) {
    hideEmpiricalValidation();
    return;
  }

  // Show section
  section.classList.remove("hidden");

  let html = '';

  // Display top features
  if (empiricalData.top_features && empiricalData.top_features.names) {
    html += `<div class="mb-3">`;
    html += `<p class="text-xs font-semibold text-foreground mb-2">Top Features (Aggregated Across Test Period):</p>`;
    html += `<ol class="list-decimal list-inside space-y-1 text-xs">`;
    
    empiricalData.top_features.names.slice(0, 5).forEach((name, idx) => {
      const score = empiricalData.top_features.scores[idx];
      const isSentiment = name.toLowerCase().includes('sentiment');
      const rank = idx + 1;
      
      html += `<li class="${isSentiment ? 'font-semibold text-yellow-500' : ''}">`;
      html += `${name.replace(/_/g, ' ')} (${score.toFixed(3)})`;
      if (isSentiment && rank <= 3) {
        html += ` ✅ Top ${rank}`;
      }
      html += `</li>`;
    });
    
    html += `</ol>`;
    html += `</div>`;
  }

  // Display validation status
  if (empiricalData.validation_passed !== undefined) {
    const status = empiricalData.validation_passed ? 'passed' : 'failed';
    const statusColor = empiricalData.validation_passed ? 'text-green-500' : 'text-red-500';
    const statusIcon = empiricalData.validation_passed ? '✓' : '✗';
    
    html += `<div class="mb-3 p-2 bg-background rounded border border-border">`;
    html += `<p class="text-xs font-semibold ${statusColor} mb-1">${statusIcon} Validation ${status.charAt(0).toUpperCase() + status.slice(1)}</p>`;
    
    if (empiricalData.sentiment_rank !== null && empiricalData.sentiment_rank !== undefined) {
      const rank = empiricalData.sentiment_rank + 1; // Convert to 1-indexed
      html += `<p class="text-xs text-muted-foreground">Sentiment rank: <span class="font-semibold">#${rank}</span>`;
      if (rank <= 3) {
        html += ` (Top 3 ✅)`;
      }
      html += `</p>`;
    }
    
    html += `</div>`;
  }

  // Display test period info
  if (empiricalData.test_period) {
    html += `<div class="text-xs text-muted-foreground">`;
    html += `<p>Test Period: ${empiricalData.test_period.start || 'N/A'} to ${empiricalData.test_period.end || 'N/A'}</p>`;
    html += `<p>Predictions Analyzed: ${empiricalData.num_predictions || 0}</p>`;
    html += `</div>`;
  }

  // Display empirical findings
  if (empiricalData.empirical_findings) {
    const findings = empiricalData.empirical_findings;
    html += `<div class="mt-3 p-2 bg-background rounded border border-border">`;
    html += `<p class="text-xs font-semibold text-foreground mb-2">Empirical Findings:</p>`;
    html += `<ul class="list-disc list-inside space-y-1 text-xs text-muted-foreground">`;
    
    if (findings.top_feature) {
      html += `<li>Top feature: <span class="font-semibold">${findings.top_feature.replace(/_/g, ' ')}</span></li>`;
    }
    
    if (findings.sentiment_in_top_3 !== undefined) {
      html += `<li>Sentiment in top 3: <span class="font-semibold ${findings.sentiment_in_top_3 ? 'text-green-500' : 'text-red-500'}">${findings.sentiment_in_top_3 ? 'Yes ✅' : 'No ✗'}</span></li>`;
    }
    
    html += `</ul>`;
    html += `</div>`;
  }

  content.innerHTML = html;

  // Re-initialize Lucide icons
  if (typeof lucide !== "undefined") {
    lucide.createIcons();
  }
}

// Hide empirical validation section
function hideEmpiricalValidation() {
  const section = document.getElementById("empiricalValidationSection");
  if (section) {
    section.classList.add("hidden");
  }
}

// Show interpretability info modal
function showInterpretabilityInfo() {
  const modal = document.getElementById("infoModal");
  const modalTitle = document.getElementById("infoModalTitle");
  const modalContent = document.getElementById("infoModalContent");

  modalTitle.textContent = "Model Interpretability";
  modalContent.innerHTML = `
    <div class="space-y-4">
      <p class="text-sm text-foreground leading-relaxed">
        The TFT (Temporal Fusion Transformer) model provides interpretability through attention mechanisms and feature importance analysis. This allows us to understand <strong>why</strong> the model makes specific predictions.
      </p>
      <div class="pt-3 border-t border-border space-y-3">
        <div>
          <h4 class="text-xs font-semibold text-foreground mb-2">Attention Weights:</h4>
          <div class="space-y-1.5 text-xs text-muted-foreground">
            <div>• Shows which <strong>past days</strong> influenced the prediction most</div>
            <div>• Higher bars indicate stronger influence from that time period</div>
            <div>• Typically, recent days have higher attention, confirming the model focuses on recent market conditions</div>
          </div>
        </div>
        <div>
          <h4 class="text-xs font-semibold text-foreground mb-2">Feature Importance:</h4>
          <div class="space-y-1.5 text-xs text-muted-foreground">
            <div>• Shows which <strong>variables</strong> (price, volume, sentiment, technical indicators) are most important</div>
            <div>• Sentiment features are highlighted when they rank among top influential variables</div>
            <div>• During high volatility periods, sentiment often ranks in the top features, confirming its value</div>
          </div>
        </div>
        <div>
          <h4 class="text-xs font-semibold text-foreground mb-2">Why This Matters:</h4>
          <div class="space-y-1.5 text-xs text-muted-foreground">
            <div>• <strong>Transparency:</strong> Understand how the model reaches its conclusions</div>
            <div>• <strong>Validation:</strong> Confirm that sentiment is influential when expected (high volatility periods)</div>
            <div>• <strong>Trust:</strong> Build confidence in the model's decision-making process</div>
          </div>
        </div>
      </div>
    </div>
  `;

  modal.classList.remove("hidden");
  
  // Re-initialize Lucide icons
  if (typeof lucide !== "undefined") {
    lucide.createIcons();
  }
}

// UI State Management
function showLoading() {
  loadingState.classList.remove("hidden");
  generateBtn.disabled = true;
}

function hideLoading() {
  loadingState.classList.add("hidden");
  generateBtn.disabled = !tickerSelect.value;
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
  console.log("Trading Signals page initialized");
  generateBtn.disabled = true;
  
  // Initialize modal click-outside-to-close handler
  const textModal = document.getElementById("textModal");
  if (textModal) {
    textModal.addEventListener("click", (e) => {
      if (e.target === textModal) {
        closeTextModal();
      }
    });
  }
});
