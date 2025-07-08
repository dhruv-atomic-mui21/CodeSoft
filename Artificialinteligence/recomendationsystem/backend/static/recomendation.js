document.addEventListener('DOMContentLoaded', () => {
  const moviePrefs = [
    ['action', '⚡ Action & Adventure'],
    ['comedy', '😄 Comedy'],
    ['drama', '🎭 Drama'],
    ['scifi', '🚀 Sci-Fi'],
    ['horror', '👻 Horror'],
    ['romance', '💕 Romance'],
    ['thriller', '🔥 Thriller'],
    ['animation', '🎨 Animation']
  ];
  const bookPrefs = [
    ['fiction', '📚 Fiction'],
    ['mystery', '🔍 Mystery'],
    ['biography', '👤 Biography'],
    ['science', '🔬 Science'],
    ['history', '🏛️ History'],
    ['fantasy', '🐉 Fantasy'],
    ['selfhelp', '💪 Self-Help'],
    ['romance-book', '💖 Romance']
  ];
  const productPrefs = [
    ['electronics', '📱 Electronics'],
    ['fashion', '👕 Fashion'],
    ['home', '🏠 Home & Garden'],
    ['sports', '⚽ Sports'],
    ['beauty', '💄 Beauty'],
    ['automotive', '🚗 Automotive'],
    ['books-prod', '📖 Books'],
    ['health', '🏥 Health']
  ];

  document.getElementById('movies-preferences').innerHTML = generatePreferences(moviePrefs);
  document.getElementById('books-preferences').innerHTML = generatePreferences(bookPrefs);
  document.getElementById('products-preferences').innerHTML = generatePreferences(productPrefs);
});

function generatePreferences(prefs) {
  return prefs.map(([value, label]) => `
    <label>
      <input type="checkbox" value="${value}"> ${label}
    </label>
  `).join('');
}

function switchTab(tabId, event) {
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(btn => btn.classList.remove('active'));

  document.getElementById(tabId).classList.add('tab-content', 'active');
  event.currentTarget.classList.add('active');
}

function createLoadingSpinner() {
  return `
    <div class="loading">
      <div class="spinner"></div>
      <p>⏳ Fetching recommendations...</p>
    </div>
  `;
}

async function getRecommendations(category, trending) {
  const prefsDiv = document.getElementById(`${category}-preferences`);
  const checkboxes = prefsDiv.querySelectorAll('input[type="checkbox"]:checked');
  const preferences = Array.from(checkboxes).map(cb => cb.value);

  const resultsContainer = document.getElementById(`${category}-results`);
  resultsContainer.innerHTML = createLoadingSpinner();

  try {
    const res = await fetch('/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ category, preferences })
    });

    const data = await res.json();
    console.log("Received recommendations data:", data);

    if (data.error) {
      resultsContainer.innerHTML = `<p class="no-recommendations" style="color:red;">❌ ${data.error}</p>`;
      return;
    }

    if (!data.recommendations || data.recommendations.length === 0) {
      resultsContainer.innerHTML = `
        <div class="no-recommendations">
          <h4>😞 No recommendations found</h4>
          <p>Try adjusting your preferences or try again later.</p>
        </div>
      `;
      return;
    }

    const allRecommendations = [];

    data.recommendations.forEach(entry => {
      const polishedList = Array.isArray(entry)
        ? entry.map(polishRawResponse)
        : [polishRawResponse(entry)];

      allRecommendations.push(...polishedList);
    });

    console.log("Polished recommendations:", allRecommendations);

    resultsContainer.innerHTML = `
      <div class="recommendation-grid">
        ${allRecommendations.map(r => `
          <div class="recommendation-card${trending ? ' trending' : ''}">
            <h4>${r.title}</h4>
            <p>${r.description}</p>
          </div>
        `).join('')}
      </div>
    `;

  } catch (err) {
    resultsContainer.innerHTML = `
      <div class="no-recommendations" style="color:red;">
        <h4>❌ Failed to get recommendations</h4>
        <p>Please check your connection and try again.</p>
      </div>
    `;
    console.error(err);
  }
}

function polishRawResponse(entry) {
  // If entry is a string and not valid JSON, try to extract info manually
  if (typeof entry === 'string') {
    try {
      // Attempt to fix malformed stringified objects
      const fixed = entry
        .replace(/'/g, '"')                           // Convert all ' to "
        .replace(/([{,])\s*"(\w+)"\s*([^:])/g, '$1"$2":$3')  // Add missing colons
        .replace(/",\s*}/g, '"}')                     // Remove trailing commas before }
        .replace(/",\s*]/g, '"]');                    // Remove trailing commas before ]
      const parsed = JSON.parse(fixed);

      if (Array.isArray(parsed)) {
        return parsed.map(polishRawResponse);
      } else {
        return polishRawResponse(parsed);
      }
    } catch (e) {
      // Fallback regex extraction
      const titleMatch = entry.match(/title['"]?\s*[:,]?\s*['"]?([^,'"}\n]+)/i);
      const descMatch = entry.match(/description['"]?\s*[:,]?\s*['"]?([^}'"]+)/i);

      return {
        title: titleMatch ? titleMatch[1].trim() : 'AI Recommendation',
        description: descMatch ? descMatch[1].trim() : 'No description available.'
      };
    }
  }

  // If entry is an object
  if (typeof entry === 'object' && entry !== null) {
    const title = (entry.title || "AI Recommendation").trim();
    const description = (entry.description || "").trim();
    return { title, description };
  }

  return {
    title: 'AI Recommendation',
    description: String(entry).slice(0, 300)
  };
}
function polishRawResponse(entry) {
  // If entry is a string, try to parse it as JSON or extract title/description 
  if (typeof entry === 'string') {
    try {
      // Attempt to parse JSON
      const parsed = JSON.parse(entry);
      if (Array.isArray(parsed)) {
        return parsed.map(polishRawResponse);
      } else if (typeof parsed === 'object') {
        return polishRawResponse(parsed);
      }
    } catch (e) {
      // If parsing fails, try to extract title and description using regex
      const titleMatch = entry.match(/title['"]?\s*[:,]?\s*['"]?([^,'"}\n]+)/i);
      const descMatch = entry.match(/description['"]?\s*[:,]?\s*['"]?([^}'"]+)/i);
      return {    
        title: titleMatch ? titleMatch[1].trim() : 'AI Recommendation',
        description: descMatch ? descMatch[1].trim() : 'No description available.'
      };
    } 

  if (entry && typeof entry === 'object') {
    const title = (entry.title || "").replace(/^\d+[\.\)\-]?\s*/, '').trim() || "AI Recommendation";
    const description = (entry.description || "").trim();
    return {
      title: title.length > 100 ? title.slice(0, 97) + '...' : title,
      description: description.length > 500 ? description.slice(0, 497) + '...' : description
    };
  }

  return {
    title: "AI Recommendation",
    description: String(entry).slice(0, 500)
  };
}
function toggleTrending(category) {
  const trendingButton = document.getElementById(`${category}-trending`);
  const isTrending = trendingButton.classList.contains('active');

  if (isTrending) {
    trendingButton.classList.remove('active');
    getRecommendations(category, false);
  } else {
    trendingButton.classList.add('active');
    getRecommendations(category, true);
  }
}
function toggleAllCheckboxes(category, checked) {
  const prefsDiv = document.getElementById(`${category}-preferences`);
  const checkboxes = prefsDiv.querySelectorAll('input[type="checkbox"]');
  checkboxes.forEach(cb => cb.checked = checked);
} 
function resetPreferences(category) {
  const prefsDiv = document.getElementById(`${category}-preferences`);
  const checkboxes = prefsDiv.querySelectorAll('input[type="checkbox"]');
  checkboxes.forEach(cb => cb.checked = false);
  document.getElementById(`${category}-results`).innerHTML = '';
}
function resetAllPreferences() {
  const categories = ['movies', 'books', 'products'];
  categories.forEach(category => resetPreferences(category));
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
  document.getElementById('movies-tab').classList.add('active');
}