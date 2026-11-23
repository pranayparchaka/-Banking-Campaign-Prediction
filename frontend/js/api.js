/* API client. The URL can be set with window.BANKING_API_BASE_URL or localStorage. */
(function () {
  const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000/api";

  function getApiBaseUrl() {
    const configured = window.BANKING_API_BASE_URL || localStorage.getItem("banking_api_base_url");
    if (configured) return configured.replace(/\/$/, "");
    // The standalone local static-server command uses port 5500 and talks to
    // the separate local FastAPI process. A deployed FastAPI app is same-origin.
    if (window.location && window.location.port === "5500") return DEFAULT_API_BASE_URL;
    if (window.location && (window.location.protocol === "http:" || window.location.protocol === "https:")) return `${window.location.origin}/api`;
    return DEFAULT_API_BASE_URL;
  }

  function setApiBaseUrl(url) {
    const normalized = String(url || "").trim().replace(/\/$/, "");
    if (!normalized) throw new Error("API base URL cannot be empty.");
    localStorage.setItem("banking_api_base_url", normalized);
    return normalized;
  }

  async function request(path, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    try {
      const response = await fetch(`${getApiBaseUrl()}${path}`, {
        ...options,
        headers: { "Content-Type": "application/json", ...(options.headers || {}) },
        signal: controller.signal,
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        const message = body?.error?.message || body?.detail || `Request failed with status ${response.status}.`;
        const error = new Error(message);
        error.status = response.status;
        error.details = body?.error?.details || body?.detail || null;
        throw error;
      }
      return body;
    } catch (error) {
      if (error.name === "AbortError") throw new Error("The API request timed out. Confirm that the backend is running.");
      if (error instanceof TypeError) throw new Error(`Cannot connect to the backend at ${getApiBaseUrl()}.`);
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  window.BankingApi = {
    getApiBaseUrl,
    setApiBaseUrl,
    health: () => request("/health"),
    summary: () => request("/summary"),
    modelInfo: () => request("/model-info"),
    rocCurve: () => request("/roc-curve"),
    predict: (customer) => request("/predict", { method: "POST", body: JSON.stringify(customer) }),
    prioritize: (customers) => request("/prioritize", { method: "POST", body: JSON.stringify({ customers }) }),
    capacityAnalysis: (customers, capacityPercentage) => request("/capacity-analysis", { method: "POST", body: JSON.stringify({ customers, capacity_percentage: Number(capacityPercentage) }) }),
  };
})();

