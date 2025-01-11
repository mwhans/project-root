class CompanionAPI {
    constructor(baseUrl = '/api') {
        this.baseUrl = baseUrl;
    }

    async request(endpoint, method = 'GET', data = null) {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(`${this.baseUrl}${endpoint}`, options);
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'API request failed');
        }

        return response.json();
    }

    // API Key management
    async updateApiKey(apiKey) {
        return this.request('/update-api-key', 'POST', { api_key: apiKey });
    }

    async checkApiKey() {
        return this.request('/check-api-key');
    }

    // Knowledge Graph endpoints
    async getLatestGraph(domainType) {
        return this.request(`/latest-graph?domain_type=${encodeURIComponent(domainType)}`);
    }

    // Search endpoints
    async search(domainType, query, config) {
        return this.request(`/search/${encodeURIComponent(domainType)}`, 'POST', {
            query,
            config
        });
    }

    // Crawler endpoints
    async startCrawl(urls, extractionStrategy) {
        return this.request('/scrape', 'POST', {
            urls,
            extraction_strategy: extractionStrategy
        });
    }

    // Configuration endpoints
    async saveConfig(config) {
        return this.request('/config', 'POST', config);
    }

    async getConfig() {
        return this.request('/config');
    }

    // System status
    async getStatus() {
        return this.request('/status');
    }
} 