class CompanionAPI {
    constructor() {
        this.baseUrl = '/api';
    }

    async request(endpoint, method = 'GET', data = null) {
        const options = {
            method,
            headers: {
                'Content-Type': 'application/json'
            }
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(`${this.baseUrl}${endpoint}`, options);
        
        if (!response.ok) {
            const error = await response.text();
            throw new Error(`API request failed: ${error}`);
        }

        return response.json();
    }

    async getLatestGraph(domainType) {
        return this.request(`/latest-graph/${domainType}`);
    }

    async search(domainType, query, config) {
        return this.request(`/search/${domainType}`, 'POST', {
            query,
            config
        });
    }

    async startCrawl(urls, extractionStrategy) {
        return this.request('/scrape', 'POST', {
            urls,
            extraction_strategy: extractionStrategy
        });
    }

    async saveConfig(config) {
        return this.request('/config', 'POST', config);
    }

    async getConfig() {
        return this.request('/config');
    }

    async getStatus() {
        return this.request('/status');
    }

    async updateApiKey(apiKey) {
        return this.request('/update-api-key', 'POST', { api_key: apiKey });
    }

    async checkApiKey() {
        return this.request('/check-api-key');
    }
} 