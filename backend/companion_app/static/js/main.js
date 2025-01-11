class CompanionApp {
    constructor() {
        this.api = new CompanionAPI();
        this.configManager = new ConfigManager();
        this.statusDisplay = document.getElementById('status-display');
        
        this.setupEventListeners();
        this.checkApiKeyStatus();
        this.loadInitialConfig();
    }

    setupEventListeners() {
        // API Key Management
        document.getElementById('toggle-key').addEventListener('click', () => {
            const input = document.getElementById('openai-key');
            input.type = input.type === 'password' ? 'text' : 'password';
        });

        document.getElementById('save-api-key').addEventListener('click', async () => {
            const apiKey = document.getElementById('openai-key').value;
            if (!apiKey) {
                this.updateStatus('API Key is required', 'error');
                return;
            }

            try {
                await this.api.updateApiKey(apiKey);
                this.updateStatus('API Key saved successfully', 'success');
                await this.checkApiKeyStatus();
            } catch (error) {
                this.updateStatus(`Failed to save API Key: ${error.message}`, 'error');
            }
        });

        // Configuration Management
        document.getElementById('save-config').addEventListener('click', async () => {
            try {
                await this.configManager.save();
                this.updateStatus('Configuration saved successfully', 'success');
            } catch (error) {
                this.updateStatus(`Failed to save configuration: ${error.message}`, 'error');
            }
        });

        // System Start
        document.getElementById('start-system').addEventListener('click', async () => {
            try {
                await this.startSystem();
            } catch (error) {
                this.updateStatus(`Failed to start system: ${error.message}`, 'error');
            }
        });
    }

    async checkApiKeyStatus() {
        try {
            const status = await this.api.checkApiKey();
            const statusElement = document.getElementById('api-key-status');
            if (status.configured) {
                statusElement.textContent = 'Status: Configured';
                statusElement.classList.remove('not-configured');
                statusElement.classList.add('configured');
            } else {
                statusElement.textContent = 'Status: Not configured';
                statusElement.classList.remove('configured');
                statusElement.classList.add('not-configured');
            }
        } catch (error) {
            console.error('Failed to check API key status:', error);
        }
    }

    async loadInitialConfig() {
        try {
            await this.configManager.load();
            this.updateStatus('Configuration loaded successfully', 'success');
        } catch (error) {
            this.updateStatus('No saved configuration found', 'info');
        }
    }

    async startSystem() {
        try {
            // Check API key first
            const keyStatus = await this.api.checkApiKey();
            if (!keyStatus.configured) {
                throw new Error('OpenAI API Key must be configured before starting the system');
            }

            // Load and validate configuration
            await this.configManager.loadFromForm();
            const errors = this.configManager.validate();
            if (errors.length > 0) {
                throw new Error('Configuration validation failed:\n' + errors.join('\n'));
            }

            this.updateStatus('Starting system...', 'info');

            // Start crawling if URLs are provided
            const startUrls = this.configManager.config.crawler.startUrls;
            if (startUrls.length > 0) {
                this.updateStatus('Starting crawler...', 'info');
                await this.api.startCrawl(
                    startUrls,
                    this.configManager.config.crawler.extractionStrategy
                );
            }

            // Load latest knowledge graph
            this.updateStatus('Loading knowledge graph...', 'info');
            const graph = await this.api.getLatestGraph(
                this.configManager.config.knowledgeGraph.domainType
            );

            // Initialize search with configuration
            this.updateStatus('Initializing search...', 'info');
            for (const [category, config] of Object.entries(this.configManager.config.search.categories)) {
                for (const query of config.queries) {
                    await this.api.search(
                        this.configManager.config.knowledgeGraph.domainType,
                        query,
                        {
                            semanticFilter: this.configManager.config.search.semanticFilter,
                            entityTypes: this.configManager.config.search.entityTypes,
                            extractionInstructions: this.configManager.config.search.extractionInstructions
                        }
                    );
                }
            }

            this.updateStatus('System started successfully', 'success');
        } catch (error) {
            throw new Error(`System startup failed: ${error.message}`);
        }
    }

    updateStatus(message, type = 'info') {
        this.statusDisplay.textContent = message;
        this.statusDisplay.className = `status-${type}`;
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    window.app = new CompanionApp();
}); 