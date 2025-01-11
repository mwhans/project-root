class CompanionApp {
    constructor() {
        this.api = new CompanionAPI();
        this.config = new ConfigManager();
        this.statusElement = document.getElementById('status-display');
        this.setupEventListeners();
        this.checkApiKeyStatus();
    }

    setupEventListeners() {
        // API Key management
        document.getElementById('toggle-key').addEventListener('click', () => {
            const input = document.getElementById('openai-key');
            input.type = input.type === 'password' ? 'text' : 'password';
        });

        document.getElementById('save-api-key').addEventListener('click', async () => {
            try {
                const apiKey = document.getElementById('openai-key').value;
                if (!apiKey) {
                    throw new Error('API key is required');
                }

                this.updateStatus('Updating API key...');
                await this.api.updateApiKey(apiKey);
                await this.checkApiKeyStatus();
                this.updateStatus('API key updated successfully', 'success');
            } catch (error) {
                this.updateStatus(`Error: ${error.message}`, 'error');
            }
        });

        // Save configuration
        document.getElementById('save-config').addEventListener('click', async () => {
            try {
                this.updateStatus('Saving configuration...');
                await this.config.save();
                this.updateStatus('Configuration saved successfully', 'success');
            } catch (error) {
                this.updateStatus(`Error: ${error.message}`, 'error');
            }
        });

        // Start system
        document.getElementById('start-system').addEventListener('click', async () => {
            try {
                // Check API key first
                const keyStatus = await this.api.checkApiKey();
                if (!keyStatus.configured) {
                    throw new Error('OpenAI API key must be configured before starting the system');
                }

                this.updateStatus('Starting system...');
                await this.startSystem();
                this.updateStatus('System started successfully', 'success');
            } catch (error) {
                this.updateStatus(`Error: ${error.message}`, 'error');
            }
        });

        // Load initial configuration
        window.addEventListener('load', async () => {
            try {
                this.updateStatus('Loading configuration...');
                await this.config.load();
                this.updateStatus('Configuration loaded successfully', 'success');
            } catch (error) {
                this.updateStatus(`Error: ${error.message}`, 'error');
            }
        });
    }

    async checkApiKeyStatus() {
        try {
            const status = await this.api.checkApiKey();
            const statusElement = document.getElementById('api-key-status');
            statusElement.textContent = `Status: ${status.configured ? 'Configured' : 'Not configured'}`;
            statusElement.className = `api-key-status ${status.configured ? 'configured' : 'not-configured'}`;
        } catch (error) {
            console.error('Error checking API key status:', error);
        }
    }

    updateStatus(message, type = 'info') {
        this.statusElement.textContent = message;
        this.statusElement.className = `status-${type}`;
    }

    async startSystem() {
        // 1. Validate configuration
        const errors = this.config.validate();
        if (errors.length > 0) {
            throw new Error('Invalid configuration:\n' + errors.join('\n'));
        }

        // 2. Start crawler if URLs are provided
        if (this.config.config.crawler.startUrls.length > 0) {
            this.updateStatus('Starting crawler...');
            await this.api.startCrawl(
                this.config.config.crawler.startUrls,
                this.config.config.crawler.extractionStrategy
            );
        }

        // 3. Load knowledge graph
        this.updateStatus('Loading knowledge graph...');
        await this.api.getLatestGraph(this.config.config.knowledgeGraph.domainType);

        // 4. Initialize search with configuration
        this.updateStatus('Initializing search...');
        await this.api.search(
            this.config.config.knowledgeGraph.domainType,
            '',
            {
                strategy_type: this.config.config.search.strategyType,
                categories: this.config.config.search.categories
            }
        );

        this.updateStatus('System is ready', 'success');
    }
}

// Initialize the application
const app = new CompanionApp(); 