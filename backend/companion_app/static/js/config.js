class ConfigManager {
    constructor() {
        this.config = {
            knowledgeGraph: {
                directory: '',
                domainType: ''
            },
            search: {
                strategyType: '',
                categories: []
            },
            crawler: {
                extractionStrategy: '',
                startUrls: []
            }
        };
    }

    loadFromForm() {
        // Knowledge Graph
        this.config.knowledgeGraph.directory = document.getElementById('kg-directory').value;
        this.config.knowledgeGraph.domainType = document.getElementById('domain-type').value;

        // Search
        this.config.search.strategyType = document.getElementById('strategy-type').value;
        this.config.search.categories = document.getElementById('search-categories').value
            .split(',')
            .map(cat => cat.trim())
            .filter(cat => cat);

        // Crawler
        this.config.crawler.extractionStrategy = document.getElementById('extraction-strategy').value;
        this.config.crawler.startUrls = document.getElementById('start-urls').value
            .split('\n')
            .map(url => url.trim())
            .filter(url => url);
    }

    updateForm() {
        // Knowledge Graph
        document.getElementById('kg-directory').value = this.config.knowledgeGraph.directory;
        document.getElementById('domain-type').value = this.config.knowledgeGraph.domainType;

        // Search
        document.getElementById('strategy-type').value = this.config.search.strategyType;
        document.getElementById('search-categories').value = this.config.search.categories.join(', ');

        // Crawler
        document.getElementById('extraction-strategy').value = this.config.crawler.extractionStrategy;
        document.getElementById('start-urls').value = this.config.crawler.startUrls.join('\n');
    }

    validate() {
        const errors = [];

        if (!this.config.knowledgeGraph.directory) {
            errors.push('Knowledge Graph Directory is required');
        }
        if (!this.config.knowledgeGraph.domainType) {
            errors.push('Domain Type is required');
        }
        if (!this.config.search.strategyType) {
            errors.push('Strategy Type is required');
        }
        if (this.config.search.categories.length === 0) {
            errors.push('At least one Search Category is required');
        }
        if (!this.config.crawler.extractionStrategy) {
            errors.push('Extraction Strategy is required');
        }

        return errors;
    }

    async save() {
        this.loadFromForm();
        const errors = this.validate();
        
        if (errors.length > 0) {
            throw new Error(errors.join('\n'));
        }

        const api = new CompanionAPI();
        await api.saveConfig(this.config);
    }

    async load() {
        const api = new CompanionAPI();
        this.config = await api.getConfig();
        this.updateForm();
    }
} 