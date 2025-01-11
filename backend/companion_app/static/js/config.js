class ConfigManager {
    constructor() {
        this.config = {
            knowledgeGraph: {
                directory: '',
                domainType: ''
            },
            search: {
                categories: {},
                semanticFilter: '',
                entityTypes: [],
                extractionInstructions: ''
            },
            crawler: {
                extractionStrategy: '',
                startUrls: []
            }
        };
    }

    loadFromForm() {
        // Knowledge Graph Config
        this.config.knowledgeGraph.directory = document.getElementById('kg-directory').value;
        this.config.knowledgeGraph.domainType = document.getElementById('domain-type').value;

        // Search Config
        const categories = document.getElementById('search-categories').value
            .split(',')
            .map(cat => cat.trim())
            .filter(cat => cat.length > 0);
        
        this.config.search.categories = {};
        categories.forEach(cat => {
            this.config.search.categories[cat] = {
                queries: [`Latest developments in ${cat}`, `Best practices for ${cat}`, `Advanced ${cat} concepts`]
            };
        });

        this.config.search.semanticFilter = document.getElementById('semantic-filter').value;
        this.config.search.entityTypes = document.getElementById('entity-types').value
            .split(',')
            .map(type => type.trim())
            .filter(type => type.length > 0);
        this.config.search.extractionInstructions = document.getElementById('extraction-instructions').value;

        // Crawler Config
        this.config.crawler.extractionStrategy = document.getElementById('extraction-strategy').value;
        this.config.crawler.startUrls = document.getElementById('start-urls').value
            .split('\n')
            .map(url => url.trim())
            .filter(url => url.length > 0);
    }

    updateForm() {
        // Knowledge Graph Config
        document.getElementById('kg-directory').value = this.config.knowledgeGraph.directory;
        document.getElementById('domain-type').value = this.config.knowledgeGraph.domainType;

        // Search Config
        const categories = Object.keys(this.config.search.categories);
        document.getElementById('search-categories').value = categories.join(', ');
        document.getElementById('semantic-filter').value = this.config.search.semanticFilter;
        document.getElementById('entity-types').value = this.config.search.entityTypes.join(', ');
        document.getElementById('extraction-instructions').value = this.config.search.extractionInstructions;

        // Crawler Config
        document.getElementById('extraction-strategy').value = this.config.crawler.extractionStrategy;
        document.getElementById('start-urls').value = this.config.crawler.startUrls.join('\n');
    }

    validate() {
        const errors = [];

        // Knowledge Graph Validation
        if (!this.config.knowledgeGraph.directory) {
            errors.push('Knowledge Graph Directory is required');
        }
        if (!this.config.knowledgeGraph.domainType) {
            errors.push('Domain Type is required');
        }

        // Search Validation
        if (Object.keys(this.config.search.categories).length === 0) {
            errors.push('At least one search category is required');
        }
        if (!this.config.search.semanticFilter) {
            errors.push('Semantic Filter is required');
        }
        if (this.config.search.entityTypes.length === 0) {
            errors.push('At least one entity type is required');
        }
        if (!this.config.search.extractionInstructions) {
            errors.push('Extraction Instructions are required');
        }

        // Crawler Validation
        if (!this.config.crawler.extractionStrategy) {
            errors.push('Extraction Strategy is required');
        }

        return errors;
    }

    async save() {
        this.loadFromForm();
        const errors = this.validate();
        
        if (errors.length > 0) {
            throw new Error('Validation failed:\n' + errors.join('\n'));
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