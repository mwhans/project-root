(function () {
    const originalFetch = window.fetch;
    window.fetch = async function (...args: Parameters<typeof fetch>): Promise<Response> {
        const response = await originalFetch(...args);
        const clonedResponse = response.clone();
        const responseBody = await clonedResponse.text();

        // Send response body to background script
        chrome.runtime.sendMessage({
            type: 'FETCH_RESPONSE',
            url: clonedResponse.url,
            status: clonedResponse.status,
            body: responseBody,
        });

        return response;
    };

    const originalXHR = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.send = function (...args: any[]): void {
        this.addEventListener('load', function () {
            chrome.runtime.sendMessage({
                type: 'XHR_RESPONSE',
                url: this.responseURL,
                status: this.status,
                body: this.responseText,
            });
        });
        originalXHR.apply(this, args);
    };
})();
