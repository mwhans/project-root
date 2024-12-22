// src/services/apiMonitor.ts
export function monitorAPIRequests() {
    // Listen for outgoing requests
    chrome.webRequest.onBeforeRequest.addListener(
        (details) => {
            console.log("Intercepted Request:", details);
            storeRequestData(details);
        },
        {
            urls: [
                "https://wwwcie.ups.com/api/shipments/*", // UPS shipment API
                "https://wwwcie.ups.com/api/rating/*",    // UPS rating API
                "https://api-sandbox.fedex.com/ship/v1/shipments*",  // FedEx shipment API
                "https://api-sandbox.fedex.com/rate/v1/rates*",      // FedEx rate API
                "https://api.usps.com/shipping/v1/shipments*", // USPS shipment API
                "https://api.usps.com/rate/v1/pricing*",       // USPS rate API
                "https://api.usps.com/rate/v1/international-pricing*", // USPS international rate API
                "https://api.usps.com/label/v1/create*", // USPS label creation API
                "https://api-sandbox.dhl.com/dgff/transportation/shipment-booking*", // DHL shipment booking API
                "https://api-sandbox.dhl.com/freight/shipping/orders/v1*",          // DHL freight orders API
                "https://api-sandbox.dhl.com*",                                     // DHL generic API
                "https://api-sandbox-eu.dhl.com/post/de/shipping/im/v1*",           // DHL shipping API (EU)
                "https://express.api.dhl.com/mydhlapi/test*",                    // DHL express API
                "https://api-sandbox.epost.docuguide.com*",                         // Docuguide API
                "https://api-uat-vzen.dhl.com/post/advertising/print-mailing*",   // DHL advertising API
            ]
        },
        ["requestBody"]
    );

    // Inject content script to capture full response body
    chrome.webRequest.onCompleted.addListener(
        (details) => {
            injectContentScript(details.tabId);
        },
        {
            urls: [
                "https://wwwcie.ups.com/api/shipments/*", // UPS shipment API
                "https://wwwcie.ups.com/api/rating/*",    // UPS rating API
                "https://api-sandbox.fedex.com/ship/v1/shipments*",  // FedEx shipment API
                "https://api-sandbox.fedex.com/rate/v1/rates*",      // FedEx rate API
                "https://api.usps.com/shipping/v1/shipments*", // USPS shipment API
                "https://api.usps.com/rate/v1/pricing*",       // USPS rate API
                "https://api.usps.com/rate/v1/international-pricing*", // USPS international rate API
                "https://api.usps.com/label/v1/create*", // USPS label creation API
                "https://api-sandbox.dhl.com/dgff/transportation/shipment-booking*", // DHL shipment booking API
                "https://api-sandbox.dhl.com/freight/shipping/orders/v1*",          // DHL freight orders API
                "https://api-sandbox.dhl.com*",                                     // DHL generic API
                "https://api-sandbox-eu.dhl.com/post/de/shipping/im/v1*",           // DHL shipping API (EU)
                "https://express.api.dhl.com/mydhlapi/test*",                    // DHL express API
                "https://api-sandbox.epost.docuguide.com*",                         // Docuguide API
                "https://api-uat-vzen.dhl.com/post/advertising/print-mailing*",   // DHL advertising API
            ]
        }
    );
}

function storeRequestData(details) {
    const requestData = {
        url: details.url,
        method: details.method,
        requestBody: details.requestBody,
        timeStamp: details.timeStamp,
    };

    chrome.storage.local.set({ [`request_${details.requestId}`]: requestData }, () => {
        console.log("Request data saved:", requestData);
    });
}

function injectContentScript(tabId) {
    if (tabId > 0) { // Ensure a valid tab ID
        chrome.scripting.executeScript({
            target: { tabId },
            files: ["contentScript.js"]
        }, () => {
            console.log("Content script injected into tab", tabId);
        });
    }
}
