// Define the structure of incoming messages
interface Message {
    type: 'FETCH_RESPONSE' | 'XHR_RESPONSE' | 'SYNC_TO_BACKEND';
    url?: string;
    status?: number;
    body?: string;
}

// Local backend configuration
const LOCAL_BACKEND_URL = 'http://localhost:3000/data'; // Replace 3000 with your backend port

// Listen for messages from the content script
chrome.runtime.onMessage.addListener(async (message: Message, sender, sendResponse) => {
    if (message.type === 'FETCH_RESPONSE' || message.type === 'XHR_RESPONSE') {
        console.log(`${message.type} captured:`, message);
        saveResponseData(message);
    } else if (message.type === 'SYNC_TO_BACKEND') {
        console.log('Syncing data to local backend...');
        try {
            await syncDataToBackend();
            console.log('Data synced successfully.');
            sendResponse({ status: 'success' });
        } catch (error) {
            console.error('Error syncing data to backend:', error);
            sendResponse({ status: 'error', error });
        }
    } else {
        console.warn('Unknown message type received:', message.type);
    }
    return true; // Keep the message channel open for asynchronous responses
});

// Save response data to Chrome storage
function saveResponseData(data: Message): void {
    const responseData = {
        url: data.url,
        status: data.status,
        body: data.body,
        timeStamp: Date.now(),
    };

    chrome.storage.local.set({ [`response_${responseData.timeStamp}`]: responseData }, () => {
        console.log('Response data saved:', responseData);
    });
}

// Sync stored data to local backend
async function syncDataToBackend(): Promise<void> {
    const storedData = await getStoredData();
    if (!storedData) {
        throw new Error('No data available to sync.');
    }

    const isAvailable = await isBackendAvailable();
    if (!isAvailable) {
        throw new Error('Local backend is not available.');
    }

    const response = await fetch(LOCAL_BACKEND_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(storedData),
    });

    if (!response.ok) {
        throw new Error(`Failed to sync data: ${response.statusText}`);
    }

    console.log('Data synced to local backend:', await response.json());
}

// Check if the local backend is available
async function isBackendAvailable(): Promise<boolean> {
    try {
        const response = await fetch(`${LOCAL_BACKEND_URL}/health`);
        return response.ok;
    } catch (error) {
        console.error('Backend health check failed:', error);
        return false;
    }
}

// Retrieve stored data from Chrome storage
async function getStoredData(): Promise<any> {
    return new Promise((resolve, reject) => {
        chrome.storage.local.get(null, (items) => {
            if (chrome.runtime.lastError) {
                reject(chrome.runtime.lastError);
            } else {
                resolve(items);
            }
        });
    });
}

// Example usage of the storage data for debugging purposes
chrome.runtime.onInstalled.addListener(() => {
    console.log('API Monitor Extension installed.');

    // Clear old storage on install for fresh debugging
    chrome.storage.local.clear(() => {
        console.log('Storage cleared.');
    });
});
