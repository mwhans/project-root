import { ethers } from 'ethers';
import { BLOCKCHAINS, ABIs, CONTENT_ASSET_STORAGE_CONTRACT } from './constants.js';
import {
    validateProvider,
    validateStorageContractAddress,
    validateStorageContractName,
    validateStorageContractAbi,
    validateBlockchainDetails,
} from './validation.js';
import logger from './logger.js';

function maskRpcUrl(url) {
    // Validation
    if (!url || typeof url !== 'string') {
        throw new Error(`URL is not defined or it is not a string. URL: ${url}`);
    }

    if (url.includes('apiKey')) {
        return url.split('apiKey')[0];
    }
    return url;
}

// Initialize rpc
export async function initializeRpc(rpcEndpoint) {
    // Validation
    if (!rpcEndpoint || typeof rpcEndpoint !== 'string') {
        logger.error(
            `RPC endpoint is not defined or it is not a string. RPC endpoint: ${rpcEndpoint}`,
        );
        process.exit(1);
    }
    // initialize all possible providers
    const Provider = ethers.providers.JsonRpcProvider;

    try {
        const provider = new Provider(rpcEndpoint);
        // eslint-disable-next-line no-await-in-loop
        await provider.getNetwork();
        logger.info(`Connected to the blockchain RPC: ${maskRpcUrl(rpcEndpoint)}.`);
        return provider;
    } catch (e) {
        logger.error(`Unable to connect to the blockchain RPC: ${maskRpcUrl(rpcEndpoint)}.`);
        process.exit(1);
    }
}

export async function getStorageContractAndAddress(
    provider,
    storageContractAddress,
    storageContractName,
    storageContractAbi,
) {
    // Validation
    validateProvider(provider);
    validateStorageContractAddress(storageContractAddress);
    validateStorageContractName(storageContractName);
    validateStorageContractAbi(storageContractAbi);

    logger.info(
        `Initializing asset contract: ${storageContractName} with address: ${storageContractAddress}`,
    );
    // initialize asset contract
    const storageContract = new ethers.Contract(
        storageContractAddress,
        storageContractAbi,
        provider,
    );

    logger.info(
        `Contract ${storageContractName} initialized with address: ${storageContractAddress}`,
    );
    return storageContract;
}

export async function getContentAssetStorageContract(provider, blockchainDetails) {
    // Validation
    validateProvider(provider);
    validateBlockchainDetails(blockchainDetails);

    const contentAssetStorageContarct =
        blockchainDetails.NAME === BLOCKCHAINS.NEUROWEB_TESTNET.NAME ||
        blockchainDetails.NAME === BLOCKCHAINS.NEUROWEB_MAINNET.NAME
            ? ABIs.ContentAssetStorage
            : ABIs.ContentAssetStorageV2;
    const storageContract = await getStorageContractAndAddress(
        provider,
        blockchainDetails.CONTENT_ASSET_STORAGE_CONTRACT_ADDRESS,
        CONTENT_ASSET_STORAGE_CONTRACT,
        contentAssetStorageContarct,
    );

    return storageContract;
}
