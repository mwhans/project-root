import Command from '../../command.js';
import {
    OPERATION_ID_STATUS,
    ERROR_TYPE,
    MAX_RETRIES_READ_CACHED_PUBLISH_DATA,
    RETRY_DELAY_READ_CACHED_PUBLISH_DATA,
} from '../../../constants/constants.js';

class ReadCachedPublishDataCommand extends Command {
    constructor(ctx) {
        super(ctx);
        this.ualService = ctx.ualService;
        this.dataService = ctx.dataService;
        this.fileService = ctx.fileService;
        this.repositoryModuleManager = ctx.repositoryModuleManager;
        this.networkModuleManager = ctx.networkModuleManager;

        this.errorType = ERROR_TYPE.STORE_ASSERTION_ERROR;
    }

    async execute(command) {
        const { event } = command.data;
        const eventData = JSON.parse(event.data);
        const { id, publishOperationId, merkleRoot, byteSize } = eventData;
        const { blockchain, contractAddress } = event;
        const operationId = await this.operationIdService.generateOperationId(
            OPERATION_ID_STATUS.PUBLISH_FINALIZATION.PUBLISH_FINALIZATION_START,
            publishOperationId,
        );
        let datasetPath;
        let cachedData;

        try {
            datasetPath = this.fileService.getPendingStorageDocumentPath(publishOperationId);

            // eslint-disable-next-line no-await-in-loop
            cachedData = await this.fileService.readFile(datasetPath, true);
        } catch (error) {
            return Command.retry();
        }

        const ual = this.ualService.deriveUAL(blockchain, contractAddress, id);

        const myPeerId = this.networkModuleManager.getPeerId().toB58String();
        if (cachedData.remotePeerId === myPeerId) {
            await this.repositoryModuleManager.saveFinalityAck(
                publishOperationId,
                ual,
                cachedData.remotePeerId,
            );
        } else {
            command.sequence.push('findPublisherNodeCommand', 'networkFinalityCommand');
        }

        return this.continueSequence(
            {
                operationId,
                ual,
                blockchain,
                contract: contractAddress,
                tokenId: id,
                merkleRoot,
                remotePeerId: cachedData.remotePeerId,
                publishOperationId,
                assertion: cachedData.assertion,
                byteSize,
                cachedMerkleRoot: cachedData.merkleRoot,
            },
            command.sequence,
        );
    }

    /**
     * Builds default readCachedPublishDataCommand
     * @param map
     * @returns {{add, data: *, delay: *, deadline: *}}
     */
    default(map) {
        const command = {
            name: 'readCachedPublishDataCommand',
            delay: RETRY_DELAY_READ_CACHED_PUBLISH_DATA,
            retries: MAX_RETRIES_READ_CACHED_PUBLISH_DATA,
            transactional: false,
        };
        Object.assign(command, map);
        return command;
    }
}

export default ReadCachedPublishDataCommand;
