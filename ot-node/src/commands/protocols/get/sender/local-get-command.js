import Command from '../../../command.js';
import {
    OPERATION_ID_STATUS,
    ERROR_TYPE,
    TRIPLE_STORE_REPOSITORIES,
} from '../../../../constants/constants.js';

class LocalGetCommand extends Command {
    constructor(ctx) {
        super(ctx);
        this.config = ctx.config;
        this.operationService = ctx.getService;
        this.operationIdService = ctx.operationIdService;
        this.tripleStoreService = ctx.tripleStoreService;
        this.pendingStorageService = ctx.pendingStorageService;
        this.paranetService = ctx.paranetService;
        this.ualService = ctx.ualService;
        this.repositoryModuleManager = ctx.repositoryModuleManager;
        this.blockchainModuleManager = ctx.blockchainModuleManager;

        this.errorType = ERROR_TYPE.GET.GET_LOCAL_ERROR;
    }

    /**
     * Executes command and produces one or more events
     * @param command
     */
    async execute(command) {
        const {
            operationId,
            blockchain,
            includeMetadata,
            contract,
            knowledgeCollectionId,
            contentType,
            assertionId,
            isOperationV0,
        } = command.data;
        let { knowledgeAssetId } = command.data;
        await this.operationIdService.updateOperationIdStatus(
            operationId,
            blockchain,
            OPERATION_ID_STATUS.GET.GET_LOCAL_START,
        );

        // const response = {};

        // if (paranetUAL) {
        //     const paranetRepository = this.paranetService.getParanetRepositoryName(paranetUAL);

        //     const ual = this.ualService.deriveUAL(blockchain, contract, tokenId);
        //     const syncedAssetRecord =
        //         await this.repositoryModuleManager.getParanetSyncedAssetRecordByUAL(ual);

        //     const nquads = await this.tripleStoreService.getAssertion(
        //         paranetRepository,
        //         syncedAssetRecord.publicAssertionId,
        //     );

        //     let privateNquads;
        //     if (syncedAssetRecord.privateAssertionId) {
        //         privateNquads = await this.tripleStoreService.getAssertion(
        //             paranetRepository,
        //             syncedAssetRecord.privateAssertionId,
        //         );
        //     }

        //     if (nquads?.length) {
        //         response.assertion = nquads;
        //         if (privateNquads?.length) {
        //             response.privateAssertion = privateNquads;
        //         }
        //     } else {
        //         this.handleError(
        //             operationId,
        //             blockchain,
        //             `Couldn't find locally asset with ${ual} in paranet ${paranetUAL}`,
        //             this.errorType,
        //         );
        //     }

        //     await this.operationService.markOperationAsCompleted(
        //         operationId,
        //         blockchain,
        //         response,
        //         [
        //             OPERATION_ID_STATUS.GET.GET_LOCAL_END,
        //             OPERATION_ID_STATUS.GET.GET_END,
        //             OPERATION_ID_STATUS.COMPLETED,
        //         ],
        //     );

        //     return Command.empty();
        // }

        // else {

        const promises = [];
        this.operationIdService.emitChangeEvent(
            OPERATION_ID_STATUS.GET.GET_LOCAL_GET_ASSERTION_START,
            operationId,
            blockchain,
        );

        let assertionPromise;

        if (assertionId) {
            assertionPromise = (async () => {
                let result = null;

                for (const repository of [
                    TRIPLE_STORE_REPOSITORIES.PRIVATE_CURRENT,
                    TRIPLE_STORE_REPOSITORIES.PUBLIC_CURRENT,
                ]) {
                    // eslint-disable-next-line no-await-in-loop
                    result = await this.tripleStoreService.getV6Assertion(repository, assertionId);
                    if (result?.length) {
                        break;
                    }
                }

                if (!result?.length) {
                    result = await this.tripleStoreService.getAssertion(
                        blockchain,
                        contract,
                        knowledgeCollectionId,
                        knowledgeAssetId,
                        contentType,
                    );
                }
                this.operationIdService.emitChangeEvent(
                    OPERATION_ID_STATUS.GET.GET_LOCAL_GET_ASSERTION_END,
                    operationId,
                    blockchain,
                );

                return typeof result === 'string'
                    ? result.split('\n').filter((res) => res.length > 0)
                    : result;
            })();
        } else {
            // TODO: Do this in clean way
            if (!knowledgeAssetId) {
                try {
                    knowledgeAssetId = await this.blockchainModuleManager.getKnowledgeAssetsRange(
                        blockchain,
                        contract,
                        knowledgeCollectionId,
                    );
                } catch (error) {
                    // Asset created on old content asset storage contract
                    knowledgeAssetId = {
                        startTokenId: 1,
                        endTokenId: 1,
                        burned: [],
                    };
                }
            } else {
                // kaId is number, so transform it to range
                knowledgeAssetId = {
                    startTokenId: knowledgeAssetId,
                    endTokenId: knowledgeAssetId,
                    burned: [],
                };
            }

            assertionPromise = this.tripleStoreService
                .getAssertion(
                    blockchain,
                    contract,
                    knowledgeCollectionId,
                    knowledgeAssetId,
                    contentType,
                )
                .then((result) => {
                    this.operationIdService.emitChangeEvent(
                        OPERATION_ID_STATUS.GET.GET_LOCAL_GET_ASSERTION_END,
                        operationId,
                        blockchain,
                    );
                    return result;
                });
        }

        promises.push(assertionPromise);

        if (includeMetadata) {
            this.operationIdService.emitChangeEvent(
                OPERATION_ID_STATUS.GET.GET_LOCAL_GET_ASSERTION_METADATA_START,
                operationId,
                blockchain,
            );
            const metadataPromise = this.tripleStoreService
                .getAssertionMetadata(blockchain, contract, knowledgeCollectionId, knowledgeAssetId)
                .then((result) => {
                    this.operationIdService.emitChangeEvent(
                        OPERATION_ID_STATUS.GET.GET_LOCAL_GET_ASSERTION_METADATA_END,
                        operationId,
                        blockchain,
                    );
                    return result;
                });
            promises.push(metadataPromise);
        }

        const [assertion, metadata] = await Promise.all(promises);

        const responseData = {
            assertion: isOperationV0
                ? [...(assertion?.public ?? []), ...(assertion?.private ?? [])]
                : assertion,
            ...(includeMetadata && metadata && { metadata }),
        };

        if (assertion?.public?.length || assertion?.private?.length || assertion?.length) {
            await this.operationService.markOperationAsCompleted(
                operationId,
                blockchain,
                responseData,
                [
                    OPERATION_ID_STATUS.GET.GET_LOCAL_END,
                    OPERATION_ID_STATUS.GET.GET_END,
                    OPERATION_ID_STATUS.COMPLETED,
                ],
            );

            return Command.empty();
        }
        // }

        await this.operationIdService.updateOperationIdStatus(
            operationId,
            blockchain,
            OPERATION_ID_STATUS.GET.GET_LOCAL_END,
        );

        return this.continueSequence(command.data, command.sequence);
    }

    async handleError(operationId, blockchain, errorMessage, errorType) {
        await this.operationService.markOperationAsFailed(
            operationId,
            blockchain,
            errorMessage,
            errorType,
        );
    }

    /**
     * Builds default localGetCommand
     * @param map
     * @returns {{add, data: *, delay: *, deadline: *}}
     */
    default(map) {
        const command = {
            name: 'localGetCommand',
            delay: 0,
            transactional: false,
        };
        Object.assign(command, map);
        return command;
    }
}

export default LocalGetCommand;
