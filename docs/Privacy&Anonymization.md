# Anonymization:
- 2 step raw data and blockchain publishing anonymization process
  - Raw data is anonymized through tokenization and hashing as well as data aggregation and generalization. Future enhancements will include differential privacy and zero knowledge proofs.
  - When publishing to the DKG blockchain an ephemeral address system will be implemented. This will be enhanced with zkPoof capabilities as time goes on.
- A tor integration will be built into the backend to handle asynchronous publishing of anonymized data to the blockchain. 
- A quantum secure TLS implementation will be built into the backend to handle secure communication with the DKG blockchain.
- All data including user data and model information will be encrypted when published to the blockchain. Only users who are a participant or have sufficient privileges as defined by governance are able to access and decrypt to view and use.
- Governance model will need to be pseudonymous. TBD structure. 