import { Keypair } from '@solana/web3.js';
import type { PublicKey } from '@solana/web3.js';

export class KeypairManager {
  private static instance: KeypairManager;
  public keypair: Keypair;

  private constructor() {
    this.keypair = this.loadOrGenerateKeypair();
  }

  public static getInstance(): KeypairManager {
    if (!KeypairManager.instance) {
      KeypairManager.instance = new KeypairManager();
    }
    return KeypairManager.instance;
  }

  private loadOrGenerateKeypair(): Keypair {
    // If SOLANA_SECRET_KEY is set, we parse it as a JSON array of bytes
    const secretKeyStr = process.env.SOLANA_SECRET_KEY;
    if (secretKeyStr) {
      try {
        const secretArray = Uint8Array.from(JSON.parse(secretKeyStr));
        console.log('Loaded Solana keypair from environment');
        return Keypair.fromSecretKey(secretArray);
      } catch (error) {
        console.error(
          'Failed to parse SOLANA_SECRET_KEY. Generating a new ephemeral keypair...',
          error
        );
      }
    }

    // Otherwise, generate a fresh ephemeral keypair
    const ephemeral = Keypair.generate();
    console.warn(
      'No SOLANA_SECRET_KEY found; using an ephemeral keypair. Public Key:',
      ephemeral.publicKey.toBase58()
    );
    return ephemeral;
  }
} 