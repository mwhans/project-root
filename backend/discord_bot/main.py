import os
from dotenv import load_dotenv
from src.bot import LegalSystemBot
from src.utils import setup_logging, load_config
from loguru import logger

def main():
    # Load environment variables
    load_dotenv()
    
    # Setup logging
    setup_logging()
    
    # Get bot token
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.error("DISCORD_TOKEN not found in environment variables")
        return
    
    try:
        # Create and run bot
        bot = LegalSystemBot()
        logger.info("Starting bot...")
        bot.run(token)
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}", exc_info=True)
        
if __name__ == "__main__":
    main() 