# Legal System Discord Bot

A Discord bot for managing a legal system server structure with roles, permissions, and automated channel management.

## Features

- Automated server setup with predefined categories and channels
- Role-based permission system with hierarchy
- Channel archiving for inactive channels
- Server structure backup and restore
- Comprehensive logging system
- Rate limit handling to avoid Discord API restrictions

## Setup

1. Create a Discord application and bot at https://discord.com/developers/applications

2. Clone this repository and install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Discord bot token:
```
DISCORD_TOKEN=your_bot_token_here
```

4. Customize the server structure in `config/server_structure.json` if needed

## Usage

Start the bot:
```bash
python main.py
```

### Available Commands

- `!legal setup_server` - Sets up the entire server structure
- `!legal backup_server` - Creates a backup of the current server structure
- `!legal archive_channel [days]` - Archives channels inactive for the specified number of days

### Role Hierarchy

1. Supreme Court Justice (Administrator)
2. Judge (Channel & Message Management)
3. Lawyer (Enhanced Member Permissions)
4. Citizen (Basic Member Permissions)

## Directory Structure

```
legal-discord-bot/
├── config/
│   └── server_structure.json
├── src/
│   ├── __init__.py
│   ├── bot.py
│   ├── channel_manager.py
│   ├── permissions_manager.py
│   └── utils.py
├── logs/
├── backups/
├── .env
├── requirements.txt
└── README.md
```

## Development

- Logs are stored in the `logs/` directory with rotation
- Backups are stored in the `backups/` directory
- Rate limiting is implemented to avoid Discord API restrictions

## Error Handling

The bot includes comprehensive error handling and logging:
- Console output for INFO level and above
- File logging for DEBUG level and above
- Automatic log rotation and compression
- Detailed error messages for troubleshooting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - feel free to use and modify as needed. 