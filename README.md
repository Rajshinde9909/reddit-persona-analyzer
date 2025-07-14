# Reddit Persona Analysis Project

This project analyzes Reddit user profiles to generate detailed personas based on their posts and comments using natural language processing and machine learning techniques.

## Features

- Scrapes Reddit user data using PRAW API
- Performs sentiment analysis and topic extraction
- Generates comprehensive user personas using LLM
- Outputs detailed persona reports with citations
- Supports batch processing of multiple users

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 3. Configure Reddit API

1. Go to [Reddit App Preferences](https://www.reddit.com/prefs/apps)
2. Click "Create App" or "Create Another App"
3. Fill in the form:
   - **Name**: Your app name (e.g., "PersonaAnalyzer")
   - **App type**: Select "script"
   - **Description**: Brief description of your app
   - **About URL**: Can be left blank
   - **Redirect URI**: Use `http://localhost:8080`
4. Click "Create app"
5. Note down your **Client ID** (under the app name) and **Client Secret**

### 4. Set Up Environment Variables

Create a `.env` file in the project root with your Reddit credentials:

```
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=YourAppName/1.0 by YourUsername
```

## Usage

### Analyze a Single User

```bash
python reddit_persona_analyzer.py https://www.reddit.com/user/username/
```

### Analyze Multiple Users

```bash
python reddit_persona_analyzer.py https://www.reddit.com/user/user1/ https://www.reddit.com/user/user2/
```

### Run with Sample Profiles

```bash
python reddit_persona_analyzer.py https://www.reddit.com/user/kojied/ https://www.reddit.com/user/Hungry-Move-6603/
```

## Output

The script generates:
- Individual persona text files for each user (e.g., `kojied_persona.txt`)
- JSON files with raw analysis data
- Summary statistics and insights

## Project Structure

```
reddit-persona-analyzer/
├── reddit_persona_analyzer.py    # Main script
├── requirements.txt               # Python dependencies
├── .env                          # Environment variables (create this)
├── README.md                     # This file
├── personas/                     # Generated persona files
│   ├── kojied_persona.txt
│   └── Hungry-Move-6603_persona.txt
└── data/                         # Raw analysis data (JSON)
```

## Assumptions and Limitations

### Assumptions
- Users have public profiles with accessible posts/comments
- Recent activity (last 100 posts/comments) is representative of user behavior
- English language content (spaCy model limitation)

### Limitations
- Reddit API rate limits (1 request per second for PRAW)
- Limited to publicly available data
- Analysis quality depends on available content volume
- Some users may have minimal post history
- Deleted or removed content is not accessible

### Technical Limitations
- Maximum 1000 posts/comments per user (API limitation)
- Text length limits for LLM processing
- Requires internet connection for API access
- Processing time depends on user activity volume

## Privacy and Ethics

This tool:
- Only accesses publicly available Reddit data
- Respects Reddit's API terms of service
- Does not store personal identifying information
- Generates personas for research/analysis purposes only

## Troubleshooting

### Common Issues

1. **Reddit API Authentication Error**
   - Verify your `.env` file credentials
   - Ensure your Reddit app is configured as "script" type

2. **Rate Limiting**
   - The script includes built-in rate limiting
   - If you encounter issues, increase delay between requests

3. **Missing spaCy Model**
   - Run: `python -m spacy download en_core_web_sm`

4. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

## Contributing

Feel free to submit issues and pull requests to improve the analysis accuracy or add new features.

## License

This project is for educational and research purposes. Please respect Reddit's terms of service and user privacy.
