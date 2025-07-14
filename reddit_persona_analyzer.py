#!/usr/bin/env python3
"""
Reddit Persona Analyzer

This script analyzes Reddit user profiles to generate detailed personas based on
their posts and comments using natural language processing and machine learning.

Author: Reddit Persona Analyzer
Date: July 2025
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import logging

import praw
import requests
from bs4 import BeautifulSoup
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('persona_analyzer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RedditPersonaAnalyzer:
    """
    Main class for analyzing Reddit user personas.
    """
    
    def __init__(self):
        """Initialize the analyzer with required components."""
        self.reddit = self._setup_reddit_client()
        self.nlp = self._setup_spacy()
        self.sentiment_analyzer = self._setup_sentiment_analyzer()
        self.summarizer = self._setup_summarizer()
        self._download_nltk_requirements()
        
        # Create output directories
        os.makedirs('personas', exist_ok=True)
        os.makedirs('data', exist_ok=True)
    
    def _setup_reddit_client(self) -> praw.Reddit:
        """Setup Reddit API client using PRAW."""
        try:
            reddit = praw.Reddit(
                client_id=os.getenv('REDDIT_CLIENT_ID'),
                client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
                user_agent=os.getenv('REDDIT_USER_AGENT', 'PersonaAnalyzer/1.0'),
                ratelimit_seconds=1
            )
            # Test the connection
            reddit.user.me()
            logger.info("Reddit API client initialized successfully")
            return reddit
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            logger.info("Falling back to web scraping mode")
            return None
    
    def _setup_spacy(self) -> spacy.Language:
        """Setup spaCy NLP pipeline."""
        try:
            nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
            return nlp
        except OSError:
            logger.error("spaCy English model not found. Please run: python -m spacy download en_core_web_sm")
            sys.exit(1)
    
    def _setup_sentiment_analyzer(self):
        """Setup NLTK sentiment analyzer."""
        try:
            return SentimentIntensityAnalyzer()
        except LookupError:
            logger.info("Downloading NLTK VADER lexicon...")
            nltk.download('vader_lexicon')
            return SentimentIntensityAnalyzer()
    
    def _setup_summarizer(self):
        """Setup transformer-based summarizer."""
        try:
            summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                max_length=150,
                min_length=50,
                do_sample=False
            )
            logger.info("Summarization model loaded successfully")
            return summarizer
        except Exception as e:
            logger.warning(f"Failed to load summarization model: {e}")
            return None
    
    def _download_nltk_requirements(self):
        """Download required NLTK data."""
        nltk_downloads = ['punkt', 'stopwords', 'vader_lexicon']
        for item in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{item}')
            except LookupError:
                logger.info(f"Downloading NLTK {item}...")
                nltk.download(item, quiet=True)
    
    def extract_username_from_url(self, url: str) -> str:
        """Extract username from Reddit URL."""
        patterns = [
            r'reddit\.com/u/([^/]+)',
            r'reddit\.com/user/([^/]+)',
            r'reddit\.com/users/([^/]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # If no pattern matches, assume the URL is just a username
        return url.strip('/')
    
    def fetch_user_data_api(self, username: str) -> Dict:
        """Fetch user data using Reddit API (PRAW)."""
        if not self.reddit:
            return self.fetch_user_data_scraping(username)
        
        try:
            user = self.reddit.redditor(username)
            
            # Check if user exists
            try:
                user.id  # This will raise an exception if user doesn't exist
            except Exception:
                logger.error(f"User {username} not found or account suspended")
                return None
            
            posts = []
            comments = []
            
            logger.info(f"Fetching posts for user {username}...")
            # Fetch submissions (posts)
            for submission in user.submissions.new(limit=100):
                posts.append({
                    'id': submission.id,
                    'title': submission.title,
                    'text': submission.selftext,
                    'subreddit': submission.subreddit.display_name,
                    'score': submission.score,
                    'num_comments': submission.num_comments,
                    'created_utc': submission.created_utc,
                    'url': submission.url
                })
                time.sleep(0.1)  # Rate limiting
            
            logger.info(f"Fetching comments for user {username}...")
            # Fetch comments
            for comment in user.comments.new(limit=100):
                comments.append({
                    'id': comment.id,
                    'text': comment.body,
                    'subreddit': comment.subreddit.display_name,
                    'score': comment.score,
                    'created_utc': comment.created_utc,
                    'is_submitter': comment.is_submitter
                })
                time.sleep(0.1)  # Rate limiting
            
            user_data = {
                'username': username,
                'posts': posts,
                'comments': comments,
                'total_posts': len(posts),
                'total_comments': len(comments),
                'fetch_method': 'api'
            }
            
            logger.info(f"Fetched {len(posts)} posts and {len(comments)} comments for {username}")
            return user_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {username}: {e}")
            return self.fetch_user_data_scraping(username)
    
    def fetch_user_data_scraping(self, username: str) -> Dict:
        """Fallback method using web scraping."""
        logger.info(f"Using web scraping for user {username}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        posts = []
        comments = []
        
        try:
            # Scrape user overview page
            url = f"https://www.reddit.com/user/{username}/.json?limit=100"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                for item in data['data']['children']:
                    item_data = item['data']
                    
                    if item['kind'] == 't3':  # Submission/Post
                        posts.append({
                            'id': item_data['id'],
                            'title': item_data.get('title', ''),
                            'text': item_data.get('selftext', ''),
                            'subreddit': item_data.get('subreddit', ''),
                            'score': item_data.get('score', 0),
                            'num_comments': item_data.get('num_comments', 0),
                            'created_utc': item_data.get('created_utc', 0),
                            'url': item_data.get('url', '')
                        })
                    
                    elif item['kind'] == 't1':  # Comment
                        comments.append({
                            'id': item_data['id'],
                            'text': item_data.get('body', ''),
                            'subreddit': item_data.get('subreddit', ''),
                            'score': item_data.get('score', 0),
                            'created_utc': item_data.get('created_utc', 0)
                        })
            
            user_data = {
                'username': username,
                'posts': posts,
                'comments': comments,
                'total_posts': len(posts),
                'total_comments': len(comments),
                'fetch_method': 'scraping'
            }
            
            logger.info(f"Scraped {len(posts)} posts and {len(comments)} comments for {username}")
            return user_data
            
        except Exception as e:
            logger.error(f"Error scraping data for {username}: {e}")
            return None
    
    def analyze_text_content(self, user_data: Dict) -> Dict:
        """Analyze text content for insights."""
        all_text = []
        post_texts = []
        comment_texts = []
        
        # Collect all text
        for post in user_data['posts']:
            text = f"{post['title']} {post['text']}".strip()
            if text and text != '[deleted]' and text != '[removed]':
                all_text.append(text)
                post_texts.append(text)
        
        for comment in user_data['comments']:
            text = comment['text'].strip()
            if text and text != '[deleted]' and text != '[removed]':
                all_text.append(text)
                comment_texts.append(text)
        
        if not all_text:
            logger.warning(f"No text content found for user {user_data['username']}")
            return {}
        
        # Combine all text
        combined_text = ' '.join(all_text)
        
        analysis = {
            'text_stats': self._analyze_text_statistics(all_text),
            'sentiment': self._analyze_sentiment(all_text),
            'topics': self._extract_topics(combined_text),
            'interests': self._extract_interests(combined_text),
            'personality_traits': self._analyze_personality(combined_text),
            'subreddit_activity': self._analyze_subreddit_activity(user_data),
            'temporal_patterns': self._analyze_temporal_patterns(user_data),
            'engagement_patterns': self._analyze_engagement_patterns(user_data)
        }
        
        return analysis
    
    def _analyze_text_statistics(self, texts: List[str]) -> Dict:
        """Analyze basic text statistics."""
        if not texts:
            return {}
        
        word_counts = []
        sentence_counts = []
        all_words = []
        
        for text in texts:
            words = word_tokenize(text.lower())
            sentences = sent_tokenize(text)
            word_counts.append(len(words))
            sentence_counts.append(len(sentences))
            all_words.extend([w for w in words if w.isalpha()])
        
        # Remove stopwords for word frequency analysis
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in all_words if w not in stop_words and len(w) > 2]
        
        return {
            'total_texts': len(texts),
            'total_words': sum(word_counts),
            'avg_words_per_text': sum(word_counts) / len(word_counts) if word_counts else 0,
            'total_sentences': sum(sentence_counts),
            'avg_sentences_per_text': sum(sentence_counts) / len(sentence_counts) if sentence_counts else 0,
            'most_common_words': Counter(filtered_words).most_common(20),
            'vocabulary_size': len(set(all_words))
        }
    
    def _analyze_sentiment(self, texts: List[str]) -> Dict:
        """Analyze sentiment of texts."""
        if not texts:
            return {}
        
        sentiments = []
        
        for text in texts:
            if text.strip():
                sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
                sentiments.append(sentiment_scores)
        
        if not sentiments:
            return {}
        
        # Calculate average sentiment
        avg_sentiment = {
            'compound': sum(s['compound'] for s in sentiments) / len(sentiments),
            'positive': sum(s['pos'] for s in sentiments) / len(sentiments),
            'negative': sum(s['neg'] for s in sentiments) / len(sentiments),
            'neutral': sum(s['neu'] for s in sentiments) / len(sentiments)
        }
        
        # Determine overall sentiment tendency
        if avg_sentiment['compound'] >= 0.05:
            overall_sentiment = 'positive'
        elif avg_sentiment['compound'] <= -0.05:
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'average_sentiment': avg_sentiment,
            'overall_sentiment': overall_sentiment,
            'sentiment_distribution': {
                'positive_posts': len([s for s in sentiments if s['compound'] > 0.05]),
                'negative_posts': len([s for s in sentiments if s['compound'] < -0.05]),
                'neutral_posts': len([s for s in sentiments if -0.05 <= s['compound'] <= 0.05])
            }
        }
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics using spaCy NER and noun phrases."""
        if not text.strip():
            return []
        
        doc = self.nlp(text)
        
        # Extract named entities
        entities = [ent.text.lower() for ent in doc.ents 
                   if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT']]
        
        # Extract noun phrases
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                       if len(chunk.text.split()) <= 3]
        
        # Combine and count
        all_topics = entities + noun_phrases
        topic_counts = Counter(all_topics)
        
        # Filter out common but non-informative phrases
        filtered_topics = [topic for topic, count in topic_counts.most_common(30) 
                          if count > 1 and len(topic) > 2]
        
        return filtered_topics[:15]  # Return top 15 topics
    
    def _extract_interests(self, text: str) -> List[str]:
        """Extract potential interests using keyword matching."""
        interest_keywords = {
            'gaming': ['game', 'gaming', 'xbox', 'playstation', 'pc', 'steam', 'nintendo'],
            'technology': ['tech', 'programming', 'code', 'developer', 'software', 'computer'],
            'sports': ['football', 'basketball', 'baseball', 'soccer', 'hockey', 'tennis'],
            'music': ['music', 'song', 'album', 'band', 'concert', 'guitar', 'piano'],
            'movies': ['movie', 'film', 'cinema', 'director', 'actor', 'netflix'],
            'science': ['science', 'research', 'physics', 'chemistry', 'biology', 'space'],
            'politics': ['politics', 'government', 'election', 'policy', 'democrat', 'republican'],
            'finance': ['stock', 'crypto', 'bitcoin', 'investment', 'trading', 'money'],
            'fitness': ['gym', 'workout', 'exercise', 'fitness', 'running', 'weightlifting'],
            'food': ['food', 'cooking', 'recipe', 'restaurant', 'chef', 'cuisine']
        }
        
        text_lower = text.lower()
        detected_interests = []
        
        for interest, keywords in interest_keywords.items():
            keyword_count = sum(text_lower.count(keyword) for keyword in keywords)
            if keyword_count > 2:  # Threshold for interest detection
                detected_interests.append((interest, keyword_count))
        
        # Sort by frequency and return top interests
        detected_interests.sort(key=lambda x: x[1], reverse=True)
        return [interest for interest, count in detected_interests[:10]]
    
    def _analyze_personality(self, text: str) -> Dict:
        """Analyze personality traits based on text patterns."""
        if not text.strip():
            return {}
        
        text_lower = text.lower()
        
        # Simple personality indicators based on language patterns
        personality_indicators = {
            'analytical': ['analyze', 'data', 'research', 'study', 'evidence', 'logic'],
            'creative': ['create', 'art', 'design', 'imagine', 'innovative', 'creative'],
            'social': ['friend', 'people', 'social', 'community', 'together', 'share'],
            'optimistic': ['great', 'awesome', 'love', 'amazing', 'wonderful', 'positive'],
            'humorous': ['lol', 'haha', 'funny', 'joke', 'humor', 'hilarious'],
            'technical': ['code', 'program', 'technical', 'software', 'development', 'algorithm'],
            'helpful': ['help', 'advice', 'support', 'assist', 'guide', 'thanks']
        }
        
        trait_scores = {}
        
        for trait, keywords in personality_indicators.items():
            score = sum(text_lower.count(keyword) for keyword in keywords)
            trait_scores[trait] = score
        
        # Normalize scores
        total_score = sum(trait_scores.values())
        if total_score > 0:
            normalized_traits = {trait: (score / total_score) * 100 
                               for trait, score in trait_scores.items() if score > 0}
        else:
            normalized_traits = {}
        
        return normalized_traits
    
    def _analyze_subreddit_activity(self, user_data: Dict) -> Dict:
        """Analyze user's subreddit activity patterns."""
        subreddit_posts = defaultdict(int)
        subreddit_comments = defaultdict(int)
        
        for post in user_data['posts']:
            subreddit_posts[post['subreddit']] += 1
        
        for comment in user_data['comments']:
            subreddit_comments[comment['subreddit']] += 1
        
        # Combine activity
        all_subreddits = defaultdict(int)
        for sub, count in subreddit_posts.items():
            all_subreddits[sub] += count
        for sub, count in subreddit_comments.items():
            all_subreddits[sub] += count
        
        top_subreddits = sorted(all_subreddits.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'top_subreddits': top_subreddits,
            'total_subreddits': len(all_subreddits),
            'post_distribution': dict(subreddit_posts),
            'comment_distribution': dict(subreddit_comments)
        }
    
    def _analyze_temporal_patterns(self, user_data: Dict) -> Dict:
        """Analyze temporal posting patterns."""
        hours = defaultdict(int)
        days = defaultdict(int)
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for post in user_data['posts']:
            if post['created_utc']:
                dt = datetime.fromtimestamp(post['created_utc'], tz=timezone.utc)
                hours[dt.hour] += 1
                days[dt.weekday()] += 1
        
        for comment in user_data['comments']:
            if comment['created_utc']:
                dt = datetime.fromtimestamp(comment['created_utc'], tz=timezone.utc)
                hours[dt.hour] += 1
                days[dt.weekday()] += 1
        
        # Find most active times
        most_active_hour = max(hours.items(), key=lambda x: x[1]) if hours else (0, 0)
        most_active_day = max(days.items(), key=lambda x: x[1]) if days else (0, 0)
        
        return {
            'hourly_distribution': dict(hours),
            'daily_distribution': {day_names[day]: count for day, count in days.items()},
            'most_active_hour': most_active_hour[0] if most_active_hour[1] > 0 else None,
            'most_active_day': day_names[most_active_day[0]] if most_active_day[1] > 0 else None
        }
    
    def _analyze_engagement_patterns(self, user_data: Dict) -> Dict:
        """Analyze user engagement patterns."""
        post_scores = [post['score'] for post in user_data['posts']]
        comment_scores = [comment['score'] for comment in user_data['comments']]
        
        engagement = {}
        
        if post_scores:
            engagement['avg_post_score'] = sum(post_scores) / len(post_scores)
            engagement['max_post_score'] = max(post_scores)
            engagement['total_post_karma'] = sum(post_scores)
        
        if comment_scores:
            engagement['avg_comment_score'] = sum(comment_scores) / len(comment_scores)
            engagement['max_comment_score'] = max(comment_scores)
            engagement['total_comment_karma'] = sum(comment_scores)
        
        # Content type preference
        total_posts = len(user_data['posts'])
        total_comments = len(user_data['comments'])
        
        if total_posts + total_comments > 0:
            engagement['post_to_comment_ratio'] = total_posts / (total_posts + total_comments)
            engagement['content_preference'] = 'posts' if total_posts > total_comments else 'comments'
        
        return engagement
    
    def generate_persona_summary(self, user_data: Dict, analysis: Dict) -> str:
        """Generate a comprehensive persona summary using LLM."""
        username = user_data['username']
        
        # Prepare context for summarization
        context_parts = []
        
        # Basic stats
        stats = analysis.get('text_stats', {})
        if stats:
            context_parts.append(f"User has written {stats.get('total_texts', 0)} posts/comments with an average of {stats.get('avg_words_per_text', 0):.1f} words per post.")
        
        # Sentiment
        sentiment = analysis.get('sentiment', {})
        if sentiment:
            overall_sentiment = sentiment.get('overall_sentiment', 'neutral')
            context_parts.append(f"Overall sentiment is {overall_sentiment}.")
        
        # Interests and topics
        interests = analysis.get('interests', [])
        topics = analysis.get('topics', [])
        if interests:
            context_parts.append(f"Main interests include: {', '.join(interests[:5])}.")
        if topics:
            context_parts.append(f"Frequently discusses: {', '.join(topics[:5])}.")
        
        # Subreddit activity
        subreddit_activity = analysis.get('subreddit_activity', {})
        top_subs = subreddit_activity.get('top_subreddits', [])
        if top_subs:
            top_sub_names = [sub[0] for sub in top_subs[:5]]
            context_parts.append(f"Most active in subreddits: {', '.join(top_sub_names)}.")
        
        # Personality traits
        personality = analysis.get('personality_traits', {})
        if personality:
            top_traits = sorted(personality.items(), key=lambda x: x[1], reverse=True)[:3]
            trait_names = [trait[0] for trait in top_traits]
            context_parts.append(f"Personality traits suggest they are: {', '.join(trait_names)}.")
        
        context_text = ' '.join(context_parts)
        
        # Use LLM for summarization if available
        if self.summarizer and context_text:
            try:
                summary_input = f"Create a user persona based on Reddit activity: {context_text}"
                if len(summary_input) > 1024:  # Truncate if too long
                    summary_input = summary_input[:1024]
                
                summary = self.summarizer(summary_input, max_length=150, min_length=50, do_sample=False)
                generated_summary = summary[0]['summary_text']
                
                return f"**AI-Generated Persona Summary:**\n{generated_summary}\n\n**Detailed Analysis:**\n{context_text}"
            
            except Exception as e:
                logger.warning(f"Failed to generate LLM summary: {e}")
                return context_text
        
        return context_text
    
    def generate_persona_report(self, user_data: Dict, analysis: Dict) -> str:
        """Generate a comprehensive persona report."""
        username = user_data['username']
        
        report = f"""# Reddit User Persona: {username}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
{self.generate_persona_summary(user_data, analysis)}

## Basic Statistics
- Total Posts: {user_data['total_posts']}
- Total Comments: {user_data['total_comments']}
- Data Source: {user_data['fetch_method']}
"""
        
        # Text Statistics
        text_stats = analysis.get('text_stats', {})
        if text_stats:
            report += f"""
## Content Analysis
- Total Content Pieces: {text_stats.get('total_texts', 0)}
- Total Words: {text_stats.get('total_words', 0):,}
- Average Words per Post: {text_stats.get('avg_words_per_text', 0):.1f}
- Vocabulary Size: {text_stats.get('vocabulary_size', 0):,}

### Most Used Words
"""
            common_words = text_stats.get('most_common_words', [])
            for word, count in common_words[:10]:
                report += f"- {word}: {count} times\n"
        
        # Sentiment Analysis
        sentiment = analysis.get('sentiment', {})
        if sentiment:
            avg_sent = sentiment.get('average_sentiment', {})
            dist = sentiment.get('sentiment_distribution', {})
            report += f"""
## Sentiment Analysis
- Overall Sentiment: {sentiment.get('overall_sentiment', 'unknown').title()}
- Positivity Score: {avg_sent.get('positive', 0):.2f}
- Negativity Score: {avg_sent.get('negative', 0):.2f}
- Neutrality Score: {avg_sent.get('neutral', 0):.2f}
- Compound Score: {avg_sent.get('compound', 0):.2f}

### Sentiment Distribution
- Positive Posts: {dist.get('positive_posts', 0)}
- Negative Posts: {dist.get('negative_posts', 0)}
- Neutral Posts: {dist.get('neutral_posts', 0)}
"""
        
        # Interests and Topics
        interests = analysis.get('interests', [])
        topics = analysis.get('topics', [])
        
        if interests:
            report += f"""
## Detected Interests
{', '.join(interests)}
"""
        
        if topics:
            report += f"""
## Frequently Discussed Topics
{', '.join(topics)}
"""
        
        # Personality Traits
        personality = analysis.get('personality_traits', {})
        if personality:
            report += "\n## Personality Indicators\n"
            sorted_traits = sorted(personality.items(), key=lambda x: x[1], reverse=True)
            for trait, score in sorted_traits:
                report += f"- {trait.title()}: {score:.1f}%\n"
        
        # Subreddit Activity
        subreddit_activity = analysis.get('subreddit_activity', {})
        if subreddit_activity:
            top_subs = subreddit_activity.get('top_subreddits', [])
            report += f"""
## Subreddit Activity
- Active in {subreddit_activity.get('total_subreddits', 0)} different subreddits

### Top Subreddits by Activity
"""
            for sub, count in top_subs:
                report += f"- r/{sub}: {count} posts/comments\n"
        
        # Temporal Patterns
        temporal = analysis.get('temporal_patterns', {})
        if temporal:
            report += f"""
## Activity Patterns
- Most Active Hour: {temporal.get('most_active_hour', 'Unknown')}:00
- Most Active Day: {temporal.get('most_active_day', 'Unknown')}
"""
        
        # Engagement Patterns
        engagement = analysis.get('engagement_patterns', {})
        if engagement:
            report += "\n## Engagement Metrics\n"
            
            if 'avg_post_score' in engagement:
                report += f"- Average Post Score: {engagement['avg_post_score']:.1f}\n"
                report += f"- Highest Post Score: {engagement['max_post_score']}\n"
                report += f"- Total Post Karma: {engagement['total_post_karma']:,}\n"
            
            if 'avg_comment_score' in engagement:
                report += f"- Average Comment Score: {engagement['avg_comment_score']:.1f}\n"
                report += f"- Highest Comment Score: {engagement['max_comment_score']}\n"
                report += f"- Total Comment Karma: {engagement['total_comment_karma']:,}\n"
            
            if 'content_preference' in engagement:
                report += f"- Content Preference: {engagement['content_preference'].title()}\n"
        
        # Citations and Examples
        report += "\n## Data Citations\n"
        report += "This persona is based on publicly available Reddit posts and comments.\n"
        
        # Add some example posts/comments as citations
        if user_data['posts']:
            report += "\n### Example Posts:\n"
            for i, post in enumerate(user_data['posts'][:3]):
                if post['title'] and post['title'] != '[deleted]':
                    report += f"{i+1}. \"{post['title']}\" (r/{post['subreddit']}, Score: {post['score']})\n"
        
        if user_data['comments']:
            report += "\n### Example Comments:\n"
            for i, comment in enumerate(user_data['comments'][:3]):
                if comment['text'] and comment['text'] != '[deleted]' and len(comment['text']) > 20:
                    text_preview = comment['text'][:100] + "..." if len(comment['text']) > 100 else comment['text']
                    report += f"{i+1}. \"{text_preview}\" (r/{comment['subreddit']}, Score: {comment['score']})\n"
        
        report += f"""
## Methodology
- Data collected using {'Reddit API (PRAW)' if user_data['fetch_method'] == 'api' else 'Web scraping'}
- Text analysis performed using spaCy and NLTK
- Sentiment analysis using VADER
- {'LLM summarization using BART' if self.summarizer else 'Rule-based analysis'}

## Limitations
- Analysis based on recent public activity only
- Sentiment and personality analysis are approximations
- Some content may have been deleted or removed
- Results should be interpreted as general tendencies, not definitive characteristics
"""
        
        return report
    
    def analyze_user(self, user_url: str) -> Optional[str]:
        """Analyze a single user and generate persona report."""
        username = self.extract_username_from_url(user_url)
        logger.info(f"Starting analysis for user: {username}")
        
        # Fetch user data
        user_data = self.fetch_user_data_api(username)
        if not user_data:
            logger.error(f"Failed to fetch data for user {username}")
            return None
        
        # Analyze the data
        logger.info(f"Analyzing content for user {username}")
        analysis = self.analyze_text_content(user_data)
        
        # Generate persona report
        persona_report = self.generate_persona_report(user_data, analysis)
        
        # Save files
        safe_username = re.sub(r'[^\w\-_]', '_', username)
        
        # Save persona report
        persona_filename = f"personas/{safe_username}_persona.txt"
        with open(persona_filename, 'w', encoding='utf-8') as f:
            f.write(persona_report)
        
        # Save raw analysis data
        analysis_data = {
            'user_data': user_data,
            'analysis': analysis,
            'generated_at': datetime.now().isoformat()
        }
        
        data_filename = f"data/{safe_username}_analysis.json"
        with open(data_filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        logger.info(f"Analysis complete for {username}. Files saved:")
        logger.info(f"  - Persona: {persona_filename}")
        logger.info(f"  - Data: {data_filename}")
        
        return persona_filename
    
    def analyze_multiple_users(self, user_urls: List[str]) -> List[str]:
        """Analyze multiple users."""
        results = []
        
        for i, url in enumerate(user_urls, 1):
            logger.info(f"Processing user {i}/{len(user_urls)}")
            result = self.analyze_user(url)
            if result:
                results.append(result)
            
            # Add delay between users to respect rate limits
            if i < len(user_urls):
                time.sleep(2)
        
        return results


def main():
    """Main function to run the Reddit persona analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze Reddit user profiles to generate detailed personas"
    )
    parser.add_argument(
        'users',
        nargs='+',
        help='Reddit user URLs or usernames to analyze'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check for environment variables
    required_env_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.warning("Reddit API access may be limited. Web scraping will be used as fallback.")
    
    # Initialize analyzer
    logger.info("Initializing Reddit Persona Analyzer...")
    analyzer = RedditPersonaAnalyzer()
    
    # Analyze users
    logger.info(f"Starting analysis of {len(args.users)} user(s)")
    results = analyzer.analyze_multiple_users(args.users)
    
    # Summary
    if results:
        logger.info(f"Analysis complete! Generated {len(results)} persona files:")
        for result in results:
            logger.info(f"  - {result}")
    else:
        logger.error("No personas were generated successfully")
        sys.exit(1)


if __name__ == "__main__":
    main()
