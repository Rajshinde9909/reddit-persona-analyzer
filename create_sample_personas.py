#!/usr/bin/env python3
"""
Demo script to create sample personas for Reddit users when API access is not available.
This demonstrates the expected output format and functionality.
"""

import json
import os
from datetime import datetime

# Create output directories
os.makedirs('personas', exist_ok=True)
os.makedirs('data', exist_ok=True)

def create_sample_persona_kojied():
    """Create a sample persona for kojied user."""
    
    # Sample data that would be collected from Reddit
    sample_data = {
        'username': 'kojied',
        'posts': [
            {
                'id': 'sample1',
                'title': 'Just finished my CS degree, looking for advice on entering tech',
                'text': 'Hey everyone! I just graduated with a Computer Science degree and I\'m looking for advice on breaking into the tech industry. Any tips?',
                'subreddit': 'cscareerquestions',
                'score': 15,
                'num_comments': 8,
                'created_utc': 1672531200,
                'url': 'https://reddit.com/r/cscareerquestions/sample1'
            },
            {
                'id': 'sample2',
                'title': 'Best resources for learning React?',
                'text': 'I\'m trying to learn React for web development. What are the best resources you\'d recommend for beginners?',
                'subreddit': 'webdev',
                'score': 23,
                'num_comments': 12,
                'created_utc': 1672617600,
                'url': 'https://reddit.com/r/webdev/sample2'
            }
        ],
        'comments': [
            {
                'id': 'comment1',
                'text': 'I completely agree with this approach. I\'ve been using JavaScript for 2 years now and React really changed my perspective on frontend development.',
                'subreddit': 'reactjs',
                'score': 5,
                'created_utc': 1672704000,
                'is_submitter': False
            },
            {
                'id': 'comment2',
                'text': 'Thanks for sharing this! As someone new to programming, this really helps clarify the concepts.',
                'subreddit': 'learnprogramming',
                'score': 8,
                'created_utc': 1672790400,
                'is_submitter': False
            }
        ],
        'total_posts': 2,
        'total_comments': 2,
        'fetch_method': 'demo'
    }
    
    # Sample analysis results
    analysis = {
        'text_stats': {
            'total_texts': 4,
            'total_words': 156,
            'avg_words_per_text': 39.0,
            'total_sentences': 12,
            'avg_sentences_per_text': 3.0,
            'most_common_words': [
                ('programming', 5), ('react', 4), ('development', 3), 
                ('javascript', 3), ('learning', 3), ('advice', 2)
            ],
            'vocabulary_size': 98
        },
        'sentiment': {
            'average_sentiment': {
                'compound': 0.3,
                'positive': 0.25,
                'negative': 0.05,
                'neutral': 0.7
            },
            'overall_sentiment': 'positive',
            'sentiment_distribution': {
                'positive_posts': 3,
                'negative_posts': 0,
                'neutral_posts': 1
            }
        },
        'topics': [
            'computer science', 'web development', 'react', 'javascript', 
            'programming', 'tech industry', 'frontend development'
        ],
        'interests': [
            'technology', 'programming', 'web development'
        ],
        'personality_traits': {
            'helpful': 35.0,
            'analytical': 25.0,
            'social': 20.0,
            'technical': 20.0
        },
        'subreddit_activity': {
            'top_subreddits': [
                ('cscareerquestions', 1),
                ('webdev', 1),
                ('reactjs', 1),
                ('learnprogramming', 1)
            ],
            'total_subreddits': 4,
            'post_distribution': {'cscareerquestions': 1, 'webdev': 1},
            'comment_distribution': {'reactjs': 1, 'learnprogramming': 1}
        },
        'temporal_patterns': {
            'most_active_hour': 14,
            'most_active_day': 'Monday'
        },
        'engagement_patterns': {
            'avg_post_score': 19.0,
            'max_post_score': 23,
            'total_post_karma': 38,
            'avg_comment_score': 6.5,
            'max_comment_score': 8,
            'total_comment_karma': 13,
            'content_preference': 'posts'
        }
    }
    
    return sample_data, analysis

def create_sample_persona_hungry_move():
    """Create a sample persona for Hungry-Move-6603 user."""
    
    sample_data = {
        'username': 'Hungry-Move-6603',
        'posts': [
            {
                'id': 'sample3',
                'title': 'What\'s your favorite workout routine?',
                'text': 'I\'ve been trying to get back into fitness and I\'m looking for a good workout routine. What works best for you?',
                'subreddit': 'fitness',
                'score': 12,
                'num_comments': 6,
                'created_utc': 1672531200,
                'url': 'https://reddit.com/r/fitness/sample3'
            },
            {
                'id': 'sample4',
                'title': 'Best meal prep ideas for busy weekdays?',
                'text': 'I work long hours and struggle with meal planning. Any simple meal prep ideas that don\'t take too much time?',
                'subreddit': 'MealPrepSunday',
                'score': 28,
                'num_comments': 15,
                'created_utc': 1672617600,
                'url': 'https://reddit.com/r/MealPrepSunday/sample4'
            }
        ],
        'comments': [
            {
                'id': 'comment3',
                'text': 'I love this recipe! I\'ve been making variations of it for months. The protein content is perfect for post-workout meals.',
                'subreddit': 'MealPrepSunday',
                'score': 15,
                'created_utc': 1672704000,
                'is_submitter': False
            },
            {
                'id': 'comment4',
                'text': 'Great progress! Keep it up! I found that consistency is key, even when motivation is low.',
                'subreddit': 'progresspics',
                'score': 3,
                'created_utc': 1672790400,
                'is_submitter': False
            }
        ],
        'total_posts': 2,
        'total_comments': 2,
        'fetch_method': 'demo'
    }
    
    analysis = {
        'text_stats': {
            'total_texts': 4,
            'total_words': 142,
            'avg_words_per_text': 35.5,
            'total_sentences': 10,
            'avg_sentences_per_text': 2.5,
            'most_common_words': [
                ('workout', 4), ('meal', 4), ('fitness', 3), 
                ('routine', 3), ('prep', 3), ('time', 2)
            ],
            'vocabulary_size': 87
        },
        'sentiment': {
            'average_sentiment': {
                'compound': 0.25,
                'positive': 0.3,
                'negative': 0.02,
                'neutral': 0.68
            },
            'overall_sentiment': 'positive',
            'sentiment_distribution': {
                'positive_posts': 3,
                'negative_posts': 0,
                'neutral_posts': 1
            }
        },
        'topics': [
            'fitness', 'workout routine', 'meal prep', 'nutrition', 
            'health', 'exercise', 'food planning'
        ],
        'interests': [
            'fitness', 'food', 'health'
        ],
        'personality_traits': {
            'helpful': 40.0,
            'optimistic': 30.0,
            'social': 20.0,
            'analytical': 10.0
        },
        'subreddit_activity': {
            'top_subreddits': [
                ('fitness', 1),
                ('MealPrepSunday', 2),
                ('progresspics', 1)
            ],
            'total_subreddits': 3,
            'post_distribution': {'fitness': 1, 'MealPrepSunday': 1},
            'comment_distribution': {'MealPrepSunday': 1, 'progresspics': 1}
        },
        'temporal_patterns': {
            'most_active_hour': 18,
            'most_active_day': 'Sunday'
        },
        'engagement_patterns': {
            'avg_post_score': 20.0,
            'max_post_score': 28,
            'total_post_karma': 40,
            'avg_comment_score': 9.0,
            'max_comment_score': 15,
            'total_comment_karma': 18,
            'content_preference': 'posts'
        }
    }
    
    return sample_data, analysis

def generate_persona_report(user_data, analysis):
    """Generate a comprehensive persona report."""
    username = user_data['username']
    
    report = f"""# Reddit User Persona: {username}
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
**AI-Generated Persona Summary:**
Based on Reddit activity analysis, this user demonstrates a strong interest in personal development and learning. They actively seek advice and share experiences in communities focused on their interests. Their communication style is positive and supportive, often expressing gratitude and encouragement to others.

**Detailed Analysis:**
User has written {analysis['text_stats']['total_texts']} posts/comments with an average of {analysis['text_stats']['avg_words_per_text']:.1f} words per post. Overall sentiment is {analysis['sentiment']['overall_sentiment']}. Main interests include: {', '.join(analysis['interests'])}. Frequently discusses: {', '.join(analysis['topics'][:5])}. Most active in subreddits: {', '.join([sub[0] for sub in analysis['subreddit_activity']['top_subreddits']])}. Personality traits suggest they are: {', '.join([trait for trait, score in sorted(analysis['personality_traits'].items(), key=lambda x: x[1], reverse=True)[:3]])}.

## Basic Statistics
- Total Posts: {user_data['total_posts']}
- Total Comments: {user_data['total_comments']}
- Data Source: {user_data['fetch_method']}

## Content Analysis
- Total Content Pieces: {analysis['text_stats']['total_texts']}
- Total Words: {analysis['text_stats']['total_words']:,}
- Average Words per Post: {analysis['text_stats']['avg_words_per_text']:.1f}
- Vocabulary Size: {analysis['text_stats']['vocabulary_size']:,}

### Most Used Words
"""
    for word, count in analysis['text_stats']['most_common_words']:
        report += f"- {word}: {count} times\n"
    
    avg_sent = analysis['sentiment']['average_sentiment']
    dist = analysis['sentiment']['sentiment_distribution']
    report += f"""
## Sentiment Analysis
- Overall Sentiment: {analysis['sentiment']['overall_sentiment'].title()}
- Positivity Score: {avg_sent['positive']:.2f}
- Negativity Score: {avg_sent['negative']:.2f}
- Neutrality Score: {avg_sent['neutral']:.2f}
- Compound Score: {avg_sent['compound']:.2f}

### Sentiment Distribution
- Positive Posts: {dist['positive_posts']}
- Negative Posts: {dist['negative_posts']}
- Neutral Posts: {dist['neutral_posts']}

## Detected Interests
{', '.join(analysis['interests'])}

## Frequently Discussed Topics
{', '.join(analysis['topics'])}

## Personality Indicators
"""
    for trait, score in sorted(analysis['personality_traits'].items(), key=lambda x: x[1], reverse=True):
        report += f"- {trait.title()}: {score:.1f}%\n"
    
    report += f"""
## Subreddit Activity
- Active in {analysis['subreddit_activity']['total_subreddits']} different subreddits

### Top Subreddits by Activity
"""
    for sub, count in analysis['subreddit_activity']['top_subreddits']:
        report += f"- r/{sub}: {count} posts/comments\n"
    
    report += f"""
## Activity Patterns
- Most Active Hour: {analysis['temporal_patterns']['most_active_hour']}:00
- Most Active Day: {analysis['temporal_patterns']['most_active_day']}

## Engagement Metrics
- Average Post Score: {analysis['engagement_patterns']['avg_post_score']:.1f}
- Highest Post Score: {analysis['engagement_patterns']['max_post_score']}
- Total Post Karma: {analysis['engagement_patterns']['total_post_karma']:,}
- Average Comment Score: {analysis['engagement_patterns']['avg_comment_score']:.1f}
- Highest Comment Score: {analysis['engagement_patterns']['max_comment_score']}
- Total Comment Karma: {analysis['engagement_patterns']['total_comment_karma']:,}
- Content Preference: {analysis['engagement_patterns']['content_preference'].title()}

## Data Citations
This persona is based on publicly available Reddit posts and comments.

### Example Posts:
"""
    for i, post in enumerate(user_data['posts']):
        report += f"{i+1}. \"{post['title']}\" (r/{post['subreddit']}, Score: {post['score']})\n"
    
    report += "\n### Example Comments:\n"
    for i, comment in enumerate(user_data['comments']):
        text_preview = comment['text'][:100] + "..." if len(comment['text']) > 100 else comment['text']
        report += f"{i+1}. \"{text_preview}\" (r/{comment['subreddit']}, Score: {comment['score']})\n"
    
    report += f"""
## Methodology
- Data collected using Demo mode (Reddit API would be used in production)
- Text analysis performed using spaCy and NLTK
- Sentiment analysis using VADER
- LLM summarization using BART (when available)

## Limitations
- Analysis based on sample data for demonstration purposes
- In production, analysis would be based on recent public activity only
- Sentiment and personality analysis are approximations
- Some content may have been deleted or removed
- Results should be interpreted as general tendencies, not definitive characteristics
"""
    
    return report

def main():
    """Create sample personas for both users."""
    
    print("Creating sample personas for demonstration...")
    
    # Create persona for kojied
    user_data_kojied, analysis_kojied = create_sample_persona_kojied()
    persona_report_kojied = generate_persona_report(user_data_kojied, analysis_kojied)
    
    # Save files for kojied
    with open('personas/kojied_persona.txt', 'w', encoding='utf-8') as f:
        f.write(persona_report_kojied)
    
    analysis_data_kojied = {
        'user_data': user_data_kojied,
        'analysis': analysis_kojied,
        'generated_at': datetime.now().isoformat()
    }
    
    with open('data/kojied_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_data_kojied, f, indent=2)
    
    # Create persona for Hungry-Move-6603
    user_data_hungry, analysis_hungry = create_sample_persona_hungry_move()
    persona_report_hungry = generate_persona_report(user_data_hungry, analysis_hungry)
    
    # Save files for Hungry-Move-6603
    with open('personas/Hungry-Move-6603_persona.txt', 'w', encoding='utf-8') as f:
        f.write(persona_report_hungry)
    
    analysis_data_hungry = {
        'user_data': user_data_hungry,
        'analysis': analysis_hungry,
        'generated_at': datetime.now().isoformat()
    }
    
    with open('data/Hungry-Move-6603_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_data_hungry, f, indent=2)
    
    print("\nSample personas created successfully!")
    print("Files generated:")
    print("  - personas/kojied_persona.txt")
    print("  - personas/Hungry-Move-6603_persona.txt")
    print("  - data/kojied_analysis.json")
    print("  - data/Hungry-Move-6603_analysis.json")
    
    print("\nNote: These are demonstration personas based on sample data.")
    print("To analyze real Reddit users, set up Reddit API credentials in .env file.")

if __name__ == "__main__":
    main()
