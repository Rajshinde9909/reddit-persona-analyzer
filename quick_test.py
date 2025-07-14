#!/usr/bin/env python3
"""
Quick test script to verify the system works without Reddit API credentials.
This creates sample personas and demonstrates functionality.
"""

import sys
import subprocess

def main():
    """Run quick test without API requirements."""
    print("🚀 Quick Test - Reddit Persona Analyzer")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("📋 Testing basic functionality...")
        
        import praw
        import spacy
        import nltk
        from transformers import pipeline
        from nltk.sentiment import SentimentIntensityAnalyzer
        import bs4
        import dotenv
        
        print("✅ All imports successful")
        
        # Test spaCy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("This is a test sentence for NLP processing.")
        print("✅ spaCy processing works")
        
        # Test NLTK sentiment
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores("This is a positive test sentence.")
        print("✅ NLTK sentiment analysis works")
        
        print("\n📝 Creating demonstration personas...")
        
        # Run sample persona creation
        result = subprocess.run([
            sys.executable, 'create_sample_personas.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Sample personas created successfully!")
            
            # Show created files
            import os
            print("\n📁 Generated files:")
            
            personas_dir = "personas"
            data_dir = "data"
            
            if os.path.exists(personas_dir):
                for file in os.listdir(personas_dir):
                    if file.endswith('.txt'):
                        print(f"  📄 {personas_dir}/{file}")
            
            if os.path.exists(data_dir):
                for file in os.listdir(data_dir):
                    if file.endswith('.json'):
                        print(f"  📊 {data_dir}/{file}")
            
            print("\n✅ System is working correctly!")
            print("\n🎯 To analyze real Reddit users:")
            print("1. Get Reddit API credentials from https://www.reddit.com/prefs/apps")
            print("2. Update the .env file with your credentials")
            print("3. Run: python reddit_persona_analyzer.py username")
            
        else:
            print(f"❌ Error creating samples: {result.stderr}")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Run: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
