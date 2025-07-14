#!/usr/bin/env python3
"""
Setup and test script for the Reddit Persona Analyzer.
This script helps users set up the environment and test the analyzer.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_virtual_environment():
    """Check if running in a virtual environment."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
        return True
    else:
        print("⚠️  Not running in virtual environment (recommended but not required)")
        return True

def check_dependencies():
    """Check if all required dependencies are installed."""
    # Map package names to their import names
    package_imports = {
        'praw': 'praw',
        'requests': 'requests', 
        'beautifulsoup4': 'bs4',
        'spacy': 'spacy',
        'nltk': 'nltk',
        'transformers': 'transformers',
        'torch': 'torch',
        'python-dotenv': 'dotenv'
    }
    
    missing_packages = []
    
    for package, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_spacy_model():
    """Check if spaCy English model is installed."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy English model (en_core_web_sm)")
        return True
    except OSError:
        print("❌ spaCy English model not found")
        print("Run: python -m spacy download en_core_web_sm")
        return False

def check_nltk_data():
    """Check if required NLTK data is available."""
    try:
        import nltk
        
        # Check for required NLTK data
        required_data = ['punkt', 'stopwords', 'vader_lexicon']
        missing_data = []
        
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
                print(f"✅ NLTK {data}")
            except LookupError:
                missing_data.append(data)
                print(f"❌ NLTK {data}")
        
        if missing_data:
            print("🔄 Downloading missing NLTK data...")
            for data in missing_data:
                nltk.download(data, quiet=True)
                print(f"✅ Downloaded NLTK {data}")
        
        return True
        
    except ImportError:
        print("❌ NLTK not installed")
        return False

def check_env_file():
    """Check if .env file exists and is properly configured."""
    env_path = Path('.env')
    
    if not env_path.exists():
        print("❌ .env file not found")
        print("Copy .env.example to .env and fill in your Reddit API credentials")
        return False
    
    # Read .env file and check for required variables
    with open(env_path, 'r') as f:
        content = f.read()
    
    required_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET', 'REDDIT_USER_AGENT']
    missing_vars = []
    placeholder_vars = []
    
    for var in required_vars:
        if var not in content:
            missing_vars.append(var)
        elif f'{var}=your_' in content or f'{var}=YourUsername' in content:
            placeholder_vars.append(var)
        else:
            print(f"✅ {var}")
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
    
    if placeholder_vars:
        print(f"⚠️  Placeholder values found: {', '.join(placeholder_vars)}")
        print("Update these with your actual Reddit API credentials")
    
    if missing_vars or placeholder_vars:
        print("\n📝 To get Reddit API credentials:")
        print("1. Go to https://www.reddit.com/prefs/apps")
        print("2. Click 'Create App' or 'Create Another App'")
        print("3. Fill in the form (select 'script' type)")
        print("4. Note your Client ID and Client Secret")
        print("5. Update the .env file")
        return False
    
    return True

def check_output_directories():
    """Check if output directories exist."""
    directories = ['personas', 'data']
    
    for directory in directories:
        dir_path = Path(directory)
        if dir_path.exists():
            print(f"✅ {directory}/ directory")
        else:
            dir_path.mkdir(exist_ok=True)
            print(f"🔄 Created {directory}/ directory")
    
    return True

def run_tests():
    """Run basic functionality tests."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test imports
        import praw
        import spacy
        import nltk
        from transformers import pipeline
        print("✅ All imports successful")
        
        # Test spaCy model
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("This is a test sentence.")
        print("✅ spaCy processing works")
        
        # Test NLTK sentiment analyzer
        from nltk.sentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores("This is a positive sentence.")
        print("✅ NLTK sentiment analysis works")
        
        print("✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def create_sample_personas():
    """Create sample personas for demonstration."""
    print("\n📝 Creating sample personas...")
    
    try:
        # Run the sample persona creation script
        result = subprocess.run([
            sys.executable, 'create_sample_personas.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Sample personas created successfully")
            print("\nFiles created:")
            print("  - personas/kojied_persona.txt")
            print("  - personas/Hungry-Move-6603_persona.txt")
            print("  - data/kojied_analysis.json")
            print("  - data/Hungry-Move-6603_analysis.json")
            return True
        else:
            print(f"❌ Error creating sample personas: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating sample personas: {e}")
        return False

def main():
    """Main setup and test function."""
    print("🚀 Reddit Persona Analyzer - Setup and Test")
    print("=" * 50)
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Virtual Environment", check_virtual_environment),
        ("Dependencies", check_dependencies),
        ("spaCy Model", check_spacy_model),
        ("NLTK Data", check_nltk_data),
        ("Environment File", check_env_file),
        ("Output Directories", check_output_directories),
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n📋 Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    # Run tests if all checks passed
    if all_passed:
        run_tests()
        create_sample_personas()
        
        print("\n✅ Setup complete!")
        print("\n📚 Next steps:")
        print("1. Update .env file with your Reddit API credentials (if not done)")
        print("2. Test with a real Reddit user:")
        print("   python reddit_persona_analyzer.py username")
        print("3. View generated personas in the personas/ directory")
        
    else:
        print("\n❌ Setup incomplete. Please fix the issues above.")
        print("\n💡 Quick fixes:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Download spaCy model: python -m spacy download en_core_web_sm")
        print("- Copy .env.example to .env and add Reddit credentials")

if __name__ == "__main__":
    main()
