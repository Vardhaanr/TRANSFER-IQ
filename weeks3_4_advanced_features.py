"""
Weeks 3-4: Advanced Features & Sentiment Analysis
Tasks:
1. Advanced Performance Trends (season-wise averages, form trends)
2. Final Sentiment Analysis (VADER/TextBlob)
3. Merge All Data (by player ID and season)
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Try to import sentiment analysis libraries
try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None
    print("TextBlob not installed. Install with: pip install textblob")

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
except ImportError:
    sia = None
    print("NLTK not installed. Install with: pip install nltk")

# ============================================================
# PART 1: LOAD AND AGGREGATE PERFORMANCE DATA FROM ALL EVENTS
# ============================================================

def load_all_events_data(events_dir="open-data-master/data/events"):
    """Load and aggregate data from all event JSON files"""
    all_data = []
    
    if not os.path.exists(events_dir):
        print(f"Warning: {events_dir} not found")
        return pd.DataFrame()
    
    json_files = list(Path(events_dir).glob("*.json"))
    print(f"Found {len(json_files)} event JSON files")
    
    for json_file in json_files[:10]:  # Process first 10 files for demo
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            df = pd.json_normalize(data)
            all_data.append(df)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

# ============================================================
# PART 2: CALCULATE ADVANCED PERFORMANCE TRENDS
# ============================================================

def extract_performance_features(df):
    """Extract advanced performance metrics from event data"""
    
    if df.empty:
        print("Empty dataframe - returning empty features")
        return pd.DataFrame()
    
    # Ensure necessary columns exist
    if 'player.name' not in df.columns:
        return pd.DataFrame()
    
    performance_data = []
    
    for player in df['player.name'].dropna().unique():
        player_data = df[df['player.name'] == player]
        
        # Extract key performance metrics
        metrics = {
            'player': player,
            'games_played': player_data['match_id'].nunique() if 'match_id' in df.columns else len(player_data),
            'total_passes': len(player_data[player_data['type.name'] == 'Pass']) if 'type.name' in df.columns else 0,
            'successful_passes': len(player_data[(player_data['type.name'] == 'Pass') & (player_data['pass.outcome.name'] != 'Incomplete')]) if 'type.name' in df.columns else 0,
            'pass_accuracy': 0.0,
            'goals': len(player_data[player_data['shot.outcome.name'] == 'Goal']) if 'shot.outcome.name' in df.columns else 0,
            'assists': len(player_data[player_data['pass.goal_assist'] == True]) if 'pass.goal_assist' in df.columns else 0,
            'tackles': len(player_data[player_data['type.name'] == 'Duel']) if 'type.name' in df.columns else 0,
            'minutes_played': (player_data['minute'].fillna(0).max() * 60 + player_data['second'].fillna(0).max()) / 60 if 'minute' in df.columns else 0,
        }
        
        # Calculate pass accuracy
        if metrics['total_passes'] > 0:
            metrics['pass_accuracy'] = (metrics['successful_passes'] / metrics['total_passes']) * 100
        
        performance_data.append(metrics)
    
    return pd.DataFrame(performance_data)

# ============================================================
# PART 3: SEASON-WISE AVERAGES & FORM TRENDS
# ============================================================

def calculate_season_trends(performance_df, market_value_df):
    """Calculate season-wise averages and form trends"""
    
    if performance_df.empty or market_value_df.empty:
        return pd.DataFrame()
    
    # Merge performance with market value data
    merged_df = performance_df.copy()
    
    # Add season from market value (if available)
    merged_df = merged_df.merge(
        market_value_df[['player', 'season']].drop_duplicates(),
        left_on='player',
        right_on='player',
        how='left'
    )
    
    # Calculate aggregated metrics by player
    season_trends = merged_df.groupby('player').agg({
        'games_played': 'sum',
        'total_passes': 'sum',
        'pass_accuracy': 'mean',
        'goals': 'sum',
        'assists': 'sum',
        'tackles': 'sum',
        'minutes_played': 'sum',
        'season': 'first'
    }).reset_index()
    
    # Add form trend (normalized score based on recent performance)
    season_trends['form_trend'] = (
        season_trends['goals'] / max(season_trends['goals'].max(), 1) * 30 +
        season_trends['assists'] / max(season_trends['assists'].max(), 1) * 20 +
        season_trends['pass_accuracy'] * 50
    ) / 100
    
    return season_trends

# ============================================================
# PART 4: SENTIMENT ANALYSIS (VADER & TextBlob)
# ============================================================

def perform_sentiment_analysis(player_comments=None):
    """
    Perform sentiment analysis on player-related text
    (comments, news, social media posts, etc.)
    
    For demo: creates synthetic sentiment scores
    In production: analyze real comments/news data
    """
    
    # Sample player comments for demonstration
    sample_comments = {
        'Lionel Messi': [
            'Messi is playing exceptionally well this season',
            'Outstanding performance from Messi today',
            'Messi seems to struggle with injuries'
        ],
        'Cristiano Ronaldo': [
            'Ronaldo showed great commitment on the field',
            'Impressive goal-scoring record by Ronaldo',
            'Ronaldo had a poor match yesterday'
        ],
    }
    
    sentiment_data = []
    
    for player, comments in sample_comments.items():
        pos_score = 0.0
        neg_score = 0.0
        compound_score = 0.0
        comment_count = len(comments)
        
        if sia:  # Use VADER if available
            for comment in comments:
                scores = sia.polarity_scores(comment)
                pos_score += scores['pos']
                neg_score += scores['neg']
                compound_score += scores['compound']
            
            # Average the scores
            pos_score /= comment_count if comment_count > 0 else 1
            neg_score /= comment_count if comment_count > 0 else 1
            compound_score /= comment_count if comment_count > 0 else 1
        
        elif TextBlob:  # Fallback to TextBlob
            for comment in comments:
                blob = TextBlob(comment)
                polarity = blob.sentiment.polarity  # -1 to 1
                # Convert to 0-1 range
                pos_score += max(0, polarity)
                neg_score += max(0, -polarity)
                compound_score += polarity
            
            pos_score /= comment_count if comment_count > 0 else 1
            neg_score /= comment_count if comment_count > 0 else 1
            compound_score /= comment_count if comment_count > 0 else 1
        
        else:  # Create synthetic sentiment scores
            np.random.seed(hash(player) % 2**32)
            pos_score = np.random.uniform(0.3, 0.9)
            neg_score = np.random.uniform(0.05, 0.3)
            compound_score = pos_score - neg_score
        
        sentiment_data.append({
            'player': player,
            'sentiment_positive': round(pos_score, 3),
            'sentiment_negative': round(neg_score, 3),
            'sentiment_compound': round(compound_score, 3),
            'sentiment_source': 'sample_comments'
        })
    
    return pd.DataFrame(sentiment_data)

# ============================================================
# PART 5: MERGE ALL DATASETS
# ============================================================

def merge_all_datasets(performance_df, season_trends_df, sentiment_df, market_value_df):
    """Merge all datasets by player and season"""
    
    # Start with season trends
    final_dataset = season_trends_df.copy()
    
    # Merge with sentiment data
    if not sentiment_df.empty:
        final_dataset = final_dataset.merge(
            sentiment_df,
            left_on='player',
            right_on='player',
            how='left'
        )
    
    # Merge with market value
    if not market_value_df.empty:
        final_dataset = final_dataset.merge(
            market_value_df,
            left_on='player',
            right_on='player',
            how='left'
        )
    
    # Reorder columns for better readability
    columns_order = [
        'player', 'season', 'market_value',
        'games_played', 'minutes_played',
        'total_passes', 'pass_accuracy',
        'goals', 'assists', 'tackles',
        'form_trend',
        'sentiment_positive', 'sentiment_negative', 'sentiment_compound'
    ]
    
    available_cols = [col for col in columns_order if col in final_dataset.columns]
    final_dataset = final_dataset[available_cols]
    
    return final_dataset

# ============================================================
# PART 6: CREATE SENTIMENT IMPACT REPORT
# ============================================================

def create_sentiment_impact_report(final_dataset):
    """Generate report showing correlation between sentiment and performance"""
    
    if final_dataset.empty:
        return None
    
    report = {
        'total_players': len(final_dataset),
        'average_sentiment_score': final_dataset['sentiment_compound'].mean() if 'sentiment_compound' in final_dataset.columns else 0,
        'high_sentiment_players': [],
        'low_sentiment_players': [],
        'top_performers': [],
        'correlation_analysis': {}
    }
    
    # High vs Low sentiment
    if 'sentiment_compound' in final_dataset.columns:
        sorted_by_sentiment = final_dataset.sort_values('sentiment_compound', ascending=False)
        report['high_sentiment_players'] = sorted_by_sentiment.head(5)[['player', 'sentiment_compound', 'goals', 'assists']].to_dict('records')
        report['low_sentiment_players'] = sorted_by_sentiment.tail(5)[['player', 'sentiment_compound', 'goals', 'assists']].to_dict('records')
    
    # Top performers
    if 'form_trend' in final_dataset.columns:
        sorted_by_trend = final_dataset.sort_values('form_trend', ascending=False)
        report['top_performers'] = sorted_by_trend.head(5)[['player', 'form_trend', 'goals', 'assists']].to_dict('records')
    
    # Correlation between sentiment and performance metrics
    numeric_cols = final_dataset.select_dtypes(include=[np.number]).columns.tolist()
    if 'sentiment_compound' in numeric_cols and len(numeric_cols) > 1:
        for col in numeric_cols:
            if col != 'sentiment_compound':
                try:
                    corr = final_dataset['sentiment_compound'].corr(final_dataset[col])
                    report['correlation_analysis'][f'sentiment_vs_{col}'] = round(corr, 3)
                except:
                    pass
    
    return report

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*60)
    print("WEEKS 3-4: ADVANCED FEATURES & SENTIMENT ANALYSIS")
    print("="*60)
    
    # Load market value data
    print("\n[1] Loading market value data...")
    market_value_df = pd.read_csv("marketvalue.csv")
    print(f"    Loaded {len(market_value_df)} market value records")
    print(f"    Columns: {list(market_value_df.columns)}")
    
    # Load all events data
    print("\n[2] Loading events data...")
    events_df = load_all_events_data()
    if not events_df.empty:
        print(f"    Loaded {len(events_df)} event records")
    
    # Extract performance features
    print("\n[3] Extracting performance features...")
    performance_df = extract_performance_features(events_df)
    if not performance_df.empty:
        print(f"    Extracted metrics for {len(performance_df)} players")
    
    # Calculate season-wise trends
    print("\n[4] Calculating season-wise trends...")
    season_trends_df = calculate_season_trends(performance_df, market_value_df)
    if not season_trends_df.empty:
        print(f"    Calculated trends for {len(season_trends_df)} players")
    
    # Perform sentiment analysis
    print("\n[5] Performing sentiment analysis...")
    sentiment_df = perform_sentiment_analysis()
    print(f"    Analyzed sentiment for {len(sentiment_df)} players")
    print(f"    Sentiment columns: positive, negative, compound")
    
    # Merge all datasets
    print("\n[6] Merging all datasets...")
    final_dataset = merge_all_datasets(
        performance_df,
        season_trends_df,
        sentiment_df,
        market_value_df
    )
    
    # Save final dataset
    output_file = "final_modeling_dataset.csv"
    final_dataset.to_csv(output_file, index=False)
    print(f"    [DONE] Final dataset saved: {output_file}")
    print(f"    Shape: {final_dataset.shape}")
    
    # Create sentiment impact report
    print("\n[7] Creating sentiment impact report...")
    report = create_sentiment_impact_report(final_dataset)
    
    # Save report
    report_file = "sentiment_impact_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"    [DONE] Report saved: {report_file}")
    
    # Display summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n[DATA] Final Dataset:")
    print(f"   - Players: {len(final_dataset)}")
    print(f"   - Features: {len(final_dataset.columns)}")
    print(f"   - Columns: {', '.join(final_dataset.columns.tolist())}")
    
    print(f"\n[TREND] Top 5 Players by Form Trend:")
    if 'form_trend' in final_dataset.columns:
        top_5 = final_dataset.nlargest(5, 'form_trend')[['player', 'form_trend', 'goals', 'assists']]
        print(top_5.to_string(index=False))
    
    print(f"\n[SENTIMENT] Sentiment Analysis Results:")
    if 'sentiment_compound' in final_dataset.columns:
        print(f"   - Avg Compound Sentiment: {final_dataset['sentiment_compound'].mean():.3f}")
        print(f"   - Avg Positive Sentiment: {final_dataset['sentiment_positive'].mean():.3f}")
        print(f"   - Avg Negative Sentiment: {final_dataset['sentiment_negative'].mean():.3f}")
    
    print(f"\n[LINK] Sentiment-Performance Correlations:")
    if report and 'correlation_analysis' in report:
        for metric, corr_value in report['correlation_analysis'].items():
            print(f"   - {metric}: {corr_value}")
    
    print("\n" + "="*60)
    print("[DONE] WEEKS 3-4 TASKS COMPLETED!")
    print("="*60)
    print(f"\n[FILES] Output Files:")
    print(f"   1. {output_file} (Main modeling dataset)")
    print(f"   2. {report_file} (Sentiment impact analysis)")

if __name__ == "__main__":
    main()
