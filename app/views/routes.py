"""
View Routes Blueprint - HTML page rendering
"""
from flask import Blueprint, render_template, current_app
from app.services.data_service import DataService

views_bp = Blueprint('views', __name__)


@views_bp.route('/')
def index():
    """Landing page - Project overview"""
    return render_template('index.html', title='SentiTrade - Home')


@views_bp.route('/analyze')
def analyze():
    """Sentiment analysis demo page"""
    examples = DataService.get_example_texts()
    return render_template('analyze.html', title='Analyze Sentiment', examples=examples)


@views_bp.route('/results')
def results():
    """Results dashboard - Model metrics and performance"""
    try:
        # Get data service from app context
        ds = current_app.data_service
        # Load all metrics
        backtesting = ds.get_final_results()
        sentiment_metrics = ds.get_sentiment_metrics()
        tft_metrics = ds.get_tft_metrics()
        
        return render_template(
            'results.html',
            title='Results & Metrics',
            backtesting=backtesting,
            sentiment_metrics=sentiment_metrics,
            tft_metrics=tft_metrics
        )
    except Exception as e:
        current_app.logger.error(f"Error loading results: {e}")
        return render_template(
            'results.html',
            title='Results & Metrics',
            error=str(e)
        )


@views_bp.route('/about')
def about():
    """About page - Methodology and technical details"""
    return render_template('about.html', title='About SentiTrade')


@views_bp.route('/signals')
def signals():
    """Trading signals page - Full pipeline demo"""
    return render_template('signals.html', title='Trading Signals')
