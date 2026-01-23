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


@views_bp.route('/about')
def about():
    """About page - Methodology and technical details"""
    return render_template('about.html', title='About SentiTrade')


@views_bp.route('/signals')
def signals():
    """Trading signals page - Full pipeline demo"""
    return render_template('signals.html', title='Trading Signals')


@views_bp.route('/ablation')
def ablation():
    """Ablation studies page - Comprehensive component analysis"""
    try:
        # Get data service from app context
        ds = current_app.data_service
        ablation_data = ds.get_ablation_studies()
        
        return render_template(
            'ablation.html',
            title='Ablation Studies',
            ablation_data=ablation_data
        )
    except Exception as e:
        current_app.logger.error(f"Error loading ablation studies: {e}")
        return render_template(
            'ablation.html',
            title='Ablation Studies',
            error=str(e)
        )
