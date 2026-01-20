"""
Run script for development
Quick start: python run.py
"""
from app import create_app

if __name__ == '__main__':
    app = create_app()
    
    print("\n" + "="*60)
    print("🚀 SentiTrade Flask Application Starting...")
    print("="*60)
    print("\n📍 Access the application at:")
    print("   http://localhost:5000")
    print("\n📋 Available pages:")
    print("   • Home:    http://localhost:5000/")
    print("   • Analyze: http://localhost:5000/analyze")
    print("   • Results: http://localhost:5000/results")
    print("   • About:   http://localhost:5000/about")
    print("\n🔌 API endpoints:")
    print("   • POST /api/analyze")
    print("   • GET  /api/metrics")
    print("   • GET  /api/examples")
    print("\n⏹  Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
