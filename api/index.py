"""
Vercel Serverless Function Entry Point
NOTE: This is NOT recommended for ML models due to size/timeout limitations
"""
from app import create_app

app = create_app()

# Vercel expects a handler function
def handler(request):
    return app(request.environ, request.start_response)

# For Vercel Python runtime
if __name__ == '__main__':
    app.run()

