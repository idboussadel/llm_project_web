"""
WSGI Entry Point for Production Deployment
"""
from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run()
