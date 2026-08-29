import os
import uvicorn
from backend.app import app

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8000))
    print(f'Starting SatQuery AI server on port {port}...')
    uvicorn.run(app, host='0.0.0.0', port=port)
