import uvicorn
from config import config

if __name__ == "__main__":
    uvicorn.run(
        "service:app",
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        reload=config.DEBUG,
        workers=1 if config.DEBUG else 4
    )