#!/usr/bin/env python3

import asyncio
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rabbitmq_consumer import main

if __name__ == "__main__":
    print("🚀 Starting RabbitMQ Consumer...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Consumer stopped by user")
    except Exception as e:
        print(f"❌ Consumer error: {e}")
