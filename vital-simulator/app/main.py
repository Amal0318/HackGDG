"""
Vital Simulator Service - ICU Telemetry Data Generator
Phase 0: Minimal service startup without business logic
"""

import asyncio
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vital-simulator")

class VitalSimulator:
    """Minimal vital signs simulator service"""
    
    def __init__(self):
        self.kafka_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:29092')
        logger.info(f"Initialized with Kafka servers: {self.kafka_servers}")
    
    async def start(self):
        """Start the vital simulator service"""
        logger.info("Vital Simulator Service started successfully")
        logger.info("Service is ready to generate ICU telemetry data")
        
        # Keep service running
        while True:
            await asyncio.sleep(10)
            logger.info("Vital Simulator Service is running...")

async def main():
    """Main entry point"""
    simulator = VitalSimulator()
    await simulator.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Vital Simulator Service stopped")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        raise