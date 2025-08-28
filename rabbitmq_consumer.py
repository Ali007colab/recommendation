#!/usr/bin/env python3

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any

import aio_pika
from aio_pika import connect_robust, Message
from sqlalchemy.orm import Session

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from database import get_db_session, Coupon, Category, CouponType, UserInteraction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RabbitMQConsumer:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.exchange = None
        self.queue = None
        
    async def connect(self):
        """الاتصال بـ RabbitMQ"""
        try:
            # إنشاء الاتصال
            self.connection = await connect_robust(
                host=config.RABBITMQ_HOST,
                port=config.RABBITMQ_PORT,
                login=config.RABBITMQ_USER,
                password=config.RABBITMQ_PASSWORD,
                virtualhost=config.RABBITMQ_VHOST
            )
            
            # إنشاء القناة
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=10)
            
            # إنشاء Exchange
            self.exchange = await self.channel.declare_exchange(
                'interaction_exchange',
                aio_pika.ExchangeType.TOPIC,
                durable=True
            )
            
            # إنشاء Queue
            self.queue = await self.channel.declare_queue(
                'interaction_queue',
                durable=True
            )
            
            # ربط Queue بـ Exchange
            await self.queue.bind(self.exchange, 'user.interaction')
            
            logger.info("✅ Connected to RabbitMQ successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to RabbitMQ: {e}")
            return False
    
    async def process_interaction_message(self, message: aio_pika.IncomingMessage):
        """معالجة رسالة التفاعل"""
        async with message.process():
            try:
                # فك تشفير الرسالة
                body = message.body.decode('utf-8')
                data = json.loads(body)
                
                logger.info(f"📨 Received interaction: {data}")
                
                # معالجة البيانات وحفظها
                success = await self.save_interaction_data(data)
                
                if success:
                    logger.info(f"✅ Successfully processed interaction for user {data.get('user_id')}")
                else:
                    logger.error(f"❌ Failed to process interaction for user {data.get('user_id')}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"❌ Invalid JSON in message: {e}")
            except Exception as e:
                logger.error(f"❌ Error processing message: {e}")
    
    async def save_interaction_data(self, data: Dict[str, Any]) -> bool:
        """حفظ بيانات التفاعل في قاعدة البيانات"""
        try:
            # إنشاء جلسة قاعدة البيانات
            db = next(get_db_session())
            
            try:
                # استخراج البيانات
                user_id = data.get('user_id')
                coupon_id = data.get('coupon_id')
                coupon_name = data.get('coupon_name')
                coupon_category = data.get('coupon_category')
                coupon_type = data.get('coupon_type')
                coupon_description = data.get('coupon_description')
                interaction_type = data.get('interaction_type')
                
                # التحقق من البيانات المطلوبة
                if not all([user_id, coupon_id, interaction_type]):
                    logger.error("❌ Missing required fields: user_id, coupon_id, or interaction_type")
                    return False
                
                # البحث عن الفئة أو إنشاؤها
                category = None
                if coupon_category:
                    category = db.query(Category).filter(Category.name == coupon_category).first()
                    if not category:
                        category = Category(name=coupon_category)
                        db.add(category)
                        db.flush()  # للحصول على ID
                        logger.info(f"➕ Created new category: {coupon_category}")
                
                # البحث عن نوع الكوبون أو إنشاؤه
                coupon_type_obj = None
                if coupon_type:
                    coupon_type_obj = db.query(CouponType).filter(CouponType.name == coupon_type).first()
                    if not coupon_type_obj:
                        coupon_type_obj = CouponType(
                            name=coupon_type,
                            description=f"Auto-created type: {coupon_type}"
                        )
                        db.add(coupon_type_obj)
                        db.flush()  # للحصول على ID
                        logger.info(f"➕ Created new coupon type: {coupon_type}")
                
                # البحث عن الكوبون أو إنشاؤه
                coupon = db.query(Coupon).filter(Coupon.id == coupon_id).first()
                if not coupon and coupon_name:
                    coupon = Coupon(
                        id=coupon_id,
                        name=coupon_name,
                        description=coupon_description or "Auto-created from interaction",
                        price=0.0,  # سعر افتراضي
                        coupon_type_id=coupon_type_obj.id if coupon_type_obj else None,
                        category_id=category.id if category else None,
                        provider_id=1,  # معرف افتراضي للمزود
                        coupon_status=1,
                        coupon_code=f"AUTO_{coupon_id}",
                        date=datetime.now().date()
                    )
                    db.add(coupon)
                    db.flush()
                    logger.info(f"➕ Created new coupon: {coupon_name}")
                
                # حساب النقاط بناءً على نوع التفاعل
                score_mapping = {
                    'search': 2.0,
                    'click': 5.0,
                    'purchase': 15.0
                }
                score = score_mapping.get(interaction_type, 1.0)
                
                # إنشاء سجل التفاعل
                interaction = UserInteraction(
                    user_id=user_id,
                    coupon_id=coupon_id,
                    action=interaction_type,
                    score=score,
                    timestamp=datetime.now()
                )
                
                db.add(interaction)
                db.commit()
                
                logger.info(f"✅ Saved interaction: User {user_id} -> Coupon {coupon_id} ({interaction_type}, score: {score})")
                return True
                
            except Exception as e:
                db.rollback()
                logger.error(f"❌ Database error: {e}")
                return False
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"❌ Error in save_interaction_data: {e}")
            return False
    
    async def start_consuming(self):
        """بدء استقبال الرسائل"""
        if not self.queue:
            logger.error("❌ Queue not initialized. Call connect() first.")
            return
        
        logger.info("🎧 Starting to consume messages...")
        
        # بدء استقبال الرسائل
        await self.queue.consume(self.process_interaction_message)
        
        try:
            # الانتظار إلى ما لا نهاية
            await asyncio.Future()
        except KeyboardInterrupt:
            logger.info("🛑 Stopping consumer...")
        finally:
            await self.close()
    
    async def close(self):
        """إغلاق الاتصالات"""
        if self.connection:
            await self.connection.close()
            logger.info("🔌 RabbitMQ connection closed")

# دالة مساعدة لإنشاء جلسة قاعدة البيانات
def get_db_session():
    """إنشاء جلسة قاعدة بيانات جديدة"""
    from database import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def main():
    """الدالة الرئيسية"""
    logger.info("🚀 Starting RabbitMQ Consumer Service...")
    
    consumer = RabbitMQConsumer()
    
    # محاولة الاتصال
    connected = await consumer.connect()
    if not connected:
        logger.error("❌ Failed to connect to RabbitMQ. Exiting...")
        return
    
    # بدء الاستقبال
    await consumer.start_consuming()

if __name__ == "__main__":
    asyncio.run(main())
