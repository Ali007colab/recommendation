#!/usr/bin/env python3

import asyncio
import json
import logging
import sys
import os
import re
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
    
    def extract_laravel_job_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """استخراج البيانات من Laravel Queue Job"""
        try:
            # التحقق من وجود هيكل Laravel Job
            if 'data' in raw_data and 'command' in raw_data['data']:
                command_string = raw_data['data']['command']
                
                # استخراج البيانات من PHP serialized object
                # البحث عن interactionData في النص
                match = re.search(r's:18:"\x00\*\x00interactionData";a:7:\{([^}]+)\}', command_string)
                if match:
                    # استخراج البيانات يدوياً من النص
                    data_section = match.group(1)
                    
                    # استخراج القيم
                    user_id_match = re.search(r's:7:"user_id";i:(\d+)', data_section)
                    coupon_id_match = re.search(r's:9:"coupon_id";i:(\d+)', data_section)
                    coupon_name_match = re.search(r's:11:"coupon_name";s:\d+:"([^"]+)"', data_section)
                    coupon_category_match = re.search(r's:15:"coupon_category";s:\d+:"([^"]+)"', data_section)
                    coupon_type_match = re.search(r's:11:"coupon_type";s:\d+:"([^"]+)"', data_section)
                    coupon_description_match = re.search(r's:18:"coupon_description";s:\d+:"([^"]+)"', data_section)
                    interaction_type_match = re.search(r's:16:"interaction_type";s:\d+:"([^"]+)"', data_section)
                    
                    # بناء البيانات المستخرجة
                    extracted_data = {}
                    
                    if user_id_match:
                        extracted_data['user_id'] = int(user_id_match.group(1))
                    
                    if coupon_id_match:
                        extracted_data['coupon_id'] = int(coupon_id_match.group(1))
                    
                    if coupon_name_match:
                        extracted_data['coupon_name'] = coupon_name_match.group(1)
                    
                    if coupon_category_match:
                        extracted_data['coupon_category'] = coupon_category_match.group(1)
                    
                    if coupon_type_match:
                        extracted_data['coupon_type'] = coupon_type_match.group(1)
                    
                    if coupon_description_match:
                        extracted_data['coupon_description'] = coupon_description_match.group(1)
                    
                    if interaction_type_match:
                        extracted_data['interaction_type'] = interaction_type_match.group(1)
                    
                    logger.info(f"📦 Extracted Laravel job data: {extracted_data}")
                    return extracted_data
            
            # إذا لم تكن Laravel job، إرجاع البيانات كما هي
            return raw_data
            
        except Exception as e:
            logger.error(f"❌ Error extracting Laravel job data: {e}")
            return raw_data
    
    async def process_interaction_message(self, message: aio_pika.IncomingMessage):
        """معالجة رسالة التفاعل"""
        async with message.process():
            try:
                # فك تشفير الرسالة
                body = message.body.decode('utf-8')
                raw_data = json.loads(body)
                
                logger.info(f"📨 Received raw message from Laravel")
                
                # استخراج البيانات من Laravel Job
                data = self.extract_laravel_job_data(raw_data)
                
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
                    logger.error(f"❌ Missing required fields. Received: user_id={user_id}, coupon_id={coupon_id}, interaction_type={interaction_type}")
                    return False
                
                logger.info(f"💾 Processing interaction: User {user_id} -> Coupon {coupon_id} ({interaction_type})")
                
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
