#!/usr/bin/env python3

import os
import sys
import json
import redis
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict
import schedule
import threading

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set cache directories before importing
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['HF_HOME'] = '/tmp/huggingface_cache'
os.environ['TORCH_HOME'] = '/tmp/torch_cache'

# Create cache directories
cache_dirs = ['/tmp/transformers_cache', '/tmp/huggingface_cache', '/tmp/torch_cache']
for cache_dir in cache_dirs:
    os.makedirs(cache_dir, exist_ok=True)

from config import config
from database import SessionLocal, Coupon, Category, CouponType, UserInteraction
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sqlalchemy import func

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/cache_recommendations.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RecommendationCacheManager:
    def __init__(self):
        self.model = None
        self.faiss_index = None
        self.coupon_ids = []
        self.vector_dim = config.VECTOR_DIM
        self.redis_client = None
        self.last_model_update = None
        self.last_cache_update = None
        
        # Initialize connections
        self.init_redis()
        self.load_model()
        
    def init_redis(self):
        """تهيئة اتصال Redis"""
        try:
            if config.REDIS_URL:
                self.redis_client = redis.from_url(config.REDIS_URL)
            else:
                self.redis_client = redis.Redis(
                    host='localhost', 
                    port=6379, 
                    db=0, 
                    decode_responses=True
                )
            
            # Test connection
            self.redis_client.ping()
            logger.info("✅ Redis connected successfully")
            
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            self.redis_client = None
    
    def load_model(self):
        """تحميل المودل"""
        try:
            logger.info("🔄 Loading sentence transformer model...")
            self.model = SentenceTransformer(config.MODEL_NAME)
            
            # Test model
            test_embedding = self.model.encode(["test sentence"])
            logger.info(f"✅ Model loaded successfully! Shape: {test_embedding.shape}")
            
            self.last_model_update = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            self.model = None
            return False
    
    def build_enhanced_text(self, coupon, category, coupon_type):
        """بناء النص المحسن للكوبون"""
        category_tokens = f"CATEGORY_{category.name} " * 25 if category else ''
        type_tokens = f"TYPE_{coupon_type.name} " * 8 if coupon_type else ''
        name_emphasis = f"TITLE_{coupon.name} " * 5
        description_reduced = f"summary_{coupon.description}"
        price_range = "expensive" if coupon.price > 100 else "affordable" if coupon.price > 20 else "cheap"
        
        return f"{category_tokens}{type_tokens}{name_emphasis} {description_reduced} {price_range}"
    
    def build_vector_store(self):
        """بناء الـ Vector Store"""
        if not self.model:
            logger.error("❌ Cannot build vector store: Model not loaded")
            return False
        
        try:
            logger.info("🔨 Building vector store...")
            
            db = SessionLocal()
            try:
                coupons = db.query(Coupon).all()
                if not coupons:
                    logger.warning("⚠️ No coupons found")
                    return False
                
                texts = []
                new_coupon_ids = []
                
                for coupon in coupons:
                    category = db.query(Category).filter(Category.id == coupon.category_id).first()
                    coupon_type = db.query(CouponType).filter(CouponType.id == coupon.coupon_type_id).first()
                    text = self.build_enhanced_text(coupon, category, coupon_type)
                    texts.append(text)
                    new_coupon_ids.append(coupon.id)
                
                logger.info(f"📊 Encoding {len(texts)} texts...")
                embeddings = self.model.encode(texts)
                
                logger.info("🏗️ Building FAISS index...")
                self.faiss_index = faiss.IndexFlatIP(self.vector_dim)
                self.faiss_index.add(embeddings.astype('float32'))
                
                self.coupon_ids = new_coupon_ids
                
                logger.info(f"✅ Vector store built with {len(coupons)} coupons")
                return True
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"❌ Error building vector store: {e}")
            return False
    
    def get_user_recommendations(self, user_id: int, top_n: int = 10):
        """الحصول على توصيات المستخدم"""
        if not self.faiss_index or not self.model:
            logger.error("❌ System not ready")
            return None
        
        db = SessionLocal()
        try:
            # Check if user exists
            interactions = db.query(UserInteraction).filter(
                UserInteraction.user_id == user_id
            ).all()
            
            if not interactions:
                # Return popular coupons for new users
                popular_coupons = db.query(
                    UserInteraction.coupon_id, 
                    func.sum(UserInteraction.score).label('total_score')
                ).group_by(UserInteraction.coupon_id)\
                 .order_by(func.sum(UserInteraction.score).desc())\
                 .limit(top_n).all()
                
                return {
                    "coupon_ids": [c.coupon_id for c in popular_coupons],
                    "method": "popular",
                    "user_categories": {},
                    "interaction_count": 0
                }
            
            # Build user profile
            weighted_emb = np.zeros(self.vector_dim)
            total_weight = 0
            user_categories = {}
            seen_coupons = {}
            
            for inter in interactions:
                coupon = db.query(Coupon).filter(Coupon.id == inter.coupon_id).first()
                if not coupon:
                    continue
                
                seen_coupons[coupon.id] = seen_coupons.get(coupon.id, 0) + inter.score
                
                category = db.query(Category).filter(Category.id == coupon.category_id).first()
                coupon_type = db.query(CouponType).filter(CouponType.id == coupon.coupon_type_id).first()
                
                text = self.build_enhanced_text(coupon, category, coupon_type)
                embedding = self.model.encode([text])[0]
                
                weight = inter.score
                if inter.action == 'purchase':
                    weight *= 2.0
                
                weighted_emb += embedding * weight
                total_weight += weight
                
                if category:
                    user_categories[category.name] = user_categories.get(category.name, 0) + inter.score
            
            if total_weight > 0:
                weighted_emb /= total_weight
            
            # Get similar coupons
            search_size = min(top_n * 10, len(self.coupon_ids))
            similarities, indices = self.faiss_index.search(
                weighted_emb.reshape(1, -1).astype('float32'), 
                search_size
            )
            
            # Smart filtering
            recommendations = []
            category_counts = {}
            max_per_category = max(2, top_n // len(user_categories)) if user_categories else top_n
            
            # Separate high and low score items
            high_score_seen = set()
            low_score_seen = set()
            
            for coupon_id, total_score in seen_coupons.items():
                if total_score >= 10.0:
                    high_score_seen.add(coupon_id)
                else:
                    low_score_seen.add(coupon_id)
            
            high_score_allowed = 0
            max_high_score = max(2, top_n // 3)
            
            for sim, idx in zip(similarities[0], indices[0]):
                if len(recommendations) >= top_n:
                    break
                
                coupon_id = self.coupon_ids[idx]
                
                # Skip low score items
                if coupon_id in low_score_seen:
                    continue
                
                # Limit high score items
                if coupon_id in high_score_seen:
                    if high_score_allowed >= max_high_score:
                        continue
                    high_score_allowed += 1
                
                # Check category diversity
                coupon = db.query(Coupon).filter(Coupon.id == coupon_id).first()
                if not coupon:
                    continue
                
                category = db.query(Category).filter(Category.id == coupon.category_id).first()
                category_name = category.name if category else "Unknown"
                
                if category_counts.get(category_name, 0) >= max_per_category:
                    continue
                
                recommendations.append(coupon_id)
                category_counts[category_name] = category_counts.get(category_name, 0) + 1
            
            # Fill remaining slots
            while len(recommendations) < top_n:
                for sim, idx in zip(similarities[0], indices[0]):
                    if len(recommendations) >= top_n:
                        break
                    
                    coupon_id = self.coupon_ids[idx]
                    if coupon_id not in recommendations and coupon_id not in seen_coupons:
                        recommendations.append(coupon_id)
            
            return {
                "coupon_ids": recommendations[:top_n],
                "method": "smart_content_based",
                "user_categories": user_categories,
                "interaction_count": len(interactions)
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations for user {user_id}: {e}")
            return None
        finally:
            db.close()
    
    def cache_user_recommendations(self, user_id: int, recommendations: Dict):
        """حفظ توصيات المستخدم في Redis"""
        if not self.redis_client:
            return False
        
        try:
            cache_key = f"rec:user:{user_id}"
            cache_data = {
                "user_id": user_id,
                "coupon_ids": recommendations["coupon_ids"],
                "method": recommendations["method"],
                "user_categories": recommendations["user_categories"],
                "interaction_count": recommendations["interaction_count"],
                "cached_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(seconds=config.RECOMMENDATIONS_CACHE_TTL)).isoformat()
            }
            
            # Store with TTL
            self.redis_client.setex(
                cache_key,
                config.RECOMMENDATIONS_CACHE_TTL,
                json.dumps(cache_data, ensure_ascii=False)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching user {user_id}: {e}")
            return False
    
    def get_active_users(self, min_interactions: int = 5, limit: int = 1000):
        """الحصول على المستخدمين النشطين"""
        db = SessionLocal()
        try:
            active_users = db.query(
                UserInteraction.user_id,
                func.count(UserInteraction.id).label('interaction_count')
            ).group_by(UserInteraction.user_id)\
             .having(func.count(UserInteraction.id) >= min_interactions)\
             .order_by(func.count(UserInteraction.id).desc())\
             .limit(limit).all()
            
            return [user.user_id for user in active_users]
            
        finally:
            db.close()
    
    def cache_all_users(self, min_interactions: int = 5, max_users: int = 1000, top_n: int = 10):
        """تخزين توصيات كل المستخدمين النشطين"""
        logger.info(f"🚀 Starting bulk caching process...")
        
        # Update model and vector store first
        if not self.build_vector_store():
            logger.error("❌ Failed to build vector store")
            return False
        
        # Get active users
        user_ids = self.get_active_users(min_interactions, max_users)
        if not user_ids:
            logger.warning("⚠️ No active users found")
            return False
        
        logger.info(f"📊 Found {len(user_ids)} active users to process")
        
        cached_count = 0
        failed_count = 0
        start_time = datetime.now()
        
        for i, user_id in enumerate(user_ids):
            try:
                # Get recommendations
                recommendations = self.get_user_recommendations(user_id, top_n)
                if not recommendations:
                    failed_count += 1
                    continue
                
                # Cache recommendations
                if self.cache_user_recommendations(user_id, recommendations):
                    cached_count += 1
                else:
                    failed_count += 1
                
                # Progress logging
                if (i + 1) % 50 == 0:
                    elapsed = datetime.now() - start_time
                    rate = (i + 1) / elapsed.total_seconds()
                    eta = timedelta(seconds=(len(user_ids) - i - 1) / rate) if rate > 0 else "Unknown"
                    
                    logger.info(f"📈 Progress: {i + 1}/{len(user_ids)} ({((i + 1)/len(user_ids)*100):.1f}%) "
                              f"- Rate: {rate:.1f} users/sec - ETA: {eta}")
                
            except Exception as e:
                logger.error(f"Error processing user {user_id}: {e}")
                failed_count += 1
                continue
        
        duration = datetime.now() - start_time
        self.last_cache_update = datetime.now()
        
        logger.info(f"✅ Bulk caching completed!")
        logger.info(f"   📊 Total users: {len(user_ids)}")
        logger.info(f"   ✅ Successfully cached: {cached_count}")
        logger.info(f"   ❌ Failed: {failed_count}")
        logger.info(f"   ⏱️ Duration: {duration}")
        logger.info(f"   🚀 Rate: {len(user_ids)/duration.total_seconds():.2f} users/sec")
        
        return True
    
    def get_cache_stats(self):
        """إحصائيات الـ Cache"""
        if not self.redis_client:
            return {"status": "redis_disconnected"}
        
        try:
            # Get all cached users
            pattern = "rec:user:*"
            keys = self.redis_client.keys(pattern)
            
            # Sample some cached data
            sample_data = []
            for key in keys[:5]:
                try:
                    data = json.loads(self.redis_client.get(key))
                    sample_data.append({
                        "user_id": data["user_id"],
                        "coupon_count": len(data["coupon_ids"]),
                        "method": data["method"],
                        "cached_at": data["cached_at"]
                    })
                except:
                    continue
            
            return {
                "status": "connected",
                "total_cached_users": len(keys),
                "last_model_update": self.last_model_update.isoformat() if self.last_model_update else None,
                "last_cache_update": self.last_cache_update.isoformat() if self.last_cache_update else None,
                "sample_cached_users": sample_data,
                "cache_ttl_hours": config.RECOMMENDATIONS_CACHE_TTL // 3600
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_scheduled_update(self):
        """تشغيل التحديث المجدول"""
        logger.info("⏰ Running scheduled cache update...")
        success = self.cache_all_users(min_interactions=3, max_users=2000, top_n=15)
        
        if success:
            logger.info("✅ Scheduled update completed successfully")
        else:
            logger.error("❌ Scheduled update failed")

def main():
    """الدالة الرئيسية"""
    logger.info("🚀 Starting Recommendation Cache Manager")
    
    # Initialize cache manager
    cache_manager = RecommendationCacheManager()
    
    # Run initial caching
    logger.info("🔄 Running initial cache population...")
    cache_manager.cache_all_users(min_interactions=3, max_users=1500, top_n=15)
    
    # Schedule regular updates
    schedule.every(2).hours.do(cache_manager.run_scheduled_update)
    
    logger.info("⏰ Scheduled updates every 2 hours")
    logger.info("🔄 Cache manager is running...")
    
    # Keep running
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("👋 Cache manager stopped by user")
    except Exception as e:
        logger.error(f"❌ Cache manager error: {e}")

if __name__ == "__main__":
    main()
