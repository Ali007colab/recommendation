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
        
        # Cache for database data to avoid repeated queries
        self.coupons_cache = {}
        self.categories_cache = {}
        self.coupon_types_cache = {}
        self.coupon_embeddings_cache = {}
        
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
    
    def load_all_data(self):
        """تحميل كل البيانات مرة واحدة لتجنب الـ queries المتكررة"""
        logger.info("📊 Loading all data into cache...")
        
        db = SessionLocal()
        try:
            # Load all coupons
            coupons = db.query(Coupon).all()
            self.coupons_cache = {c.id: c for c in coupons}
            logger.info(f"   📦 Loaded {len(coupons)} coupons")
            
            # Load all categories
            categories = db.query(Category).all()
            self.categories_cache = {c.id: c for c in categories}
            logger.info(f"   📂 Loaded {len(categories)} categories")
            
            # Load all coupon types
            coupon_types = db.query(CouponType).all()
            self.coupon_types_cache = {ct.id: ct for ct in coupon_types}
            logger.info(f"   🏷️ Loaded {len(coupon_types)} coupon types")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
        finally:
            db.close()
    
    def build_enhanced_text(self, coupon, category, coupon_type):
        """بناء النص المحسن للكوبون"""
        category_tokens = f"CATEGORY_{category.name} " * 25 if category else ''
        type_tokens = f"TYPE_{coupon_type.name} " * 8 if coupon_type else ''
        name_emphasis = f"TITLE_{coupon.name} " * 5
        description_reduced = f"summary_{coupon.description}"
        price_range = "expensive" if coupon.price > 100 else "affordable" if coupon.price > 20 else "cheap"
        
        return f"{category_tokens}{type_tokens}{name_emphasis} {description_reduced} {price_range}"
    
    def precompute_embeddings(self):
        """حساب الـ embeddings لكل الكوبونات مسبقاً"""
        logger.info("🧮 Precomputing all coupon embeddings...")
        
        self.coupon_embeddings_cache = {}
        texts = []
        coupon_ids = []
        
        for coupon_id, coupon in self.coupons_cache.items():
            category = self.categories_cache.get(coupon.category_id)
            coupon_type = self.coupon_types_cache.get(coupon.coupon_type_id)
            text = self.build_enhanced_text(coupon, category, coupon_type)
            texts.append(text)
            coupon_ids.append(coupon_id)
        
        if texts:
            logger.info(f"   🔄 Encoding {len(texts)} texts...")
            embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
            
            for coupon_id, embedding in zip(coupon_ids, embeddings):
                self.coupon_embeddings_cache[coupon_id] = embedding
            
            logger.info(f"   ✅ Precomputed {len(embeddings)} embeddings")
        
        return len(self.coupon_embeddings_cache) > 0
    
    def build_vector_store(self):
        """بناء الـ Vector Store"""
        if not self.model:
            logger.error("❌ Cannot build vector store: Model not loaded")
            return False
        
        try:
            logger.info("🔨 Building vector store...")
            
            # Load all data first
            if not self.load_all_data():
                return False
            
            # Precompute embeddings
            if not self.precompute_embeddings():
                return False
            
            if not self.coupons_cache:
                logger.warning("⚠️ No coupons found")
                return False
            
            # Build FAISS index
            embeddings = []
            new_coupon_ids = []
            
            for coupon_id, embedding in self.coupon_embeddings_cache.items():
                embeddings.append(embedding)
                new_coupon_ids.append(coupon_id)
            
            embeddings = np.array(embeddings)
            
            logger.info("🏗️ Building FAISS index...")
            self.faiss_index = faiss.IndexFlatIP(self.vector_dim)
            self.faiss_index.add(embeddings.astype('float32'))
            
            self.coupon_ids = new_coupon_ids
            
            logger.info(f"✅ Vector store built with {len(self.coupons_cache)} coupons")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error building vector store: {e}")
            return False
    
    def get_all_users_with_interactions(self):
        """الحصول على كل المستخدمين اللي عندهم تفاعلات"""
        db = SessionLocal()
        try:
            all_users = db.query(
                UserInteraction.user_id,
                func.count(UserInteraction.id).label('interaction_count')
            ).group_by(UserInteraction.user_id)\
             .order_by(func.count(UserInteraction.id).desc())\
             .all()
            
            logger.info(f"📊 Found {len(all_users)} total users with interactions")
            
            if all_users:
                max_interactions = all_users[0].interaction_count
                min_interactions = all_users[-1].interaction_count
                avg_interactions = sum(u.interaction_count for u in all_users) / len(all_users)
                
                logger.info(f"   📈 Interactions range: {min_interactions} - {max_interactions}")
                logger.info(f"   📊 Average interactions: {avg_interactions:.1f}")
            
            return [user.user_id for user in all_users]
            
        except Exception as e:
            logger.error(f"Error getting users: {e}")
            return []
        finally:
            db.close()
    
    def get_user_recommendations_fast(self, user_id: int, top_n: int = 10):
        """الحصول على توصيات المستخدم بطريقة سريعة"""
        if not self.faiss_index or not self.model:
            logger.error("❌ System not ready")
            return None
        
        db = SessionLocal()
        try:
            # Get user interactions
            interactions = db.query(UserInteraction).filter(
                UserInteraction.user_id == user_id
            ).all()
            
            if not interactions:
                logger.warning(f"⚠️ No interactions found for user {user_id}")
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
            
            # Build user profile using cached embeddings
            weighted_emb = np.zeros(self.vector_dim)
            total_weight = 0
            user_categories = {}
            seen_coupons = {}
            
            for inter in interactions:
                # Use cached data instead of queries
                coupon = self.coupons_cache.get(inter.coupon_id)
                if not coupon:
                    continue
                
                seen_coupons[coupon.id] = seen_coupons.get(coupon.id, 0) + inter.score
                
                # Use precomputed embedding
                embedding = self.coupon_embeddings_cache.get(coupon.id)
                if embedding is None:
                    continue
                
                weight = inter.score
                if inter.action == 'purchase':
                    weight *= 2.0
                
                weighted_emb += embedding * weight
                total_weight += weight
                
                # Use cached category
                category = self.categories_cache.get(coupon.category_id)
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
                
                # Check category diversity using cached data
                coupon = self.coupons_cache.get(coupon_id)
                if not coupon:
                    continue
                
                category = self.categories_cache.get(coupon.category_id)
                category_name = category.name if category else "Unknown"
                
                if category_counts.get(category_name, 0) >= max_per_category:
                    continue
                
                recommendations.append(coupon_id)
                category_counts[category_name] = category_counts.get(category_name, 0) + 1
            
            # Fill remaining slots if needed
            while len(recommendations) < top_n:
                for sim, idx in zip(similarities[0], indices[0]):
                    if len(recommendations) >= top_n:
                        break
                    
                    coupon_id = self.coupon_ids[idx]
                    if coupon_id not in recommendations and coupon_id not in seen_coupons:
                        recommendations.append(coupon_id)
                        break
                else:
                    break  # No more recommendations available
            
            return {
                "coupon_ids": recommendations[:top_n],
                "method": "smart_content_based",
                "user_categories": user_categories,
                "interaction_count": len(interactions)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting recommendations for user {user_id}: {e}")
            return None
        finally:
            db.close()
    
    def cache_user_recommendations(self, user_id: int, recommendations: Dict):
        """حفظ توصيات المستخدم في Redis مع timeout"""
        if not self.redis_client:
            logger.error(f"❌ Redis not available for user {user_id}")
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
            
            # Store with TTL and timeout
            self.redis_client.setex(
                cache_key,
                config.RECOMMENDATIONS_CACHE_TTL,
                json.dumps(cache_data, ensure_ascii=False)
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error caching user {user_id}: {e}")
            return False
    
    def cache_all_users_robust(self, top_n: int = 15):
        """تخزين توصيات كل المستخدمين مع معالجة أفضل للأخطاء"""
        logger.info(f"🚀 Starting ROBUST bulk caching process...")
        
        # Update model and vector store first
        if not self.build_vector_store():
            logger.error("❌ Failed to build vector store")
            return False
        
        # Get ALL users with interactions
        user_ids = self.get_all_users_with_interactions()
        if not user_ids:
            logger.warning("⚠️ No users found with interactions")
            return False
        
        logger.info(f"📊 Processing ALL {len(user_ids)} users...")
        
        cached_count = 0
        failed_count = 0
        start_time = datetime.now()
        
        for i, user_id in enumerate(user_ids):
            try:
                logger.info(f"🔄 Processing user {user_id} ({i+1}/{len(user_ids)})")
                
                # Add timeout for each user
                user_start_time = datetime.now()
                
                # Get recommendations with fast method
                recommendations = self.get_user_recommendations_fast(user_id, top_n)
                if not recommendations:
                    logger.warning(f"⚠️ No recommendations for user {user_id}")
                    failed_count += 1
                    continue
                
                # Cache recommendations
                if self.cache_user_recommendations(user_id, recommendations):
                    cached_count += 1
                    user_duration = datetime.now() - user_start_time
                    logger.info(f"✅ User {user_id} cached successfully in {user_duration.total_seconds():.2f}s")
                else:
                    logger.error(f"❌ Failed to cache user {user_id}")
                    failed_count += 1
                
                # Progress logging every 5 users
                if (i + 1) % 5 == 0:
                    elapsed = datetime.now() - start_time
                    rate = (i + 1) / elapsed.total_seconds()
                    eta = timedelta(seconds=(len(user_ids) - i - 1) / rate) if rate > 0 else "Unknown"
                    
                    logger.info(f"📈 Progress: {i + 1}/{len(user_ids)} ({((i + 1)/len(user_ids)*100):.1f}%) "
                              f"- Rate: {rate:.1f} users/sec - ETA: {eta}")
                    logger.info(f"   ✅ Cached: {cached_count}, ❌ Failed: {failed_count}")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Error processing user {user_id}: {e}")
                failed_count += 1
                continue
        
        duration = datetime.now() - start_time
        self.last_cache_update = datetime.now()
        
        logger.info(f"🎉 ROBUST bulk caching finished!")
        logger.info(f"   📊 Total users processed: {len(user_ids)}")
        logger.info(f"   ✅ Successfully cached: {cached_count}")
        logger.info(f"   ❌ Failed: {failed_count}")
        logger.info(f"   ⏱️ Duration: {duration}")
        logger.info(f"   🚀 Rate: {len(user_ids)/duration.total_seconds():.2f} users/sec")
        
        return cached_count > 0

def main():
    """الدالة الرئيسية"""
    logger.info("🚀 Starting ROBUST Recommendation Cache Manager")
    
    # Initialize cache manager
    cache_manager = RecommendationCacheManager()
    
    # Run robust caching for ALL users
    logger.info("🔄 Running ROBUST cache population for ALL users...")
    success = cache_manager.cache_all_users_robust(top_n=15)
    
    if success:
        logger.info("✅ Initial caching completed successfully")
    else:
        logger.error("❌ Initial caching failed")
        return
    
    # Schedule regular updates every 2 hours
    schedule.every(2).hours.do(lambda: cache_manager.cache_all_users_robust(top_n=15))
    
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