#!/usr/bin/env python3

import logging
import sys
import os
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from database import get_db, create_tables, Coupon, Category, CouponType, UserInteraction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Recommendation Service", version="1.0.0")

model = None
faiss_index = None
coupon_ids = []
vector_dim = config.VECTOR_DIM

def build_enhanced_text(coupon, category, coupon_type):
    category_tokens = f"CATEGORY_{category.name} " * 25 if category else ''
    type_tokens = f"TYPE_{coupon_type.name} " * 8 if coupon_type else ''
    name_emphasis = f"TITLE_{coupon.name} " * 5
    description_reduced = f"summary_{coupon.description}"
    price_range = "expensive" if coupon.price > 100 else "affordable" if coupon.price > 20 else "cheap"
    
    return f"{category_tokens}{type_tokens}{name_emphasis} {description_reduced} {price_range}"

@app.on_event("startup")
async def startup_event():
    global model
    logger.info("🚀 Starting Recommendation Service...")
    
    try:
        create_tables()
        logger.info("✅ Database ready")
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
    
    try:
        logger.info("📥 Loading sentence transformer model...")
        model = SentenceTransformer(config.MODEL_NAME)
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
    
    logger.info("✅ Service started")

@app.get("/")
def root():
    return {
        "service": "Recommendation Service",
        "version": "1.0.0",
        "status": "🟢 running",
        "model_loaded": "✅" if model else "❌",
        "vector_store_built": "✅" if faiss_index else "❌"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "model_status": "loaded" if model else "not_loaded",
        "vector_store_status": "built" if faiss_index else "not_built"
    }

@app.post("/build_vector_store")
def build_vector_store(db: Session = Depends(get_db)):
    global faiss_index, coupon_ids
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    logger.info("🔨 Building vector store...")
    
    coupons = db.query(Coupon).all()
    if not coupons:
        raise HTTPException(status_code=404, detail="No coupons found")
    
    texts = []
    coupon_ids = []
    
    for coupon in coupons:
        category = db.query(Category).filter(Category.id == coupon.category_id).first()
        coupon_type = db.query(CouponType).filter(CouponType.id == coupon.coupon_type_id).first()
        text = build_enhanced_text(coupon, category, coupon_type)
        texts.append(text)
        coupon_ids.append(coupon.id)
    
    logger.info(f"📊 Encoding {len(texts)} texts...")
    embeddings = model.encode(texts)
    
    logger.info("🏗️ Building FAISS index...")
    faiss_index = faiss.IndexFlatIP(vector_dim)
    faiss_index.add(embeddings.astype('float32'))
    
    logger.info(f"✅ Vector store built with {len(coupons)} coupons")
    return {"message": f"Vector store built with {len(coupons)} coupons"}

@app.post("/log_event")
def log_event(user_id: int, coupon_id: int, action: str, db: Session = Depends(get_db)):
    score = {'search': 2.0, 'click': 5.0, 'purchase': 15.0}.get(action, 1.0)
    
    interaction = UserInteraction(
        user_id=user_id,
        coupon_id=coupon_id,
        action=action,
        score=score
    )
    
    db.add(interaction)
    db.commit()
    
    return {"message": "Event logged successfully", "score": score}

@app.get("/get_recommendations")
def get_recommendations(user_id: int, top_n: int = 10, db: Session = Depends(get_db)):
    if faiss_index is None:
        raise HTTPException(status_code=500, detail="Vector store not built")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    interactions = db.query(UserInteraction).filter(UserInteraction.user_id == user_id).all()
    
    if not interactions:
        popular_coupons = db.query(UserInteraction.coupon_id, func.sum(UserInteraction.score).label('total_score'))\
                           .group_by(UserInteraction.coupon_id)\
                           .order_by(func.sum(UserInteraction.score).desc())\
                           .limit(top_n).all()
        return {"recommendations": [c.coupon_id for c in popular_coupons], "method": "popular"}
    
    weighted_emb = np.zeros(vector_dim)
    total_weight = 0
    user_categories = {}
    seen_coupons = set()
    
    for inter in interactions:
        coupon = db.query(Coupon).filter(Coupon.id == inter.coupon_id).first()
        if not coupon:
            continue
            
        seen_coupons.add(coupon.id)
        category = db.query(Category).filter(Category.id == coupon.category_id).first()
        coupon_type = db.query(CouponType).filter(CouponType.id == coupon.coupon_type_id).first()
        
        text = build_enhanced_text(coupon, category, coupon_type)
        embedding = model.encode([text])[0]
        
        weight = inter.score
        if inter.action == 'purchase':
            weight *= 2.0
        
        weighted_emb += embedding * weight
        total_weight += weight
        
        if category:
            user_categories[category.name] = user_categories.get(category.name, 0) + inter.score
    
    if total_weight > 0:
        weighted_emb /= total_weight
    
    similarities, indices = faiss_index.search(weighted_emb.reshape(1, -1).astype('float32'), min(top_n * 3, len(coupon_ids)))
    
    recommendations = []
    category_counts = {}
    max_per_category = max(2, top_n // len(user_categories)) if user_categories else top_n
    
    for sim, idx in zip(similarities[0], indices[0]):
        if len(recommendations) >= top_n:
            break
            
        coupon_id = coupon_ids[idx]
        if coupon_id in seen_coupons:
            continue
            
        coupon = db.query(Coupon).filter(Coupon.id == coupon_id).first()
        if not coupon:
            continue
            
        category = db.query(Category).filter(Category.id == coupon.category_id).first()
        category_name = category.name if category else "Unknown"
        
        if category_counts.get(category_name, 0) >= max_per_category:
            continue
            
        recommendations.append(coupon_id)
        category_counts[category_name] = category_counts.get(category_name, 0) + 1
    
    while len(recommendations) < top_n:
        for sim, idx in zip(similarities[0], indices[0]):
            if len(recommendations) >= top_n:
                break
                
            coupon_id = coupon_ids[idx]
            if coupon_id not in seen_coupons and coupon_id not in recommendations:
                recommendations.append(coupon_id)
    
    return {
        "recommendations": recommendations[:top_n],
        "method": "content_based",
        "user_categories": user_categories
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        log_level="info"
    )