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
from collections import defaultdict
from datetime import datetime, timedelta

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

@app.get("/evaluate_system_performance")
def evaluate_system_performance(
    test_users: int = 50,
    top_k: int = 10,
    relevance_threshold: float = 0.1,
    db: Session = Depends(get_db)
):
    """تقييم شامل لأداء النظام باستخدام Precision, Recall, ومقاييس أخرى"""
    
    if faiss_index is None or model is None:
        raise HTTPException(status_code=500, detail="System not ready")
    
    print(f"🔍 Starting comprehensive system evaluation...")
    
    # الحصول على مستخدمين للاختبار
    active_users = db.query(UserInteraction.user_id).distinct().limit(test_users * 2).all()
    test_user_ids = [user[0] for user in active_users[:test_users]]
    
    if len(test_user_ids) < 10:
        raise HTTPException(status_code=400, detail="Not enough users for testing")
    
    # مقاييس التقييم
    precision_scores = []
    recall_scores = []
    f1_scores = []
    ndcg_scores = []
    diversity_scores = []
    coverage_scores = []
    
    # تفاصيل التقييم
    evaluation_details = []
    all_recommended_items = set()
    total_catalog_items = db.query(Coupon.id).count()
    
    print(f"📊 Evaluating {len(test_user_ids)} users...")
    
    for i, user_id in enumerate(test_user_ids):
        try:
            # تقسيم البيانات: 80% تدريب، 20% اختبار
            user_interactions = db.query(UserInteraction).filter(
                UserInteraction.user_id == user_id
            ).order_by(UserInteraction.timestamp).all()
            
            if len(user_interactions) < 5:  # تخطي المستخدمين بتفاعلات قليلة
                continue
            
            # تقسيم زمني: آخر 20% كاختبار
            split_point = int(len(user_interactions) * 0.8)
            train_interactions = user_interactions[:split_point]
            test_interactions = user_interactions[split_point:]
            
            if not test_interactions:
                continue
            
            # إنشاء مجموعة الاختبار (العناصر الفعلية التي تفاعل معها)
            test_items = set()
            relevant_items = set()
            
            for inter in test_interactions:
                test_items.add(inter.coupon_id)
                # اعتبار العناصر ذات النقاط العالية كـ "relevant"
                if inter.score >= 5.0:  # click أو purchase
                    relevant_items.add(inter.coupon_id)
            
            if not relevant_items:
                continue
            
            # محاكاة بيانات التدريب (استخدام فقط train_interactions)
            temp_interactions = db.query(UserInteraction).filter(
                UserInteraction.user_id == user_id,
                UserInteraction.timestamp <= train_interactions[-1].timestamp
            ).all()
            
            # الحصول على التوصيات
            recommendations_data = get_recommendations(user_id, top_k, db)
            recommended_items = set(recommendations_data["recommendations"])
            all_recommended_items.update(recommended_items)
            
            # حساب Precision و Recall
            true_positives = len(recommended_items.intersection(relevant_items))
            false_positives = len(recommended_items - relevant_items)
            false_negatives = len(relevant_items - recommended_items)
            
            precision = true_positives / len(recommended_items) if recommended_items else 0
            recall = true_positives / len(relevant_items) if relevant_items else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # حساب NDCG (Normalized Discounted Cumulative Gain)
            dcg = 0
            idcg = 0
            
            # ترتيب العناصر المثالي (حسب النقاط الفعلية)
            ideal_items = sorted(relevant_items, key=lambda x: max(
                [inter.score for inter in test_interactions if inter.coupon_id == x]
            ), reverse=True)
            
            for j, item_id in enumerate(recommendations_data["recommendations"]):
                if item_id in relevant_items:
                    # وزن العنصر حسب أعلى نقاط حصل عليها
                    relevance_score = max([inter.score for inter in test_interactions 
                                         if inter.coupon_id == item_id] + [0])
                    dcg += relevance_score / np.log2(j + 2)
            
            for j, item_id in enumerate(ideal_items[:top_k]):
                relevance_score = max([inter.score for inter in test_interactions 
                                     if inter.coupon_id == item_id] + [0])
                idcg += relevance_score / np.log2(j + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            
            # حساب التنوع (عدد الفئات المختلفة)
            recommended_categories = set()
            for item_id in recommended_items:
                coupon = db.query(Coupon).filter(Coupon.id == item_id).first()
                if coupon:
                    recommended_categories.add(coupon.category_id)
            
            diversity = len(recommended_categories) / len(recommended_items) if recommended_items else 0
            
            # حفظ النتائج
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            ndcg_scores.append(ndcg)
            diversity_scores.append(diversity)
            
            evaluation_details.append({
                'user_id': user_id,
                'total_interactions': len(user_interactions),
                'train_interactions': len(train_interactions),
                'test_interactions': len(test_interactions),
                'relevant_items': len(relevant_items),
                'recommended_items': len(recommended_items),
                'true_positives': true_positives,
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'ndcg': round(ndcg, 4),
                'diversity': round(diversity, 4)
            })
            
        except Exception as e:
            print(f"Error evaluating user {user_id}: {e}")
            continue
    
    if not precision_scores:
        raise HTTPException(status_code=400, detail="No valid evaluations completed")
    
    # حساب Coverage (تغطية الكتالوج)
    catalog_coverage = len(all_recommended_items) / total_catalog_items
    
    # حساب المتوسطات والإحصائيات
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_ndcg = np.mean(ndcg_scores)
    avg_diversity = np.mean(diversity_scores)
    
    # حساب الانحراف المعياري
    std_precision = np.std(precision_scores)
    std_recall = np.std(recall_scores)
    std_f1 = np.std(f1_scores)
    
    # تقييم الأداء العام
    overall_score = (avg_precision * 0.3 + avg_recall * 0.3 + avg_f1 * 0.2 + 
                    avg_ndcg * 0.1 + avg_diversity * 0.05 + catalog_coverage * 0.05)
    
    # تحديد مستوى الأداء
    if overall_score >= 0.8:
        performance_level = "Excellent"
    elif overall_score >= 0.6:
        performance_level = "Good"
    elif overall_score >= 0.4:
        performance_level = "Fair"
    else:
        performance_level = "Poor"
    
    # إحصائيات إضافية
    precision_distribution = {
        'excellent': len([p for p in precision_scores if p >= 0.8]),
        'good': len([p for p in precision_scores if 0.6 <= p < 0.8]),
        'fair': len([p for p in precision_scores if 0.4 <= p < 0.6]),
        'poor': len([p for p in precision_scores if p < 0.4])
    }
    
    recall_distribution = {
        'excellent': len([r for r in recall_scores if r >= 0.8]),
        'good': len([r for r in recall_scores if 0.6 <= r < 0.8]),
        'fair': len([r for r in recall_scores if 0.4 <= r < 0.6]),
        'poor': len([r for r in recall_scores if r < 0.4])
    }
    
    return {
        "evaluation_summary": {
            "overall_performance_score": round(overall_score, 4),
            "performance_level": performance_level,
            "users_evaluated": len(evaluation_details),
            "evaluation_timestamp": datetime.now().isoformat()
        },
        "core_metrics": {
            "precision": {
                "average": round(avg_precision, 4),
                "std_deviation": round(std_precision, 4),
                "min": round(min(precision_scores), 4),
                "max": round(max(precision_scores), 4),
                "distribution": precision_distribution
            },
            "recall": {
                "average": round(avg_recall, 4),
                "std_deviation": round(std_recall, 4),
                "min": round(min(recall_scores), 4),
                "max": round(max(recall_scores), 4),
                "distribution": recall_distribution
            },
            "f1_score": {
                "average": round(avg_f1, 4),
                "std_deviation": round(std_f1, 4),
                "min": round(min(f1_scores), 4),
                "max": round(max(f1_scores), 4)
            }
        },
        "advanced_metrics": {
            "ndcg": {
                "average": round(avg_ndcg, 4),
                "description": "Normalized Discounted Cumulative Gain"
            },
            "diversity": {
                "average": round(avg_diversity, 4),
                "description": "Category diversity in recommendations"
            },
            "catalog_coverage": {
                "score": round(catalog_coverage, 4),
                "items_recommended": len(all_recommended_items),
                "total_catalog_items": total_catalog_items,
                "description": "Percentage of catalog items recommended"
            }
        },
        "benchmarking": {
            "industry_comparison": {
                "precision": "Industry avg: 0.15-0.25, Your system: " + str(round(avg_precision, 4)),
                "recall": "Industry avg: 0.10-0.20, Your system: " + str(round(avg_recall, 4)),
                "f1_score": "Industry avg: 0.12-0.22, Your system: " + str(round(avg_f1, 4))
            },
            "system_strengths": [],
            "improvement_areas": []
        },
        "detailed_results": evaluation_details[:10],  # أول 10 مستخدمين للعرض
        "recommendations_for_improvement": []
    }

@app.get("/quick_performance_check")
def quick_performance_check(db: Session = Depends(get_db)):
    """فحص سريع لأداء النظام"""
    
    # إحصائيات سريعة
    total_users = db.query(UserInteraction.user_id).distinct().count()
    total_interactions = db.query(UserInteraction).count()
    total_coupons = db.query(Coupon).count()
    
    # توزيع التفاعلات
    action_distribution = db.query(
        UserInteraction.action, 
        func.count(UserInteraction.id)
    ).group_by(UserInteraction.action).all()
    
    # أكثر الفئات نشاطاً
    popular_categories = db.query(
        Category.name,
        func.count(UserInteraction.id).label('interactions')
    ).join(Coupon, Category.id == Coupon.category_id)\
     .join(UserInteraction, Coupon.id == UserInteraction.coupon_id)\
     .group_by(Category.name)\
     .order_by(func.count(UserInteraction.id).desc())\
     .limit(10).all()
    
    # حساب معدل التحويل
    searches = db.query(UserInteraction).filter(UserInteraction.action == 'search').count()
    clicks = db.query(UserInteraction).filter(UserInteraction.action == 'click').count()
    purchases = db.query(UserInteraction).filter(UserInteraction.action == 'purchase').count()
    
    conversion_rates = {
        'search_to_click': round((clicks / searches * 100), 2) if searches > 0 else 0,
        'click_to_purchase': round((purchases / clicks * 100), 2) if clicks > 0 else 0,
        'overall_conversion': round((purchases / searches * 100), 2) if searches > 0 else 0
    }
    
    return {
        "system_health": {
            "total_users": total_users,
            "total_interactions": total_interactions,
            "total_coupons": total_coupons,
            "avg_interactions_per_user": round(total_interactions / total_users, 2) if total_users > 0 else 0
        },
        "interaction_distribution": {action[0]: action[1] for action in action_distribution},
        "conversion_rates": conversion_rates,
        "popular_categories": [{"category": cat[0], "interactions": cat[1]} for cat in popular_categories],
        "system_status": "Healthy" if total_users > 10 and total_interactions > 100 else "Needs More Data"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        log_level="info"
    )