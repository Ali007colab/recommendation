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

@app.get("/evaluate_system_performance_fixed")
def evaluate_system_performance_fixed(
    test_users: int = 50,
    top_k: int = 10,
    db: Session = Depends(get_db)
):
    """تقييم مُصحح للنظام - يحل مشكلة الـ Zero Overlap"""
    
    if faiss_index is None or model is None:
        raise HTTPException(status_code=500, detail="System not ready")
    
    print(f"🔍 Starting FIXED evaluation with {test_users} users...")
    
    # الحصول على مستخدمين نشطين
    active_users = db.query(
        UserInteraction.user_id,
        func.count(UserInteraction.id).label('interaction_count')
    ).group_by(UserInteraction.user_id)\
     .having(func.count(UserInteraction.id) >= 10)\
     .order_by(func.count(UserInteraction.id).desc())\
     .limit(test_users).all()
    
    if len(active_users) < 5:
        raise HTTPException(status_code=400, detail="Not enough active users")
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    ndcg_scores = []
    evaluation_details = []
    
    for i, (user_id, interaction_count) in enumerate(active_users):
        try:
            print(f"  📊 Evaluating user {user_id} ({i+1}/{len(active_users)})...")
            
            # الحصول على كل تفاعلات المستخدم مرتبة زمنياً
            all_interactions = db.query(UserInteraction).filter(
                UserInteraction.user_id == user_id
            ).order_by(UserInteraction.timestamp).all()
            
            if len(all_interactions) < 10:
                continue
            
            # تقسيم زمني: 80% تدريب، 20% اختبار
            split_point = int(len(all_interactions) * 0.8)
            train_interactions = all_interactions[:split_point]
            test_interactions = all_interactions[split_point:]
            
            if len(test_interactions) < 2:
                continue
            
            # تحديد العناصر المناسبة في مجموعة الاختبار
            relevant_items = set()
            test_scores = {}
            
            for inter in test_interactions:
                if inter.score >= 5.0:  # click أو purchase
                    relevant_items.add(inter.coupon_id)
                    test_scores[inter.coupon_id] = inter.score
            
            if not relevant_items:
                continue
            
            # **الحل الرئيسي:** إنشاء نسخة مؤقتة من التفاعلات للتدريب فقط
            
            # حذف مؤقت لتفاعلات الاختبار من قاعدة البيانات
            test_interaction_ids = [inter.id for inter in test_interactions]
            
            # حفظ تفاعلات الاختبار
            test_data = []
            for inter in test_interactions:
                test_data.append({
                    'user_id': inter.user_id,
                    'coupon_id': inter.coupon_id,
                    'action': inter.action,
                    'score': inter.score,
                    'timestamp': inter.timestamp
                })
            
            # حذف تفاعلات الاختبار مؤقتاً
            db.query(UserInteraction).filter(
                UserInteraction.id.in_(test_interaction_ids)
            ).delete(synchronize_session=False)
            db.commit()
            
            # الحصول على التوصيات بناءً على بيانات التدريب فقط
            try:
                recommendations_data = get_recommendations(user_id, top_k, db)
                recommended_items = set(recommendations_data["recommendations"])
            except Exception as e:
                print(f"Error getting recommendations for user {user_id}: {e}")
                recommended_items = set()
            
            # إعادة إدراج تفاعلات الاختبار
            for data in test_data:
                new_interaction = UserInteraction(
                    user_id=data['user_id'],
                    coupon_id=data['coupon_id'],
                    action=data['action'],
                    score=data['score'],
                    timestamp=data['timestamp']
                )
                db.add(new_interaction)
            db.commit()
            
            # حساب المقاييس
            true_positives = len(recommended_items.intersection(relevant_items))
            false_positives = len(recommended_items - relevant_items)
            false_negatives = len(relevant_items - recommended_items)
            
            precision = true_positives / len(recommended_items) if recommended_items else 0
            recall = true_positives / len(relevant_items) if relevant_items else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # حساب NDCG
            dcg = 0
            idcg = 0
            
            # DCG للتوصيات الفعلية
            for j, item_id in enumerate(recommendations_data["recommendations"]):
                if item_id in relevant_items:
                    relevance_score = test_scores.get(item_id, 0) / 15.0  # normalize to 0-1
                    dcg += relevance_score / np.log2(j + 2)
            
            # IDCG للترتيب المثالي
            sorted_relevant = sorted(relevant_items, 
                                   key=lambda x: test_scores.get(x, 0), 
                                   reverse=True)
            
            for j, item_id in enumerate(sorted_relevant[:top_k]):
                relevance_score = test_scores.get(item_id, 0) / 15.0
                idcg += relevance_score / np.log2(j + 2)
            
            ndcg = dcg / idcg if idcg > 0 else 0
            
            # حفظ النتائج
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            ndcg_scores.append(ndcg)
            
            evaluation_details.append({
                'user_id': user_id,
                'total_interactions': len(all_interactions),
                'train_interactions': len(train_interactions),
                'test_interactions': len(test_interactions),
                'relevant_items': len(relevant_items),
                'recommended_items': len(recommended_items),
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'ndcg': round(ndcg, 4),
                'overlap_items': list(recommended_items.intersection(relevant_items)),
                'missed_relevant': list(relevant_items - recommended_items)[:3]
            })
            
        except Exception as e:
            print(f"❌ Error evaluating user {user_id}: {e}")
            continue
    
    if not precision_scores:
        raise HTTPException(status_code=400, detail="No valid evaluations completed")
    
    # حساب الإحصائيات
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_ndcg = np.mean(ndcg_scores)
    
    # تحديد مستوى الأداء
    if avg_f1 >= 0.25:
        performance_level = "Excellent"
    elif avg_f1 >= 0.15:
        performance_level = "Good"
    elif avg_f1 >= 0.08:
        performance_level = "Fair"
    else:
        performance_level = "Poor"
    
    # توزيع الأداء
    precision_distribution = {
        'excellent': len([p for p in precision_scores if p >= 0.3]),
        'good': len([p for p in precision_scores if 0.15 <= p < 0.3]),
        'fair': len([p for p in precision_scores if 0.05 <= p < 0.15]),
        'poor': len([p for p in precision_scores if p < 0.05])
    }
    
    return {
        "fixed_evaluation_summary": {
            "performance_level": performance_level,
            "users_evaluated": len(evaluation_details),
            "evaluation_method": "Temporal Split (80% train, 20% test)",
            "evaluation_timestamp": datetime.now().isoformat()
        },
        "core_metrics": {
            "precision": {
                "average": round(avg_precision, 4),
                "std_deviation": round(np.std(precision_scores), 4),
                "min": round(min(precision_scores), 4),
                "max": round(max(precision_scores), 4),
                "distribution": precision_distribution
            },
            "recall": {
                "average": round(avg_recall, 4),
                "std_deviation": round(np.std(recall_scores), 4),
                "min": round(min(recall_scores), 4),
                "max": round(max(recall_scores), 4)
            },
            "f1_score": {
                "average": round(avg_f1, 4),
                "std_deviation": round(np.std(f1_scores), 4),
                "min": round(min(f1_scores), 4),
                "max": round(max(f1_scores), 4)
            },
            "ndcg": {
                "average": round(avg_ndcg, 4),
                "description": "Normalized Discounted Cumulative Gain"
            }
        },
        "improvement_analysis": {
            "previous_f1": 0.0000,
            "current_f1": round(avg_f1, 4),
            "improvement": f"{((avg_f1 - 0.0) / 0.01 * 100):.1f}% improvement" if avg_f1 > 0 else "No improvement",
            "users_with_good_performance": len([f for f in f1_scores if f >= 0.15])
        },
        "detailed_results": evaluation_details[:10],
        "recommendations_for_improvement": [
            "Consider collaborative filtering for cold start users",
            "Implement popularity-based fallback",
            "Add more sophisticated feature engineering",
            "Consider ensemble methods"
        ] if avg_f1 < 0.15 else [
            "System performing well",
            "Consider A/B testing for further optimization",
            "Monitor performance over time"
        ]
    }

@app.get("/simple_evaluation")
def simple_evaluation(test_users: int = 30, db: Session = Depends(get_db)):
    """تقييم بسيط وسريع للنظام"""
    
    print(f"⚡ Simple evaluation with {test_users} users...")
    
    # الحصول على مستخدمين نشطين
    active_users = db.query(
        UserInteraction.user_id,
        func.count(UserInteraction.id).label('interaction_count')
    ).group_by(UserInteraction.user_id)\
     .having(func.count(UserInteraction.id) >= 5)\
     .order_by(func.random())\
     .limit(test_users).all()
    
    results = []
    precision_scores = []
    recall_scores = []
    
    for user_id, interaction_count in active_users:
        try:
            # الحصول على تفاعلات المستخدم
            user_interactions = db.query(UserInteraction).filter(
                UserInteraction.user_id == user_id
            ).all()
            
            # العناصر التي أحبها المستخدم (purchases فقط)
            liked_items = set()
            for inter in user_interactions:
                if inter.action == 'purchase':  # فقط المشتريات
                    liked_items.add(inter.coupon_id)
            
            if not liked_items:
                continue
            
            # الحصول على التوصيات
            recommendations_data = get_recommendations(user_id, 10, db)
            recommended_items = set(recommendations_data["recommendations"])
            
            if not recommended_items:
                continue
            
            # حساب التطابق
            matches = recommended_items.intersection(liked_items)
            
            # حساب المقاييس
            precision = len(matches) / len(recommended_items)
            recall = len(matches) / len(liked_items)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            results.append({
                'user_id': user_id,
                'total_interactions': interaction_count,
                'liked_items_count': len(liked_items),
                'recommended_items_count': len(recommended_items),
                'matches': len(matches),
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1_score': round(f1, 4),
                'matching_items': list(matches)[:3],
                'user_top_categories': dict(list(recommendations_data.get("user_categories", {}).items())[:3])
            })
            
        except Exception as e:
            print(f"Error with user {user_id}: {e}")
            continue
    
    if not results:
        return {"error": "No valid users for evaluation"}
    
    # حساب المتوسطات
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_recall = np.mean(recall_scores) if recall_scores else 0
    avg_f1 = np.mean([r['f1_score'] for r in results]) if results else 0
    
    # تقييم الأداء
    if avg_f1 >= 0.20:
        performance = "Good"
    elif avg_f1 >= 0.10:
        performance = "Fair"
    else:
        performance = "Needs Improvement"
    
    return {
        "simple_evaluation_summary": {
            "performance_level": performance,
            "users_evaluated": len(results),
            "evaluation_method": "Purchase-based relevance",
            "average_precision": round(avg_precision, 4),
            "average_recall": round(avg_recall, 4),
            "average_f1_score": round(avg_f1, 4)
        },
        "performance_distribution": {
            "good_users": len([r for r in results if r['f1_score'] >= 0.2]),
            "fair_users": len([r for r in results if 0.1 <= r['f1_score'] < 0.2]),
            "poor_users": len([r for r in results if r['f1_score'] < 0.1])
        },
        "sample_results": results[:10],
        "interpretation": {
            "precision_meaning": "% of recommendations that user actually purchased",
            "recall_meaning": "% of user's purchases that were recommended",
            "evaluation_note": "Based on actual purchase behavior"
        }
    }

@app.get("/debug_recommendations")
def debug_recommendations(user_id: int, db: Session = Depends(get_db)):
    """تشخيص مفصل لمشكلة التوصيات"""
    
    # تفاعلات المستخدم
    interactions = db.query(UserInteraction).filter(
        UserInteraction.user_id == user_id
    ).order_by(UserInteraction.timestamp.desc()).all()
    
    if not interactions:
        return {"error": f"No interactions found for user {user_id}"}
    
    # تحليل التفاعلات
    interaction_analysis = {
        'total_interactions': len(interactions),
        'searches': len([i for i in interactions if i.action == 'search']),
        'clicks': len([i for i in interactions if i.action == 'click']),
        'purchases': len([i for i in interactions if i.action == 'purchase']),
        'unique_coupons': len(set(i.coupon_id for i in interactions)),
        'total_score': sum(i.score for i in interactions)
    }
    
    # الكوبونات المستبعدة
    seen_coupons = set(inter.coupon_id for inter in interactions)
    
    # فئات المستخدم
    user_categories = {}
    for inter in interactions:
        coupon = db.query(Coupon).filter(Coupon.id == inter.coupon_id).first()
        if coupon:
            category = db.query(Category).filter(Category.id == coupon.category_id).first()
            if category:
                user_categories[category.name] = user_categories.get(category.name, 0) + inter.score
    
    # الحصول على التوصيات
    try:
        recommendations_data = get_recommendations(user_id, 10, db)
        recommendations_success = True
    except Exception as e:
        recommendations_data = {"error": str(e)}
        recommendations_success = False
    
    # تحليل التوصيات
    if recommendations_success:
        recommended_items = set(recommendations_data["recommendations"])
        overlap_with_history = recommended_items.intersection(seen_coupons)
        
        # تحليل فئات التوصيات
        rec_categories = {}
        for coupon_id in recommendations_data["recommendations"]:
            coupon = db.query(Coupon).filter(Coupon.id == coupon_id).first()
            if coupon:
                category = db.query(Category).filter(Category.id == coupon.category_id).first()
                if category:
                    rec_categories[category.name] = rec_categories.get(category.name, 0) + 1
    
    return {
        "user_analysis": {
            "user_id": user_id,
            "interaction_summary": interaction_analysis,
            "top_categories": dict(sorted(user_categories.items(), key=lambda x: x[1], reverse=True)[:5]),
            "recent_interactions": [
                {
                    "coupon_id": i.coupon_id,
                    "action": i.action,
                    "score": i.score,
                    "timestamp": i.timestamp.isoformat()
                } for i in interactions[:5]
            ]
        },
        "recommendation_analysis": {
            "success": recommendations_success,
            "recommendations": recommendations_data.get("recommendations", []) if recommendations_success else [],
            "recommended_categories": rec_categories if recommendations_success else {},
            "overlap_with_history": list(overlap_with_history) if recommendations_success else [],
            "overlap_count": len(overlap_with_history) if recommendations_success else 0
        },
        "system_behavior": {
            "excludes_seen_items": len(overlap_with_history) == 0 if recommendations_success else "Unknown",
            "total_excluded_items": len(seen_coupons),
            "recommendation_pool_size": 1000 - len(seen_coupons),  # assuming 1000 total coupons
            "category_alignment": "Good" if recommendations_success and any(
                cat in user_categories for cat in rec_categories.keys()
            ) else "Poor"
        },
        "diagnosis": {
            "main_issue": "System excludes ALL previously interacted items",
            "impact": "Zero overlap with test set in evaluation",
            "solution_needed": "Allow some re-recommendation or change evaluation method",
            "recommendation": "Use temporal split or modify recommendation logic"
        }
    }

@app.get("/quick_performance_check")
def quick_performance_check(sample_size: int = 10, db: Session = Depends(get_db)):
    """فحص سريع جداً للأداء"""
    
    start_time = datetime.now()
    
    # إحصائيات أساسية
    total_users = db.query(UserInteraction.user_id).distinct().count()
    total_interactions = db.query(UserInteraction).count()
    total_coupons = db.query(Coupon).count()
    
    # اختبار عينة صغيرة
    sample_users = db.query(UserInteraction.user_id).distinct().limit(sample_size).all()
    
    successful_recommendations = 0
    avg_recommendation_time = 0
    
    for user_tuple in sample_users:
        user_id = user_tuple[0]
        try:
            rec_start = datetime.now()
            recommendations = get_recommendations(user_id, 5, db)
            rec_time = (datetime.now() - rec_start).total_seconds()
            
            if recommendations.get("recommendations"):
                successful_recommendations += 1
                avg_recommendation_time += rec_time
                
        except Exception as e:
            print(f"Error with user {user_id}: {e}")
    
    avg_recommendation_time = avg_recommendation_time / successful_recommendations if successful_recommendations > 0 else 0
    
    # تقييم الصحة
    health_score = 0
    if total_users >= 100: health_score += 25
    if total_interactions >= 1000: health_score += 25
    if successful_recommendations >= sample_size * 0.8: health_score += 25
    if avg_recommendation_time < 2.0: health_score += 25
    
    health_status = (
        "Excellent" if health_score >= 90 else
        "Good" if health_score >= 70 else
        "Fair" if health_score >= 50 else
        "Poor"
    )
    
    duration = (datetime.now() - start_time).total_seconds()
    
    return {
        "quick_check_summary": {
            "health_status": health_status,
            "health_score": health_score,
            "check_duration": round(duration, 2)
        },
        "system_stats": {
            "total_users": total_users,
            "total_interactions": total_interactions,
            "total_coupons": total_coupons,
            "avg_interactions_per_user": round(total_interactions / total_users, 2) if total_users > 0 else 0
        },
        "performance_test": {
            "sample_size": sample_size,
            "successful_recommendations": successful_recommendations,
            "success_rate": round(successful_recommendations / sample_size * 100, 1),
            "avg_recommendation_time": round(avg_recommendation_time, 3),
            "recommendation_speed": "Fast" if avg_recommendation_time < 1 else "Moderate" if avg_recommendation_time < 3 else "Slow"
        },
        "system_readiness": {
            "vector_store": "Ready" if faiss_index is not None else "Not Built",
            "ml_model": "Loaded" if model is not None else "Not Loaded",
            "database": "Connected"
        }
    }

@app.get("/get_recommendations_with_rerank")
def get_recommendations_with_rerank(user_id: int, top_n: int = 10, allow_seen: bool = True, db: Session = Depends(get_db)):
    """توصيات مع السماح بإعادة اقتراح بعض العناصر المُشاهدة"""
    
    if faiss_index is None or model is None:
        raise HTTPException(status_code=500, detail="System not ready")
    
    interactions = db.query(UserInteraction).filter(UserInteraction.user_id == user_id).all()
    
    if not interactions:
        # مستخدم جديد - اقترح الأشهر
        popular_coupons = db.query(UserInteraction.coupon_id, func.sum(UserInteraction.score).label('total_score'))\
                           .group_by(UserInteraction.coupon_id)\
                           .order_by(func.sum(UserInteraction.score).desc())\
                           .limit(top_n).all()
        return {"recommendations": [c.coupon_id for c in popular_coupons], "method": "popular"}
    
    # بناء ملف المستخدم
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
    
    # البحث في المتجهات
    similarities, indices = faiss_index.search(weighted_emb.reshape(1, -1).astype('float32'), min(top_n * 5, len(coupon_ids)))
    
    recommendations = []
    category_counts = {}
    max_per_category = max(2, top_n // len(user_categories)) if user_categories else top_n
    
    # **التعديل الرئيسي:** السماح ببعض العناصر المُشاهدة
    seen_allowed = 0
    max_seen_allowed = max(1, top_n // 4)  # السماح بـ 25% من العناصر المُشاهدة
    
    for sim, idx in zip(similarities[0], indices[0]):
        if len(recommendations) >= top_n:
            break
            
        coupon_id = coupon_ids[idx]
        
        # التحقق من العناصر المُشاهدة
        if coupon_id in seen_coupons:
            if not allow_seen or seen_allowed >= max_seen_allowed:
                continue
            seen_allowed += 1
        
        coupon = db.query(Coupon).filter(Coupon.id == coupon_id).first()
        if not coupon:
            continue
            
        category = db.query(Category).filter(Category.id == coupon.category_id).first()
        category_name = category.name if category else "Unknown"
        
        # تحديد التوزيع
        if category_counts.get(category_name, 0) >= max_per_category:
            continue
            
        recommendations.append(coupon_id)
        category_counts[category_name] = category_counts.get(category_name, 0) + 1
    
    # ملء الباقي إذا لم نصل للعدد المطلوب
    while len(recommendations) < top_n:
        for sim, idx in zip(similarities[0], indices[0]):
            if len(recommendations) >= top_n:
                break
                
            coupon_id = coupon_ids[idx]
            if coupon_id not in recommendations:
                recommendations.append(coupon_id)
    
    return {
        "recommendations": recommendations[:top_n],
        "method": "content_based_with_rerank",
        "user_categories": user_categories,
        "seen_items_included": seen_allowed,
        "total_seen_items": len(seen_coupons)
    }

@app.get("/get_recommendations_simple")
def get_recommendations_simple(user_id: int, top_n: int = 10, db: Session = Depends(get_db)):
    """توصيات بسيطة - تركز على التشابه فقط"""
    
    if faiss_index is None or model is None:
        raise HTTPException(status_code=500, detail="System not ready")
    
    interactions = db.query(UserInteraction).filter(UserInteraction.user_id == user_id).all()
    
    if not interactions:
        # مستخدم جديد
        popular_coupons = db.query(Coupon.id).order_by(func.random()).limit(top_n).all()
        return {"recommendations": [c.id for c in popular_coupons], "method": "random_for_new_user"}
    
    # بناء ملف المستخدم البسيط
    user_text = ""
    user_categories = {}
    
    for inter in interactions:
        coupon = db.query(Coupon).filter(Coupon.id == inter.coupon_id).first()
        if coupon:
            category = db.query(Category).filter(Category.id == coupon.category_id).first()
            coupon_type = db.query(CouponType).filter(CouponType.id == coupon.coupon_type_id).first()
            
            # وزن النص حسب النقاط
            weight = int(inter.score / 2)  # تقليل الوزن
            text = build_enhanced_text(coupon, category, coupon_type)
            user_text += (text + " ") * weight
            
            if category:
                user_categories[category.name] = user_categories.get(category.name, 0) + inter.score
    
    # تشفير ملف المستخدم
    user_embedding = model.encode([user_text])[0]
    
    # البحث
    similarities, indices = faiss_index.search(user_embedding.reshape(1, -1).astype('float32'), top_n * 2)
    
    # أخذ أفضل النتائج مباشرة (بدون تعقيد)
    recommendations = []
    for sim, idx in zip(similarities[0], indices[0]):
        if len(recommendations) >= top_n:
            break
        recommendations.append(coupon_ids[idx])
    
    return {
        "recommendations": recommendations,
        "method": "simple_content_based",
        "user_categories": user_categories
    }

@app.get("/evaluate_simple_system")
def evaluate_simple_system(test_users: int = 20, db: Session = Depends(get_db)):
    """تقييم النظام البسيط"""
    
    active_users = db.query(UserInteraction.user_id).distinct().limit(test_users).all()
    
    results = []
    
    for user_tuple in active_users:
        user_id = user_tuple[0]
        
        # تفاعلات المستخدم
        interactions = db.query(UserInteraction).filter(
            UserInteraction.user_id == user_id
        ).order_by(UserInteraction.timestamp).all()
        
        if len(interactions) < 5:
            continue
        
        # تقسيم بسيط: آخر 20% كاختبار
        split = int(len(interactions) * 0.8)
        test_interactions = interactions[split:]
        
        # العناصر المناسبة
        relevant_items = set(inter.coupon_id for inter in test_interactions if inter.score >= 5.0)
        
        if not relevant_items:
            continue
        
        # التوصيات مع السماح بإعادة الاقتراح
        try:
            rec_data = get_recommendations_with_rerank(user_id, 10, True, db)
            recommended = set(rec_data["recommendations"])
        except:
            continue
        
        # حساب التطابق
        matches = len(recommended.intersection(relevant_items))
        precision = matches / len(recommended) if recommended else 0
        recall = matches / len(relevant_items) if relevant_items else 0
        
        results.append({
            'user_id': user_id,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'matches': matches,
            'relevant_count': len(relevant_items),
            'recommended_count': len(recommended)
        })
    
    if not results:
        return {"error": "No results"}
    
    avg_precision = sum(r['precision'] for r in results) / len(results)
    avg_recall = sum(r['recall'] for r in results) / len(results)
    avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
    
    return {
        "simple_evaluation": {
            "average_precision": round(avg_precision, 4),
            "average_recall": round(avg_recall, 4),
            "average_f1": round(avg_f1, 4),
            "users_evaluated": len(results)
        },
        "expected_improvement": f"F1 should improve from 0.0134 to ~{avg_f1:.4f}",
        "sample_results": results[:5]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.SERVICE_HOST,
        port=config.SERVICE_PORT,
        log_level="info"
    )

@app.get("/get_recommendations_smart_rerank")
def get_recommendations_smart_rerank(user_id: int, top_n: int = 10, db: Session = Depends(get_db)):
    """توصيات ذكية مع السماح بإعادة الاقتراح بناءً على التفضيلات"""
    
    if faiss_index is None or model is None:
        raise HTTPException(status_code=500, detail="System not ready")
    
    interactions = db.query(UserInteraction).filter(UserInteraction.user_id == user_id).all()
    
    if not interactions:
        popular_coupons = db.query(UserInteraction.coupon_id, func.sum(UserInteraction.score).label('total_score'))\
                           .group_by(UserInteraction.coupon_id)\
                           .order_by(func.sum(UserInteraction.score).desc())\
                           .limit(top_n).all()
        return {"recommendations": [c.coupon_id for c in popular_coupons], "method": "popular"}
    
    # بناء ملف المستخدم
    weighted_emb = np.zeros(vector_dim)
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
    
    search_size = min(top_n * 10, len(coupon_ids))
    similarities, indices = faiss_index.search(weighted_emb.reshape(1, -1).astype('float32'), search_size)
    
    recommendations = []
    category_counts = {}
    max_per_category = max(2, top_n // len(user_categories)) if user_categories else top_n
    
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
            
        coupon_id = coupon_ids[idx]
        
        if coupon_id in low_score_seen:
            continue
        
        if coupon_id in high_score_seen:
            if high_score_allowed >= max_high_score:
                continue
            high_score_allowed += 1
        
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
            if coupon_id not in recommendations and coupon_id not in seen_coupons:
                recommendations.append(coupon_id)
    
    return {
        "recommendations": recommendations[:top_n],
        "method": "smart_rerank_content_based",
        "user_categories": user_categories,
        "rerank_stats": {
            "high_score_items_included": high_score_allowed,
            "total_seen_items": len(seen_coupons),
            "high_score_seen_items": len(high_score_seen),
            "low_score_seen_items": len(low_score_seen)
        }
    }

@app.get("/evaluate_smart_system")
def evaluate_smart_system(test_users: int = 20, top_k: int = 10, db: Session = Depends(get_db)):
    """تقييم النظام الذكي مع إعادة الاقتراح"""
    
    logger.info(f"🔍 Starting smart evaluation with {test_users} users...")
    
    try:
        # الحصول على مستخدمين نشطين
        active_users = db.query(
            UserInteraction.user_id,
            func.count(UserInteraction.id).label('interaction_count')
        ).group_by(UserInteraction.user_id)\
         .having(func.count(UserInteraction.id) >= 10)\
         .order_by(func.count(UserInteraction.id).desc())\
         .limit(test_users).all()
        
        if not active_users:
            return {"error": "No active users found"}
        
        logger.info(f"📊 Found {len(active_users)} active users")
        
        precision_scores = []
        recall_scores = []
        f1_scores = []
        evaluation_details = []
        
        for i, (user_id, interaction_count) in enumerate(active_users):
            try:
                logger.info(f"  Evaluating user {user_id} ({i+1}/{len(active_users)})")
                
                # تفاعلات المستخدم
                all_interactions = db.query(UserInteraction).filter(
                    UserInteraction.user_id == user_id
                ).order_by(UserInteraction.timestamp).all()
                
                if len(all_interactions) < 10:
                    continue
                
                # تقسيم زمني
                split_point = int(len(all_interactions) * 0.8)
                test_interactions = all_interactions[split_point:]
                
                # العناصر المناسبة في الاختبار
                relevant_items = set()
                for inter in test_interactions:
                    if inter.score >= 5.0:  # click أو purchase
                        relevant_items.add(inter.coupon_id)
                
                if not relevant_items:
                    continue
                
                # حذف مؤقت لتفاعلات الاختبار
                test_interaction_ids = [inter.id for inter in test_interactions]
                
                # حفظ تفاعلات الاختبار
                test_data = []
                for inter in test_interactions:
                    test_data.append({
                        'user_id': inter.user_id,
                        'coupon_id': inter.coupon_id,
                        'action': inter.action,
                        'score': inter.score,
                        'timestamp': inter.timestamp
                    })
                
                # حذف مؤقت
                db.query(UserInteraction).filter(
                    UserInteraction.id.in_(test_interaction_ids)
                ).delete(synchronize_session=False)
                db.commit()
                
                # الحصول على التوصيات الذكية
                try:
                    recommendations_data = get_recommendations_smart_rerank(user_id, top_k, db)
                    recommended_items = set(recommendations_data["recommendations"])
                except Exception as e:
                    logger.error(f"Error getting recommendations for user {user_id}: {e}")
                    recommended_items = set()
                
                # إعادة إدراج تفاعلات الاختبار
                for data in test_data:
                    new_interaction = UserInteraction(
                        user_id=data['user_id'],
                        coupon_id=data['coupon_id'],
                        action=data['action'],
                        score=data['score'],
                        timestamp=data['timestamp']
                    )
                    db.add(new_interaction)
                db.commit()
                
                # حساب المقاييس
                true_positives = len(recommended_items.intersection(relevant_items))
                
                precision = true_positives / len(recommended_items) if recommended_items else 0
                recall = true_positives / len(relevant_items) if relevant_items else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                
                evaluation_details.append({
                    'user_id': user_id,
                    'relevant_items': len(relevant_items),
                    'recommended_items': len(recommended_items),
                    'true_positives': true_positives,
                    'precision': round(precision, 4),
                    'recall': round(recall, 4),
                    'f1_score': round(f1, 4),
                    'overlap_items': list(recommended_items.intersection(relevant_items))[:3]
                })
                
            except Exception as e:
                logger.error(f"Error with user {user_id}: {e}")
                continue
        
        if not evaluation_details:
            return {"error": "No valid evaluations completed"}
        
        # حساب النتائج
        avg_precision = np.mean(precision_scores) if precision_scores else 0
        avg_recall = np.mean(recall_scores) if recall_scores else 0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        
        # مقارنة مع النتائج السابقة
        previous_f1 = 0.0134
        improvement_f1 = ((avg_f1 - previous_f1) / previous_f1 * 100) if previous_f1 > 0 else 0
        
        # تحديد مستوى الأداء
        if avg_f1 >= 0.15:
            performance_level = "Good"
        elif avg_f1 >= 0.05:
            performance_level = "Fair"
        else:
            performance_level = "Poor"
        
        return {
            "smart_evaluation_summary": {
                "performance_level": performance_level,
                "users_evaluated": len(evaluation_details),
                "evaluation_method": "Smart re-ranking with high-score preferences",
                "evaluation_timestamp": datetime.now().isoformat()
            },
            "core_metrics": {
                "precision": {
                    "average": round(avg_precision, 4),
                    "previous": 0.0167,
                    "improvement": f"{round(((avg_precision - 0.0167) / 0.0167 * 100) if 0.0167 > 0 else 0, 1)}%"
                },
                "recall": {
                    "average": round(avg_recall, 4),
                    "previous": 0.0113,
                    "improvement": f"{round(((avg_recall - 0.0113) / 0.0113 * 100) if 0.0113 > 0 else 0, 1)}%"
                },
                "f1_score": {
                    "average": round(avg_f1, 4),
                    "previous": previous_f1,
                    "improvement": f"{round(improvement_f1, 1)}%"
                }
            },
            "performance_distribution": {
                "good_users": len([r for r in evaluation_details if r['f1_score'] >= 0.15]),
                "fair_users": len([r for r in evaluation_details if 0.05 <= r['f1_score'] < 0.15]),
                "poor_users": len([r for r in evaluation_details if r['f1_score'] < 0.05]),
                "users_with_matches": len([r for r in evaluation_details if r['true_positives'] > 0])
            },
            "sample_results": evaluation_details[:5],
            "improvement_summary": f"F1-Score improved from {previous_f1:.4f} to {avg_f1:.4f} ({improvement_f1:.1f}% improvement)"
        }
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return {"error": f"Evaluation failed: {str(e)}"}

@app.get("/simple_smart_test")
def simple_smart_test(user_id: int = 372, db: Session = Depends(get_db)):
    """اختبار بسيط للنظام الذكي"""
    
    try:
        # اختبار التوصيات العادية
        normal_recs = get_recommendations(user_id, 5, db)
        
        # اختبار التوصيات الذكية
        smart_recs = get_recommendations_smart_rerank(user_id, 5, db)
        
        return {
            "test_result": "Success",
            "user_id": user_id,
            "normal_recommendations": normal_recs["recommendations"],
            "smart_recommendations": smart_recs["recommendations"],
            "difference": {
                "overlap": len(set(normal_recs["recommendations"]).intersection(set(smart_recs["recommendations"]))),
                "smart_method_stats": smart_recs.get("rerank_stats", {})
            }
        }
        
    except Exception as e:
        return {"error": f"Test failed: {str(e)}"}

@app.get("/compare_recommendation_methods")
def compare_recommendation_methods(user_id: int, top_n: int = 10, db: Session = Depends(get_db)):
    """مقارنة بين طرق التوصية المختلفة"""
    
    try:
        original_recs = get_recommendations(user_id, top_n, db)
        smart_recs = get_recommendations_smart_rerank(user_id, top_n, db)
        
        original_set = set(original_recs["recommendations"])
        smart_set = set(smart_recs["recommendations"])
        
        overlap = original_set.intersection(smart_set)
        only_original = original_set - smart_set
        only_smart = smart_set - original_set
        
        return {
            "user_id": user_id,
            "comparison_summary": {
                "overlap_count": len(overlap),
                "overlap_percentage": round(len(overlap) / top_n * 100, 1),
                "unique_to_original": len(only_original),
                "unique_to_smart": len(only_smart)
            },
            "original_method": {
                "recommendations": original_recs["recommendations"],
                "method": original_recs.get("method", "content_based")
            },
            "smart_method": {
                "recommendations": smart_recs["recommendations"],
                "method": smart_recs.get("method", "smart_rerank"),
                "rerank_stats": smart_recs.get("rerank_stats", {})
            },
            "analysis": {
                "overlap_items": list(overlap),
                "only_in_original": list(only_original),
                "only_in_smart": list(only_smart)
            }
        }
        
    except Exception as e:
        return {"error": f"Comparison failed: {str(e)}"}

@app.get("/test_user_recommendations")
def test_user_recommendations(user_id: int = 372, db: Session = Depends(get_db)):
    """اختبار سريع لتوصيات مستخدم محدد"""
    
    interactions = db.query(UserInteraction).filter(
        UserInteraction.user_id == user_id
    ).all()
    
    if not interactions:
        return {"error": f"No interactions found for user {user_id}"}
    
    user_stats = {
        "total_interactions": len(interactions),
        "searches": len([i for i in interactions if i.action == 'search']),
        "clicks": len([i for i in interactions if i.action == 'click']),
        "purchases": len([i for i in interactions if i.action == 'purchase']),
        "total_score": sum(i.score for i in interactions),
        "unique_coupons": len(set(i.coupon_id for i in interactions))
    }
    
    try:
        smart_recommendations = get_recommendations_smart_rerank(user_id, 10, db)
        
        return {
            "user_analysis": {
                "user_id": user_id,
                "stats": user_stats,
                "top_categories": dict(list(smart_recommendations["user_categories"].items())[:5])
            },
            "recommendations": smart_recommendations,
            "test_result": "Success - Smart recommendations generated"
        }
        
    except Exception as e:
        return {
            "user_analysis": {
                "user_id": user_id,
                "stats": user_stats
            },
            "error": f"Failed to generate recommendations: {str(e)}"
        }
        