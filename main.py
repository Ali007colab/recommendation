# import logging
# from fastapi import FastAPI, Depends, HTTPException
# from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime, func
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker, Session
# from sqlalchemy.exc import IntegrityError
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import faiss
# from datetime import datetime
# import os

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# DATABASE_URL = "mysql+pymysql://root:@localhost/recommendation_db"
# logger.info("Connecting to database: %s", DATABASE_URL)
# try:
#     engine = create_engine(DATABASE_URL)
#     SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#     logger.info("Database connection established successfully")
# except Exception as e:
#     logger.error("Failed to connect to database: %s", str(e))
#     raise

# Base = declarative_base()

# class Coupon(Base):
#     __tablename__ = "coupons"
#     id = Column(Integer, primary_key=True, index=True)
#     name = Column(String(255))
#     description = Column(String(255))
#     price = Column(Float)
#     coupon_type_id = Column(Integer, ForeignKey("coupon_types.id"))
#     category_id = Column(Integer, ForeignKey("categories.id"))
#     provider_id = Column(Integer)
#     coupon_status = Column(Integer)
#     coupon_code = Column(String(255))
#     date = Column(String(255))
#     created_at = Column(DateTime, default=func.now())
#     updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# class Category(Base):
#     __tablename__ = "categories"
#     id = Column(Integer, primary_key=True)
#     name = Column(String(255))

# class CouponType(Base):
#     __tablename__ = "coupon_types"
#     id = Column(Integer, primary_key=True)
#     name = Column(String(255))
#     description = Column(String(255))

# class UserInteraction(Base):
#     __tablename__ = "user_interactions"
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, index=True)
#     coupon_id = Column(Integer, index=True)
#     action = Column(String(50))
#     score = Column(Float)
#     timestamp = Column(DateTime, default=func.now())

# logger.info("Creating database tables if not exist")
# Base.metadata.create_all(bind=engine)
# logger.info("Database tables created successfully")

# app = FastAPI()
# logger.info("FastAPI application initialized")

# logger.info("Loading sentence-transformers model: all-MiniLM-L6-v2")
# try:
#     model = SentenceTransformer('all-MiniLM-L6-v2')
#     logger.info("Model loaded successfully")
# except Exception as e:
#     logger.error("Failed to load model: %s", str(e))
#     raise

# faiss_index = None
# coupon_ids = []
# vector_dim = 384
# logger.info("Initialized FAISS index placeholder")

# def build_vector_store(db: Session):
#     global faiss_index, coupon_ids
#     logger.info("Building FAISS vector store")
#     coupons = db.query(Coupon).all()
#     logger.info("Retrieved %d coupons from database", len(coupons))
#     texts = []
#     coupon_ids = []
#     for coupon in coupons:
#         category = db.query(Category).filter(Category.id == coupon.category_id).first()
#         coupon_type = db.query(CouponType).filter(CouponType.id == coupon.coupon_type_id).first()
#         category_emphasis = f"{category.name} " * 8 if category else ''
#         text = f"{category_emphasis}{coupon.name} {coupon.description} {coupon_type.name if coupon_type else ''}"
#         texts.append(text)
#         coupon_ids.append(coupon.id)
    
#     if not texts:
#         logger.warning("No coupons found for vector store")
#         return
    
#     logger.info("Generating embeddings for %d coupons", len(texts))
#     embeddings = model.encode(texts)
#     logger.info("Embeddings generated successfully")
    
#     faiss_index = faiss.IndexFlatIP(vector_dim)
#     embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
#     faiss_index.add(embeddings.astype('float32'))
#     logger.info("FAISS index built with %d vectors", faiss_index.ntotal)

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
#         logger.info("Database session closed")

# @app.post("/log_event")
# def log_event(user_id: int, coupon_id: int, action: str, db: Session = Depends(get_db)):
#     logger.info("Received log_event request: user_id=%d, coupon_id=%d, action=%s", user_id, coupon_id, action)
#     if action not in ['search', 'click', 'purchase']:
#         logger.error("Invalid action: %s", action)
#         raise HTTPException(status_code=400, detail="Invalid action")
    
#     score = {'search': 2.0, 'click': 5.0, 'purchase': 15.0}[action]
    
#     interaction = UserInteraction(user_id=user_id, coupon_id=coupon_id, action=action, score=score)
#     try:
#         db.add(interaction)
#         db.commit()
#         logger.info("Interaction logged with enhanced score: user_id=%d, coupon_id=%d, score=%.1f", user_id, coupon_id, score)
#     except IntegrityError as e:
#         db.rollback()
#         logger.error("Failed to log interaction: %s", str(e))
#         raise HTTPException(status_code=400, detail="Database error")
#     return {"status": "logged", "score": score}

# @app.post("/analyze")
# def analyze(db: Session = Depends(get_db)):
#     logger.info("Starting analyze endpoint")
#     build_vector_store(db)
#     logger.info("Analyze completed")
#     return {"status": "vector store updated"}

# def apply_category_scaling(recommendations, user_categories, db, min_category_percentage=0.1, max_category_percentage=0.7):
#     if not user_categories or len(recommendations) < 5:
#         return recommendations
    
#     total_user_score = sum(user_categories.values())
#     category_targets = {}
    
#     for category_id, score in user_categories.items():
#         natural_percentage = score / total_user_score
#         scaled_percentage = max(min_category_percentage, min(natural_percentage, max_category_percentage))
#         category_targets[category_id] = int(len(recommendations) * scaled_percentage)
    
#     remaining_slots = len(recommendations) - sum(category_targets.values())
#     if remaining_slots > 0:
#         sorted_categories = sorted(user_categories.items(), key=lambda x: x[1], reverse=True)
#         for category_id, _ in sorted_categories:
#             if remaining_slots <= 0:
#                 break
#             category_targets[category_id] += 1
#             remaining_slots -= 1
    
#     scaled_recommendations = []
#     category_counts = {cat_id: 0 for cat_id in category_targets.keys()}
#     used_coupons = set()
    
#     for rec_id in recommendations:
#         coupon = db.query(Coupon).filter(Coupon.id == rec_id).first()
#         if not coupon or rec_id in used_coupons:
#             continue
        
#         category_id = coupon.category_id
#         target_count = category_targets.get(category_id, 0)
#         current_count = category_counts.get(category_id, 0)
        
#         if current_count < target_count:
#             scaled_recommendations.append(rec_id)
#             category_counts[category_id] = current_count + 1
#             used_coupons.add(rec_id)
    
#     for rec_id in recommendations:
#         if rec_id not in used_coupons and len(scaled_recommendations) < len(recommendations):
#             scaled_recommendations.append(rec_id)
#             used_coupons.add(rec_id)
    
#     return scaled_recommendations

# @app.get("/get_recommendations")
# def get_recommendations_enhanced(user_id: int, top_n: int = 5, enable_scaling: bool = True, db: Session = Depends(get_db)):
#     logger.info("Enhanced recommendations for user_id=%d", user_id)
    
#     if faiss_index is None:
#         raise HTTPException(status_code=500, detail="Vector store not built")
    
#     interactions = db.query(UserInteraction).filter(UserInteraction.user_id == user_id).all()
    
#     if not interactions:
#         popular_coupons = db.query(UserInteraction.coupon_id, func.sum(UserInteraction.score).label('total_score'))\
#                            .group_by(UserInteraction.coupon_id)\
#                            .order_by(func.sum(UserInteraction.score).desc())\
#                            .limit(top_n).all()
#         return {"recommendations": [c.coupon_id for c in popular_coupons]}
    
#     category_weights = {}
#     seen_coupons = set()
    
#     for inter in interactions:
#         coupon = db.query(Coupon).filter(Coupon.id == inter.coupon_id).first()
#         if not coupon: continue
        
#         seen_coupons.add(coupon.id)
        
#         action_multiplier = 3.0 if inter.action == 'purchase' else 1.5 if inter.action == 'click' else 1.0
#         final_score = inter.score * action_multiplier
        
#         category_weights[coupon.category_id] = category_weights.get(coupon.category_id, 0) + final_score
        
#         logger.info("Enhanced scoring: action=%s, base_score=%.1f, multiplier=%.1f, final=%.1f", 
#                     inter.action, inter.score, action_multiplier, final_score)
    
#     sorted_categories = sorted(category_weights.items(), key=lambda x: x[1], reverse=True)
    
#     recommendations = []
    
#     primary_category = sorted_categories[0][0] if sorted_categories else None
#     if primary_category:
#         primary_coupons = db.query(Coupon).filter(
#             Coupon.category_id == primary_category,
#             ~Coupon.id.in_(seen_coupons)
#         ).limit(max(1, int(top_n * 0.6))).all()
        
#         recommendations.extend([c.id for c in primary_coupons])
    
#     remaining_slots = top_n - len(recommendations)
#     if remaining_slots > 0:
#         excluded_ids = list(seen_coupons) + recommendations
        
#         for category_id, score in sorted_categories[1:]:
#             similar_coupons = db.query(Coupon).filter(
#                 Coupon.category_id == category_id,
#                 ~Coupon.id.in_(excluded_ids)
#             ).limit(remaining_slots).all()
            
#             recommendations.extend([c.id for c in similar_coupons])
#             remaining_slots -= len(similar_coupons)
            
#             if remaining_slots <= 0:
#                 break
        
#         if remaining_slots > 0:
#             other_coupons = db.query(Coupon).filter(
#                 ~Coupon.id.in_(list(seen_coupons) + recommendations)
#             ).limit(remaining_slots).all()
            
#             recommendations.extend([c.id for c in other_coupons])

#     if enable_scaling and len(recommendations) >= 5:
#         recommendations = apply_category_scaling(recommendations, category_weights, db)

#     logger.info("Enhanced recommendations: %s", recommendations[:top_n])
#     return {"recommendations": recommendations[:top_n]}

# def build_enhanced_text(coupon, category, coupon_type):
#     category_tokens = f"CATEGORY_{category.name} " * 25 if category else ''
#     type_tokens = f"TYPE_{coupon_type.name} " * 8 if coupon_type else ''
    
#     name_emphasis = f"TITLE_{coupon.name} {coupon.name} {coupon.name} {coupon.name} {coupon.name} "
    
#     description_reduced = f"summary_{coupon.description} " if coupon.description else ''
    
#     price_range = "expensive" if coupon.price > 100 else "affordable" if coupon.price > 20 else "cheap"
    
#     return f"{category_tokens}{type_tokens}{name_emphasis}{description_reduced}{price_range}"

# def build_user_profile_enhanced(interactions, db):
#     category_scores = {}
#     type_scores = {}
    
#     for inter in interactions:
#         coupon = db.query(Coupon).filter(Coupon.id == inter.coupon_id).first()
#         if not coupon: continue
        
#         category_scores[coupon.category_id] = category_scores.get(coupon.category_id, 0) + inter.score
        
#         type_scores[coupon.coupon_type_id] = type_scores.get(coupon.coupon_type_id, 0) + inter.score
    
#     return category_scores, type_scores

# @app.get("/evaluate_similarity")
# def evaluate_similarity(user_id: int, db: Session = Depends(get_db)):
    
#     if faiss_index is None:
#         raise HTTPException(status_code=500, detail="Vector store not built")
    
#     interactions = db.query(UserInteraction).filter(UserInteraction.user_id == user_id).all()
    
#     if not interactions:
#         return {"error": "No interactions found for user"}
    
#     weighted_emb = np.zeros(vector_dim)
#     total_weight = 0
#     user_categories = {}
    
#     for inter in interactions:
#         coupon = db.query(Coupon).filter(Coupon.id == inter.coupon_id).first()
#         if not coupon: continue
        
#         category = db.query(Category).filter(Category.id == coupon.category_id).first()
#         coupon_type = db.query(CouponType).filter(CouponType.id == coupon.coupon_type_id).first()
        
#         category_tokens = f"{category.name} " * 15 if category else ''
#         type_tokens = f"{coupon_type.name} " * 8 if coupon_type else ''
#         price_range = "expensive" if coupon.price > 100 else "affordable" if coupon.price > 20 else "cheap"
#         text = f"{category_tokens}{type_tokens}{coupon.name} {coupon.description} {price_range}"
        
#         emb = model.encode([text])[0]
        
#         action_boost = 4.0 if inter.action == 'purchase' else 2.0 if inter.action == 'click' else 1.0
#         final_weight = inter.score * action_boost
        
#         weighted_emb += emb * final_weight
#         total_weight += final_weight
        
#         cat_name = category.name if category else "Unknown"
#         user_categories[cat_name] = user_categories.get(cat_name, 0) + inter.score
    
#     if total_weight == 0:
#         return {"error": "No valid interactions"}
    
#     user_vector = weighted_emb / total_weight
    
#     # الإصلاح: استدعاء الدالة مباشرة بدل تمرير db كـ parameter
#     recommendations_result = get_recommendations_enhanced(user_id, 10, True, db)
#     recommendations = recommendations_result["recommendations"]
    
#     similarities = []
#     category_distribution = {}
    
#     for rec_id in recommendations:
#         if rec_id >= len(coupon_ids):
#             continue
            
#         rec_coupon = db.query(Coupon).filter(Coupon.id == rec_id).first()
#         if not rec_coupon: continue
        
#         rec_category = db.query(Category).filter(Category.id == rec_coupon.category_id).first()
#         rec_type = db.query(CouponType).filter(CouponType.id == rec_coupon.coupon_type_id).first()
        
#         category_tokens = f"{rec_category.name} " * 15 if rec_category else ''
#         type_tokens = f"{rec_type.name} " * 8 if rec_type else ''
#         price_range = "expensive" if rec_coupon.price > 100 else "affordable" if rec_coupon.price > 20 else "cheap"
#         rec_text = f"{category_tokens}{type_tokens}{rec_coupon.name} {rec_coupon.description} {price_range}"
        
#         rec_vector = model.encode([rec_text])[0]
        
#         similarity = cosine_similarity([user_vector], [rec_vector])[0][0]
        
#         similarities.append({
#             "coupon_id": rec_id,
#             "coupon_name": rec_coupon.name,
#             "category": rec_category.name if rec_category else "Unknown",
#             "cosine_similarity": float(similarity),
#             "price": float(rec_coupon.price)
#         })
        
#         cat_name = rec_category.name if rec_category else "Unknown"
#         category_distribution[cat_name] = category_distribution.get(cat_name, 0) + 1
    
#     avg_similarity = np.mean([s["cosine_similarity"] for s in similarities]) if similarities else 0
    
#     quality_score = "Excellent" if avg_similarity > 0.8 else "Good" if avg_similarity > 0.6 else "Fair" if avg_similarity > 0.4 else "Poor"
    
#     return {
#         "user_id": user_id,
#         "user_preferred_categories": user_categories,
#         "recommendations_analysis": similarities,
#         "average_cosine_similarity": float(avg_similarity),
#         "quality_assessment": quality_score,
#         "category_distribution_in_recommendations": category_distribution,
#         "total_recommendations": len(similarities)
#     }