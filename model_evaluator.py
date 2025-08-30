#!/usr/bin/env python3

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from database import SessionLocal, Coupon, Category, CouponType
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/model_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        self.models = {}
        self.evaluation_results = {}
        
        # Available models for testing
        self.available_models = {
            "current": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "description": "Current model (English focused)"
            },
            "multilingual": {
                "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "description": "Multilingual model (Arabic + English)"
            },
            "multilingual_cased": {
                "name": "sentence-transformers/distiluse-base-multilingual-cased",
                "description": "Multilingual cased (Better Arabic)"
            }
        }
    
    def preprocess_arabic_text(self, text: str) -> str:
        """Preprocess Arabic text"""
        # Remove diacritics
        text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
        
        # Normalize Arabic letters
        text = re.sub(r'[أإآ]', 'ا', text)
        text = re.sub(r'[ىي]', 'ي', text)
        text = re.sub(r'ة', 'ه', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_model(self, model_key: str) -> bool:
        """Load model for evaluation (doesn't affect current system)"""
        try:
            if model_key in self.models:
                logger.info(f"📦 Using cached model: {model_key}")
                return True
            
            if model_key not in self.available_models:
                logger.error(f"❌ Unknown model: {model_key}")
                return False
            
            model_info = self.available_models[model_key]
            logger.info(f"🔄 Loading evaluation model: {model_info['name']}")
            
            start_time = time.time()
            model = SentenceTransformer(model_info['name'])
            load_time = time.time() - start_time
            
            # Test model
            test_embedding = model.encode(["test"])
            
            self.models[model_key] = {
                'model': model,
                'info': model_info,
                'load_time': load_time,
                'dimension': test_embedding.shape[1]
            }
            
            logger.info(f"✅ Model loaded: {model_key} (dim: {test_embedding.shape[1]}, time: {load_time:.2f}s)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_key}: {e}")
            return False
    
    def create_test_dataset(self, sample_size: int = 50) -> Dict:
        """Create test dataset from existing coupons"""
        logger.info(f"📊 Creating test dataset (size: {sample_size})")
        
        db = SessionLocal()
        try:
            # Get sample coupons
            coupons = db.query(Coupon).limit(sample_size).all()
            
            test_cases = {
                'arabic': [],
                'english': [],
                'mixed': [],
                'coupon_data': []
            }
            
            for coupon in coupons:
                category = db.query(Category).filter(Category.id == coupon.category_id).first()
                coupon_type = db.query(CouponType).filter(CouponType.id == coupon.coupon_type_id).first()
                
                # Store coupon data
                test_cases['coupon_data'].append({
                    'id': coupon.id,
                    'name': coupon.name,
                    'description': coupon.description,
                    'category': category.name if category else 'Unknown',
                    'type': coupon_type.name if coupon_type else 'Unknown',
                    'price': float(coupon.price)
                })
                
                # Arabic test cases
                test_cases['arabic'].extend([
                    f"كوبون خصم {coupon.name}",
                    f"عرض خاص على {category.name if category else 'منتجات'}",
                    f"تخفيض بقيمة {coupon.price} ريال",
                    f"كوبون {coupon_type.name if coupon_type else 'خصم'}"
                ])
                
                # English test cases
                test_cases['english'].extend([
                    f"discount coupon {coupon.name}",
                    f"special offer on {category.name if category else 'products'}",
                    f"save {coupon.price} SAR",
                    f"{coupon_type.name if coupon_type else 'discount'} coupon"
                ])
                
                # Mixed language cases
                test_cases['mixed'].extend([
                    f"كوبون discount {coupon.name}",
                    f"عرض special على {category.name if category else 'products'}",
                ])
            
            logger.info(f"✅ Test dataset created:")
            logger.info(f"   🇸🇦 Arabic cases: {len(test_cases['arabic'])}")
            logger.info(f"   🇺🇸 English cases: {len(test_cases['english'])}")
            logger.info(f"   🌍 Mixed cases: {len(test_cases['mixed'])}")
            
            return test_cases
            
        finally:
            db.close()
    
    def evaluate_single_model(self, model_key: str, test_dataset: Dict) -> Dict:
        """Evaluate a single model"""
        logger.info(f"🧪 Evaluating model: {model_key}")
        
        if not self.load_model(model_key):
            return {'success': False, 'error': f'Failed to load model {model_key}'}
        
        model_data = self.models[model_key]
        model = model_data['model']
        
        results = {
            'model_key': model_key,
            'model_info': model_data['info'],
            'dimension': model_data['dimension'],
            'load_time': model_data['load_time'],
            'success': True
        }
        
        try:
            # Test each language separately
            for lang in ['arabic', 'english', 'mixed']:
                if not test_dataset[lang]:
                    continue
                
                logger.info(f"   Testing {lang} texts...")
                
                texts = test_dataset[lang]
                
                # Preprocess Arabic texts
                if lang in ['arabic', 'mixed']:
                    texts = [self.preprocess_arabic_text(text) for text in texts]
                
                # Encode texts
                start_time = time.time()
                embeddings = model.encode(texts, batch_size=16)
                encoding_time = time.time() - start_time
                
                # Calculate similarities
                similarity_matrix = cosine_similarity(embeddings)
                
                # Remove diagonal (self-similarity)
                mask = np.ones(similarity_matrix.shape, dtype=bool)
                np.fill_diagonal(mask, False)
                similarities = similarity_matrix[mask]
                
                # Calculate metrics
                results[f'{lang}_metrics'] = {
                    'total_texts': len(texts),
                    'encoding_time': encoding_time,
                    'encoding_speed': len(texts) / encoding_time,
                    'avg_similarity': float(np.mean(similarities)),
                    'max_similarity': float(np.max(similarities)),
                    'min_similarity': float(np.min(similarities)),
                    'std_similarity': float(np.std(similarities)),
                    'median_similarity': float(np.median(similarities))
                }
                
                logger.info(f"     ✅ {lang}: avg={np.mean(similarities):.3f}, speed={len(texts)/encoding_time:.1f} texts/sec")
            
            # Overall performance rating
            arabic_score = results.get('arabic_metrics', {}).get('avg_similarity', 0)
            english_score = results.get('english_metrics', {}).get('avg_similarity', 0)
            mixed_score = results.get('mixed_metrics', {}).get('avg_similarity', 0)
            
            overall_score = (arabic_score + english_score + mixed_score) / 3
            results['overall_score'] = float(overall_score)
            results['performance_rating'] = self._rate_performance(overall_score)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {model_key}: {e}")
            return {'success': False, 'error': str(e)}
    
    def compare_models(self, model_keys: List[str], sample_size: int = 50) -> Dict:
        """Compare multiple models"""
        logger.info(f"🔬 Starting model comparison: {', '.join(model_keys)}")
        
        # Create test dataset
        test_dataset = self.create_test_dataset(sample_size)
        
        comparison_results = {
            'timestamp': datetime.now().isoformat(),
            'test_dataset_size': sample_size,
            'models_compared': model_keys,
            'results': {},
            'summary': {}
        }
        
        # Evaluate each model
        for model_key in model_keys:
            logger.info(f"\n{'='*50}")
            logger.info(f"🎯 Evaluating: {model_key}")
            logger.info(f"{'='*50}")
            
            result = self.evaluate_single_model(model_key, test_dataset)
            comparison_results['results'][model_key] = result
        
        # Generate summary
        comparison_results['summary'] = self._generate_comparison_summary(comparison_results['results'])
        
        # Save results
        self.evaluation_results[datetime.now().isoformat()] = comparison_results
        
        return comparison_results
    
    def _rate_performance(self, score: float) -> str:
        """Rate performance based on similarity score"""
        if score >= 0.7:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.5:
            return "Fair"
        elif score >= 0.4:
            return "Poor"
        else:
            return "Very Poor"
    
    def _generate_comparison_summary(self, results: Dict) -> Dict:
        """Generate comparison summary"""
        summary = {
            'best_overall': None,
            'best_arabic': None,
            'best_english': None,
            'fastest_encoding': None,
            'recommendations': []
        }
        
        # Find best performers
        best_overall_score = 0
        best_arabic_score = 0
        best_english_score = 0
        fastest_speed = 0
        
        for model_key, result in results.items():
            if not result.get('success'):
                continue
            
            # Overall performance
            overall_score = result.get('overall_score', 0)
            if overall_score > best_overall_score:
                best_overall_score = overall_score
                summary['best_overall'] = {
                    'model': model_key,
                    'score': overall_score,
                    'rating': result.get('performance_rating')
                }
            
            # Arabic performance
            arabic_score = result.get('arabic_metrics', {}).get('avg_similarity', 0)
            if arabic_score > best_arabic_score:
                best_arabic_score = arabic_score
                summary['best_arabic'] = {
                    'model': model_key,
                    'score': arabic_score
                }
            
            # English performance
            english_score = result.get('english_metrics', {}).get('avg_similarity', 0)
            if english_score > best_english_score:
                best_english_score = english_score
                summary['best_english'] = {
                    'model': model_key,
                    'score': english_score
                }
            
            # Encoding speed
            avg_speed = np.mean([
                result.get('arabic_metrics', {}).get('encoding_speed', 0),
                result.get('english_metrics', {}).get('encoding_speed', 0)
            ])
            if avg_speed > fastest_speed:
                fastest_speed = avg_speed
                summary['fastest_encoding'] = {
                    'model': model_key,
                    'speed': avg_speed
                }
        
        # Generate recommendations
        if summary['best_overall']:
            if summary['best_overall']['score'] > 0.6:
                summary['recommendations'].append(f"✅ {summary['best_overall']['model']} shows excellent overall performance")
            else:
                summary['recommendations'].append(f"⚠️ All models show moderate performance - consider fine-tuning")
        
        if summary['best_arabic'] and summary['best_english']:
            if summary['best_arabic']['model'] != summary['best_english']['model']:
                summary['recommendations'].append(f"🌍 Different models excel in different languages - consider ensemble approach")
        
        return summary
    
    def save_results(self, filename: str = None):
        """Save evaluation results to file"""
        if not filename:
            filename = f"/tmp/model_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 Results saved to: {filename}")
        return filename

def main():
    """Main evaluation function"""
    logger.info("🚀 Starting Model Evaluation (Non-Intrusive)")
    
    evaluator = ModelEvaluator()
    
    # Compare all available models
    models_to_compare = ["current", "multilingual", "multilingual_cased"]
    
    try:
        results = evaluator.compare_models(models_to_compare, sample_size=30)
        
        # Print summary
        print("\n" + "="*80)
        print("📊 MODEL COMPARISON SUMMARY")
        print("="*80)
        
        summary = results['summary']
        
        if summary['best_overall']:
            print(f"🏆 Best Overall: {summary['best_overall']['model']} (score: {summary['best_overall']['score']:.3f})")
        
        if summary['best_arabic']:
            print(f"🇸🇦 Best Arabic: {summary['best_arabic']['model']} (score: {summary['best_arabic']['score']:.3f})")
        
        if summary['best_english']:
            print(f"🇺🇸 Best English: {summary['best_english']['model']} (score: {summary['best_english']['score']:.3f})")
        
        if summary['fastest_encoding']:
            print(f"⚡ Fastest: {summary['fastest_encoding']['model']} ({summary['fastest_encoding']['speed']:.1f} texts/sec)")
        
        print("\n💡 Recommendations:")
        for rec in summary['recommendations']:
            print(f"   {rec}")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        for model_key, result in results['results'].items():
            if result.get('success'):
                print(f"\n🔹 {model_key}:")
                print(f"   Overall Score: {result.get('overall_score', 0):.3f} ({result.get('performance_rating', 'Unknown')})")
                
                if 'arabic_metrics' in result:
                    ar = result['arabic_metrics']
                    print(f"   Arabic: {ar['avg_similarity']:.3f} ({ar['encoding_speed']:.1f} texts/sec)")
                
                if 'english_metrics' in result:
                    en = result['english_metrics']
                    print(f"   English: {en['avg_similarity']:.3f} ({en['encoding_speed']:.1f} texts/sec)")
        
        # Save results
        filename = evaluator.save_results()
        print(f"\n💾 Full results saved to: {filename}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")

if __name__ == "__main__":
    main()
