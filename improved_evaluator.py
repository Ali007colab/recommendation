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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from database import SessionLocal, Coupon, Category, CouponType
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedModelEvaluator:
    def __init__(self):
        self.models = {}
        self.available_models = {
            "current": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "description": "Current model (English focused)"
            },
            "multilingual": {
                "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "description": "Multilingual model"
            }
        }
    
    def load_model(self, model_key: str) -> bool:
        """Load model for evaluation"""
        try:
            if model_key in self.models:
                return True
            
            model_info = self.available_models[model_key]
            logger.info(f"🔄 Loading: {model_info['name']}")
            
            start_time = time.time()
            model = SentenceTransformer(model_info['name'])
            load_time = time.time() - start_time
            
            self.models[model_key] = {
                'model': model,
                'info': model_info,
                'load_time': load_time
            }
            
            logger.info(f"✅ Loaded: {model_key} ({load_time:.2f}s)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_key}: {e}")
            return False
    
    def create_realistic_test_dataset(self, sample_size: int = 30) -> Dict:
        """Create realistic test dataset based on actual data"""
        logger.info(f"📊 Creating realistic test dataset")
        
        db = SessionLocal()
        try:
            coupons = db.query(Coupon).limit(sample_size).all()
            
            test_cases = {
                'similar_pairs': [],      # Should have HIGH similarity
                'different_pairs': [],    # Should have LOW similarity
                'category_groups': {},    # Same category should be similar
                'original_texts': []      # Original coupon texts
            }
            
            for coupon in coupons:
                category = db.query(Category).filter(Category.id == coupon.category_id).first()
                coupon_type = db.query(CouponType).filter(CouponType.id == coupon.coupon_type_id).first()
                
                # Original coupon text (as it exists)
                original_text = f"{coupon.name} - {coupon.description}"
                test_cases['original_texts'].append({
                    'text': original_text,
                    'category': category.name if category else 'Unknown',
                    'type': coupon_type.name if coupon_type else 'Unknown',
                    'price': float(coupon.price)
                })
                
                # Group by category for similarity testing
                cat_name = category.name if category else 'Unknown'
                if cat_name not in test_cases['category_groups']:
                    test_cases['category_groups'][cat_name] = []
                test_cases['category_groups'][cat_name].append(original_text)
            
            # Create similar pairs (same category)
            for category, texts in test_cases['category_groups'].items():
                if len(texts) >= 2:
                    for i in range(len(texts)-1):
                        test_cases['similar_pairs'].append((texts[i], texts[i+1]))
            
            # Create different pairs (different categories)
            categories = list(test_cases['category_groups'].keys())
            if len(categories) >= 2:
                for i, cat1 in enumerate(categories):
                    for cat2 in categories[i+1:]:
                        if test_cases['category_groups'][cat1] and test_cases['category_groups'][cat2]:
                            text1 = test_cases['category_groups'][cat1][0]
                            text2 = test_cases['category_groups'][cat2][0]
                            test_cases['different_pairs'].append((text1, text2))
            
            logger.info(f"✅ Test dataset created:")
            logger.info(f"   📄 Original texts: {len(test_cases['original_texts'])}")
            logger.info(f"   ✅ Similar pairs: {len(test_cases['similar_pairs'])}")
            logger.info(f"   ❌ Different pairs: {len(test_cases['different_pairs'])}")
            logger.info(f"   📂 Categories: {len(test_cases['category_groups'])}")
            
            return test_cases
            
        finally:
            db.close()
    
    def evaluate_model_realistic(self, model_key: str, test_dataset: Dict) -> Dict:
        """Evaluate model with realistic expectations"""
        logger.info(f"🧪 Evaluating {model_key} with realistic tests")
        
        if not self.load_model(model_key):
            return {'success': False, 'error': f'Failed to load {model_key}'}
        
        model = self.models[model_key]['model']
        
        results = {
            'model_key': model_key,
            'success': True,
            'tests': {}
        }
        
        try:
            # Test 1: Similar pairs should have HIGH similarity
            if test_dataset['similar_pairs']:
                logger.info("   Testing similar pairs...")
                similarities = []
                
                for text1, text2 in test_dataset['similar_pairs']:
                    emb1 = model.encode([text1])
                    emb2 = model.encode([text2])
                    sim = cosine_similarity(emb1, emb2)[0][0]
                    similarities.append(sim)
                
                results['tests']['similar_pairs'] = {
                    'count': len(similarities),
                    'avg_similarity': float(np.mean(similarities)),
                    'min_similarity': float(np.min(similarities)),
                    'max_similarity': float(np.max(similarities)),
                    'expectation': 'HIGH (>0.5)',
                    'performance': 'Good' if np.mean(similarities) > 0.5 else 'Poor'
                }
            
            # Test 2: Different pairs should have LOWER similarity
            if test_dataset['different_pairs']:
                logger.info("   Testing different pairs...")
                similarities = []
                
                for text1, text2 in test_dataset['different_pairs']:
                    emb1 = model.encode([text1])
                    emb2 = model.encode([text2])
                    sim = cosine_similarity(emb1, emb2)[0][0]
                    similarities.append(sim)
                
                results['tests']['different_pairs'] = {
                    'count': len(similarities),
                    'avg_similarity': float(np.mean(similarities)),
                    'min_similarity': float(np.min(similarities)),
                    'max_similarity': float(np.max(similarities)),
                    'expectation': 'LOWER than similar pairs',
                    'performance': 'Good' if np.mean(similarities) < results['tests']['similar_pairs']['avg_similarity'] else 'Poor'
                }
            
            # Test 3: Category clustering
            logger.info("   Testing category clustering...")
            category_scores = {}
            
            for category, texts in test_dataset['category_groups'].items():
                if len(texts) >= 2:
                    embeddings = model.encode(texts)
                    sim_matrix = cosine_similarity(embeddings)
                    
                    # Average similarity within category (excluding diagonal)
                    mask = np.ones(sim_matrix.shape, dtype=bool)
                    np.fill_diagonal(mask, False)
                    avg_sim = np.mean(sim_matrix[mask])
                    
                    category_scores[category] = float(avg_sim)
            
            results['tests']['category_clustering'] = {
                'categories': category_scores,
                'avg_within_category': float(np.mean(list(category_scores.values()))) if category_scores else 0,
                'expectation': 'Items in same category should be similar',
                'performance': 'Good' if np.mean(list(category_scores.values())) > 0.3 else 'Poor' if category_scores else 'No data'
            }
            
            # Test 4: Encoding speed
            logger.info("   Testing encoding speed...")
            all_texts = [item['text'] for item in test_dataset['original_texts']]
            
            start_time = time.time()
            embeddings = model.encode(all_texts)
            encoding_time = time.time() - start_time
            
            results['tests']['performance'] = {
                'total_texts': len(all_texts),
                'encoding_time': encoding_time,
                'texts_per_second': len(all_texts) / encoding_time,
                'expectation': '>20 texts/sec',
                'performance': 'Good' if (len(all_texts) / encoding_time) > 20 else 'Slow'
            }
            
            # Overall assessment
            good_tests = sum(1 for test in results['tests'].values() 
                           if isinstance(test, dict) and test.get('performance') == 'Good')
            total_tests = len(results['tests'])
            
            results['overall_score'] = good_tests / total_tests if total_tests > 0 else 0
            results['overall_rating'] = self._rate_realistic_performance(results['overall_score'])
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {model_key}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _rate_realistic_performance(self, score: float) -> str:
        """Rate performance realistically"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def compare_models_realistic(self, model_keys: List[str], sample_size: int = 30) -> Dict:
        """Compare models with realistic expectations"""
        logger.info(f"🔬 Realistic model comparison: {', '.join(model_keys)}")
        
        test_dataset = self.create_realistic_test_dataset(sample_size)
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'realistic_evaluation',
            'models': {},
            'summary': {}
        }
        
        for model_key in model_keys:
            logger.info(f"\n{'='*40}")
            logger.info(f"🎯 Testing: {model_key}")
            logger.info(f"{'='*40}")
            
            result = self.evaluate_model_realistic(model_key, test_dataset)
            comparison['models'][model_key] = result
        
        # Generate summary
        comparison['summary'] = self._generate_realistic_summary(comparison['models'])
        
        return comparison
    
    def _generate_realistic_summary(self, results: Dict) -> Dict:
        """Generate realistic summary"""
        summary = {
            'best_model': None,
            'key_findings': [],
            'recommendations': []
        }
        
        best_score = 0
        for model_key, result in results.items():
            if result.get('success') and result.get('overall_score', 0) > best_score:
                best_score = result['overall_score']
                summary['best_model'] = {
                    'model': model_key,
                    'score': best_score,
                    'rating': result.get('overall_rating')
                }
        
        # Key findings
        for model_key, result in results.items():
            if not result.get('success'):
                continue
            
            tests = result.get('tests', {})
            
            # Similar pairs performance
            if 'similar_pairs' in tests:
                sim_perf = tests['similar_pairs']
                if sim_perf['avg_similarity'] > 0.5:
                    summary['key_findings'].append(f"✅ {model_key}: Good at finding similar items ({sim_perf['avg_similarity']:.3f})")
                else:
                    summary['key_findings'].append(f"⚠️ {model_key}: Weak at finding similar items ({sim_perf['avg_similarity']:.3f})")
            
            # Speed performance
            if 'performance' in tests:
                speed = tests['performance']['texts_per_second']
                if speed > 50:
                    summary['key_findings'].append(f"⚡ {model_key}: Fast encoding ({speed:.1f} texts/sec)")
                elif speed < 20:
                    summary['key_findings'].append(f"🐌 {model_key}: Slow encoding ({speed:.1f} texts/sec)")
        
        # Recommendations
        if summary['best_model']:
            if summary['best_model']['score'] > 0.6:
                summary['recommendations'].append(f"✅ Use {summary['best_model']['model']} - shows good performance")
            else:
                summary['recommendations'].append("⚠️ All models show moderate performance - consider fine-tuning")
        
        return summary

def main():
    """Main realistic evaluation"""
    logger.info("🚀 Starting Realistic Model Evaluation")
    
    evaluator = ImprovedModelEvaluator()
    
    results = evaluator.compare_models_realistic(["current", "multilingual"], sample_size=20)
    
    # Print results
    print("\n" + "="*60)
    print("📊 REALISTIC MODEL COMPARISON")
    print("="*60)
    
    for model_key, result in results['models'].items():
        if result.get('success'):
            print(f"\n🔹 {model_key.upper()}:")
            print(f"   Overall Rating: {result.get('overall_rating')} ({result.get('overall_score', 0):.2f})")
            
            tests = result.get('tests', {})
            
            if 'similar_pairs' in tests:
                sp = tests['similar_pairs']
                print(f"   Similar Items: {sp['avg_similarity']:.3f} ({sp['performance']})")
            
            if 'different_pairs' in tests:
                dp = tests['different_pairs']
                print(f"   Different Items: {dp['avg_similarity']:.3f} ({dp['performance']})")
            
            if 'performance' in tests:
                perf = tests['performance']
                print(f"   Speed: {perf['texts_per_second']:.1f} texts/sec ({perf['performance']})")
    
    # Summary
    summary = results['summary']
    if summary['best_model']:
        print(f"\n🏆 Best Model: {summary['best_model']['model']} ({summary['best_model']['rating']})")
    
    print(f"\n💡 Key Findings:")
    for finding in summary['key_findings']:
        print(f"   {finding}")
    
    print(f"\n🎯 Recommendations:")
    for rec in summary['recommendations']:
        print(f"   {rec}")

if __name__ == "__main__":
    main()
