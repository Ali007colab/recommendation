#!/usr/bin/env python3

import redis
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List
import argparse
from tabulate import tabulate

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config

class RedisMonitor:
    def __init__(self):
        self.redis_client = None
        self.init_redis()
    
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
            print("✅ Redis connected successfully")
            
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            self.redis_client = None
    
    def get_all_recommendation_keys(self):
        """الحصول على كل مفاتيح التوصيات"""
        if not self.redis_client:
            return []
        
        try:
            pattern = "rec:user:*"
            keys = self.redis_client.keys(pattern)
            return sorted(keys)
        except Exception as e:
            print(f"Error getting keys: {e}")
            return []
    
    def get_user_recommendations(self, user_id: int):
        """الحصول على توصيات مستخدم محدد"""
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"rec:user:{user_id}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return data
            
            return None
            
        except Exception as e:
            print(f"Error getting user {user_id} recommendations: {e}")
            return None
    
    def get_cache_statistics(self):
        """إحصائيات شاملة للـ Cache"""
        if not self.redis_client:
            return None
        
        try:
            keys = self.get_all_recommendation_keys()
            
            if not keys:
                return {
                    "total_users": 0,
                    "message": "No cached recommendations found"
                }
            
            # Sample data analysis
            sample_size = min(100, len(keys))
            sample_keys = keys[:sample_size]
            
            methods = {}
            categories_count = {}
            interaction_counts = []
            coupon_counts = []
            cache_ages = []
            
            for key in sample_keys:
                try:
                    data_str = self.redis_client.get(key)
                    if not data_str:
                        continue
                    
                    data = json.loads(data_str)
                    
                    # Method distribution
                    method = data.get('method', 'unknown')
                    methods[method] = methods.get(method, 0) + 1
                    
                    # Coupon count distribution
                    coupon_count = len(data.get('coupon_ids', []))
                    coupon_counts.append(coupon_count)
                    
                    # Interaction count
                    interaction_count = data.get('interaction_count', 0)
                    interaction_counts.append(interaction_count)
                    
                    # Categories
                    user_categories = data.get('user_categories', {})
                    for category in user_categories.keys():
                        categories_count[category] = categories_count.get(category, 0) + 1
                    
                    # Cache age
                    cached_at = data.get('cached_at')
                    if cached_at:
                        cached_time = datetime.fromisoformat(cached_at.replace('Z', '+00:00').replace('+00:00', ''))
                        age_hours = (datetime.now() - cached_time).total_seconds() / 3600
                        cache_ages.append(age_hours)
                
                except Exception as e:
                    continue
            
            # Calculate statistics
            avg_coupons = sum(coupon_counts) / len(coupon_counts) if coupon_counts else 0
            avg_interactions = sum(interaction_counts) / len(interaction_counts) if interaction_counts else 0
            avg_cache_age = sum(cache_ages) / len(cache_ages) if cache_ages else 0
            
            # Top categories
            top_categories = sorted(categories_count.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "total_users": len(keys),
                "sampled_users": len(sample_keys),
                "methods_distribution": methods,
                "average_coupons_per_user": round(avg_coupons, 2),
                "average_interactions_per_user": round(avg_interactions, 2),
                "average_cache_age_hours": round(avg_cache_age, 2),
                "top_categories": top_categories,
                "coupon_count_range": {
                    "min": min(coupon_counts) if coupon_counts else 0,
                    "max": max(coupon_counts) if coupon_counts else 0
                },
                "interaction_count_range": {
                    "min": min(interaction_counts) if interaction_counts else 0,
                    "max": max(interaction_counts) if interaction_counts else 0
                }
            }
            
        except Exception as e:
            print(f"Error getting cache statistics: {e}")
            return None
    
    def search_users_by_criteria(self, min_interactions=None, method=None, has_category=None, limit=50):
        """البحث عن المستخدمين حسب معايير محددة"""
        if not self.redis_client:
            return []
        
        keys = self.get_all_recommendation_keys()
        matching_users = []
        
        for key in keys:
            try:
                data_str = self.redis_client.get(key)
                if not data_str:
                    continue
                
                data = json.loads(data_str)
                user_id = data.get('user_id')
                
                # Apply filters
                if min_interactions and data.get('interaction_count', 0) < min_interactions:
                    continue
                
                if method and data.get('method') != method:
                    continue
                
                if has_category:
                    user_categories = data.get('user_categories', {})
                    if has_category not in user_categories:
                        continue
                
                matching_users.append({
                    'user_id': user_id,
                    'coupon_count': len(data.get('coupon_ids', [])),
                    'method': data.get('method'),
                    'interaction_count': data.get('interaction_count', 0),
                    'cached_at': data.get('cached_at'),
                    'top_categories': list(data.get('user_categories', {}).keys())[:3]
                })
                
                if len(matching_users) >= limit:
                    break
                
            except Exception as e:
                continue
        
        return matching_users
    
    def get_redis_info(self):
        """معلومات Redis العامة"""
        if not self.redis_client:
            return None
        
        try:
            info = self.redis_client.info()
            
            return {
                "redis_version": info.get('redis_version'),
                "used_memory_human": info.get('used_memory_human'),
                "used_memory_peak_human": info.get('used_memory_peak_human'),
                "connected_clients": info.get('connected_clients'),
                "total_commands_processed": info.get('total_commands_processed'),
                "keyspace_hits": info.get('keyspace_hits'),
                "keyspace_misses": info.get('keyspace_misses'),
                "uptime_in_days": info.get('uptime_in_days'),
                "role": info.get('role')
            }
            
        except Exception as e:
            print(f"Error getting Redis info: {e}")
            return None
    
    def cleanup_expired_keys(self):
        """تنظيف المفاتيح المنتهية الصلاحية"""
        if not self.redis_client:
            return 0
        
        keys = self.get_all_recommendation_keys()
        cleaned_count = 0
        
        for key in keys:
            try:
                ttl = self.redis_client.ttl(key)
                if ttl == -2:  # Key doesn't exist
                    cleaned_count += 1
                elif ttl == -1:  # Key exists but has no expiry
                    # Set expiry for keys without TTL
                    self.redis_client.expire(key, config.RECOMMENDATIONS_CACHE_TTL)
            except Exception as e:
                continue
        
        return cleaned_count
    
    def export_cache_data(self, output_file="cache_export.json"):
        """تصدير بيانات الـ Cache إلى ملف"""
        if not self.redis_client:
            return False
        
        try:
            keys = self.get_all_recommendation_keys()
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_users": len(keys),
                "users": {}
            }
            
            for key in keys:
                try:
                    data_str = self.redis_client.get(key)
                    if data_str:
                        data = json.loads(data_str)
                        user_id = str(data.get('user_id'))
                        export_data["users"][user_id] = data
                except Exception as e:
                    continue
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Cache data exported to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error exporting cache data: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Redis Cache Monitor')
    parser.add_argument('--action', choices=['stats', 'user', 'search', 'info', 'cleanup', 'export', 'list'], 
                       default='stats', help='Action to perform')
    parser.add_argument('--user-id', type=int, help='User ID to inspect')
    parser.add_argument('--min-interactions', type=int, help='Minimum interactions filter')
    parser.add_argument('--method', help='Method filter (popular, smart_content_based)')
    parser.add_argument('--category', help='Category filter')
    parser.add_argument('--limit', type=int, default=50, help='Result limit')
    parser.add_argument('--output', default='cache_export.json', help='Export output file')
    
    args = parser.parse_args()
    
    print(f"🔍 Redis Cache Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    monitor = RedisMonitor()
    
    if args.action == 'stats':
        print("📊 Cache Statistics:")
        stats = monitor.get_cache_statistics()
        if stats:
            print(f"   Total Users: {stats['total_users']:,}")
            print(f"   Average Coupons per User: {stats['average_coupons_per_user']}")
            print(f"   Average Interactions per User: {stats['average_interactions_per_user']}")
            print(f"   Average Cache Age: {stats['average_cache_age_hours']:.1f} hours")
            
            print("\n📈 Methods Distribution:")
            for method, count in stats['methods_distribution'].items():
                percentage = (count / stats['sampled_users']) * 100
                print(f"   {method}: {count} ({percentage:.1f}%)")
            
            print("\n🏆 Top Categories:")
            for category, count in stats['top_categories']:
                print(f"   {category}: {count} users")
        else:
            print("   No statistics available")
    
    elif args.action == 'user':
        if not args.user_id:
            print("❌ Please provide --user-id")
            return
        
        print(f"👤 User {args.user_id} Recommendations:")
        user_data = monitor.get_user_recommendations(args.user_id)
        if user_data:
            print(f"   Method: {user_data.get('method')}")
            print(f"   Coupon Count: {len(user_data.get('coupon_ids', []))}")
            print(f"   Interaction Count: {user_data.get('interaction_count')}")
            print(f"   Cached At: {user_data.get('cached_at')}")
            print(f"   Expires At: {user_data.get('expires_at')}")
            
            print("\n🎯 Recommended Coupon IDs:")
            coupon_ids = user_data.get('coupon_ids', [])
            for i, coupon_id in enumerate(coupon_ids[:10], 1):
                print(f"   {i:2d}. Coupon ID: {coupon_id}")
            
            if len(coupon_ids) > 10:
                print(f"   ... and {len(coupon_ids) - 10} more")
            
            print("\n📂 User Categories:")
            categories = user_data.get('user_categories', {})
            for category, score in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {category}: {score:.1f}")
        else:
            print("   User not found in cache")
    
    elif args.action == 'search':
        print("🔍 Searching Users:")
        print(f"   Filters: min_interactions={args.min_interactions}, method={args.method}, category={args.category}")
        
        users = monitor.search_users_by_criteria(
            min_interactions=args.min_interactions,
            method=args.method,
            has_category=args.category,
            limit=args.limit
        )
        
        if users:
            headers = ['User ID', 'Coupons', 'Method', 'Interactions', 'Top Categories']
            table_data = []
            
            for user in users:
                table_data.append([
                    user['user_id'],
                    user['coupon_count'],
                    user['method'],
                    user['interaction_count'],
                    ', '.join(user['top_categories'])
                ])
            
            print(tabulate(table_data, headers=headers, tablefmt='grid'))
            print(f"\nFound {len(users)} matching users")
        else:
            print("   No matching users found")
    
    elif args.action == 'info':
        print("ℹ️ Redis Server Info:")
        info = monitor.get_redis_info()
        if info:
            for key, value in info.items():
                print(f"   {key}: {value}")
        else:
            print("   Unable to get Redis info")
    
    elif args.action == 'cleanup':
        print("🧹 Cleaning up expired keys...")
        cleaned = monitor.cleanup_expired_keys()
        print(f"   Cleaned {cleaned} expired keys")
    
    elif args.action == 'export':
        print(f"📤 Exporting cache data to {args.output}...")
        success = monitor.export_cache_data(args.output)
        if not success:
            print("   Export failed")
    
    elif args.action == 'list':
        print("📋 Cached Users List:")
        keys = monitor.get_all_recommendation_keys()
        
        if keys:
            user_ids = []
            for key in keys[:args.limit]:
                try:
                    user_id = int(key.split(':')[-1])
                    user_ids.append(user_id)
                except:
                    continue
            
            user_ids.sort()
            
            # Display in columns
            cols = 10
            for i in range(0, len(user_ids), cols):
                row = user_ids[i:i+cols]
                print("   " + " ".join(f"{uid:6d}" for uid in row))
            
            print(f"\nTotal: {len(user_ids)} cached users")
        else:
            print("   No cached users found")

if __name__ == "__main__":
    main()
