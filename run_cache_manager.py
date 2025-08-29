#!/usr/bin/env python3

import sys
import os
import argparse
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cache_recommendations import RecommendationCacheManager

def main():
    parser = argparse.ArgumentParser(description='Recommendation Cache Manager')
    parser.add_argument('--action', choices=['cache', 'stats', 'daemon'], 
                       default='cache', help='Action to perform')
    parser.add_argument('--min-interactions', type=int, default=3, 
                       help='Minimum user interactions')
    parser.add_argument('--max-users', type=int, default=1000, 
                       help='Maximum users to process')
    parser.add_argument('--top-n', type=int, default=15, 
                       help='Number of recommendations per user')
    
    args = parser.parse_args()
    
    print(f"🚀 Recommendation Cache Manager - {datetime.now()}")
    print(f"Action: {args.action}")
    
    # Initialize cache manager
    cache_manager = RecommendationCacheManager()
    
    if args.action == 'cache':
        print(f"📊 Caching recommendations for up to {args.max_users} users...")
        print(f"   Min interactions: {args.min_interactions}")
        print(f"   Recommendations per user: {args.top_n}")
        
        success = cache_manager.cache_all_users(
            min_interactions=args.min_interactions,
            max_users=args.max_users,
            top_n=args.top_n
        )
        
        if success:
            print("✅ Caching completed successfully")
        else:
            print("❌ Caching failed")
    
    elif args.action == 'stats':
        stats = cache_manager.get_cache_stats()
        print("📊 Cache Statistics:")
        print(f"   Status: {stats.get('status')}")
        print(f"   Total cached users: {stats.get('total_cached_users', 0)}")
        print(f"   Last model update: {stats.get('last_model_update', 'Never')}")
        print(f"   Last cache update: {stats.get('last_cache_update', 'Never')}")
        print(f"   Cache TTL: {stats.get('cache_ttl_hours', 0)} hours")
        
        if stats.get('sample_cached_users'):
            print("   Sample cached users:")
            for user in stats['sample_cached_users']:
                print(f"     User {user['user_id']}: {user['coupon_count']} coupons ({user['method']})")
    
    elif args.action == 'daemon':
        print("🔄 Starting daemon mode...")
        from cache_recommendations import main as daemon_main
        daemon_main()

if __name__ == "__main__":
    main()
