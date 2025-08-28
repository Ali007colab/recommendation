#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import get_db_session, UserInteraction, Coupon, Category, CouponType

def clear_database():
    """تفريغ قاعدة البيانات بالكامل"""
    db = get_db_session()
    
    try:
        print("🗑️ بدء تفريغ قاعدة البيانات...")
        
        # عد السجلات قبل الحذف
        interactions_count = db.query(UserInteraction).count()
        coupons_count = db.query(Coupon).count()
        types_count = db.query(CouponType).count()
        categories_count = db.query(Category).count()
        
        print(f"📊 السجلات الحالية:")
        print(f"   التفاعلات: {interactions_count}")
        print(f"   الكوبونات: {coupons_count}")
        print(f"   أنواع الكوبونات: {types_count}")
        print(f"   الفئات: {categories_count}")
        
        # حذف البيانات بالترتيب الصحيح
        print("\n🔥 حذف التفاعلات...")
        db.query(UserInteraction).delete()
        
        print("🔥 حذف الكوبونات...")
        db.query(Coupon).delete()
        
        print("🔥 حذف أنواع الكوبونات...")
        db.query(CouponType).delete()
        
        print("🔥 حذف الفئات...")
        db.query(Category).delete()
        
        # حفظ التغييرات
        db.commit()
        
        print("✅ تم تفريغ قاعدة البيانات بنجاح!")
        print(f"✅ تم حذف {interactions_count + coupons_count + types_count + categories_count} سجل")
        
    except Exception as e:
        print(f"❌ خطأ في تفريغ قاعدة البيانات: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    clear_database()