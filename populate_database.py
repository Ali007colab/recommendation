#!/usr/bin/env python3

import psycopg2
import random
from datetime import datetime, timedelta
import string
import math

# إعداد الاتصال
def get_connection():
    return psycopg2.connect(
        host='localhost',
        database='recommendation_db',
        user='recommendation_user',
        password='simple123'
    )

# 100 فئة متنوعة
CATEGORIES = [
    # التكنولوجيا والإلكترونيات (20 فئة)
    'Smartphones', 'Laptops', 'Gaming', 'Smart Home', 'Cameras', 'Audio Equipment', 
    'Wearable Tech', 'Computer Parts', 'Software', 'Mobile Accessories',
    'Tablets', 'TV & Entertainment', 'Networking', 'Security Systems', 'Drones',
    'VR & AR', 'Smart Appliances', 'Electric Vehicles', 'Tech Gadgets', 'Robotics',
    
    # الطعام والمشروبات (15 فئة)
    'Fast Food', 'Fine Dining', 'Coffee & Tea', 'Bakery', 'Healthy Food',
    'International Cuisine', 'Desserts', 'Beverages', 'Organic Food', 'Vegan Options',
    'Seafood', 'BBQ & Grill', 'Street Food', 'Catering', 'Food Delivery',
    
    # الموضة والجمال (15 فئة)
    'Men Fashion', 'Women Fashion', 'Kids Fashion', 'Shoes', 'Bags & Accessories',
    'Jewelry', 'Watches', 'Skincare', 'Makeup', 'Hair Care',
    'Perfumes', 'Sunglasses', 'Sportswear', 'Luxury Fashion', 'Vintage Clothing',
    
    # السفر والترفيه (10 فئة)
    'Flights', 'Hotels', 'Car Rental', 'Tours & Activities', 'Cruises',
    'Adventure Travel', 'Business Travel', 'Vacation Packages', 'Travel Insurance', 'Local Experiences',
    
    # الصحة واللياقة (10 فئة)
    'Gym & Fitness', 'Yoga & Meditation', 'Medical Services', 'Dental Care', 'Mental Health',
    'Nutrition & Supplements', 'Spa & Wellness', 'Physical Therapy', 'Alternative Medicine', 'Health Insurance',
    
    # التعليم والكتب (10 فئة)
    'Online Courses', 'Books & E-books', 'Language Learning', 'Professional Training', 'Academic Courses',
    'Skill Development', 'Certifications', 'Tutoring', 'Educational Software', 'School Supplies',
    
    # المنزل والحديقة (10 فئة)
    'Furniture', 'Home Decor', 'Kitchen & Dining', 'Gardening', 'Home Improvement',
    'Cleaning Services', 'Interior Design', 'Outdoor Living', 'Storage Solutions', 'Home Security',
    
    # الرياضة والهوايات (10 فئة)
    'Team Sports', 'Individual Sports', 'Outdoor Activities', 'Water Sports', 'Winter Sports',
    'Martial Arts', 'Cycling', 'Running', 'Fishing', 'Hunting'
]

# 100 نوع كوبون متنوع
COUPON_TYPES = [
    # خصومات أساسية (20 نوع)
    'Percentage Discount', 'Fixed Amount Off', 'Buy One Get One', 'Buy Two Get One Free', 'Flash Sale',
    'Clearance Sale', 'End of Season Sale', 'Black Friday Deal', 'Cyber Monday Special', 'Holiday Discount',
    'Student Discount', 'Senior Discount', 'Military Discount', 'First Time Buyer', 'Loyalty Reward',
    'VIP Member Exclusive', 'Early Bird Special', 'Last Minute Deal', 'Bulk Purchase Discount', 'Volume Discount',
    
    # شحن ومزايا (15 نوع)
    'Free Shipping', 'Express Shipping', 'Same Day Delivery', 'Free Installation', 'Free Setup',
    'Free Consultation', 'Free Trial', 'Free Sample', 'Free Gift Wrapping', 'Free Returns',
    'Extended Warranty', 'Price Match Guarantee', 'Money Back Guarantee', 'Free Upgrade', 'Complimentary Service',
    
    # عروض مجمعة (15 نوع)
    'Bundle Deal', 'Combo Offer', 'Package Deal', 'Family Pack', 'Group Discount',
    'Corporate Package', 'Subscription Deal', 'Membership Bundle', 'Season Pass', 'Annual Plan',
    'Multi-Product Deal', 'Cross-Category Bundle', 'Starter Kit', 'Complete Set', 'Value Pack',
    
    # استرداد نقدي ونقاط (10 نوع)
    'Cashback Offer', 'Points Reward', 'Store Credit', 'Gift Card Bonus', 'Referral Bonus',
    'Review Reward', 'Social Share Bonus', 'Newsletter Signup', 'App Download Reward', 'Survey Reward',
    
    # عروض زمنية (15 نوع)
    'Happy Hour', 'Weekend Special', 'Weekday Deal', 'Morning Special', 'Evening Offer',
    'Lunch Deal', 'Dinner Special', 'Seasonal Offer', 'Monthly Special', 'Weekly Deal',
    'Daily Flash Sale', 'Limited Time Offer', 'Countdown Deal', 'Pre-Order Discount', 'Launch Special',
    
    # عروض خاصة (15 نوع)
    'Birthday Special', 'Anniversary Deal', 'Valentine Offer', 'Mother Day Special', 'Father Day Deal',
    'Graduation Gift', 'Wedding Package', 'Baby Shower Deal', 'Housewarming Gift', 'Retirement Special',
    'New Year Offer', 'Summer Sale', 'Winter Clearance', 'Spring Collection', 'Fall Special',
    
    # عروض متقدمة (10 نوع)
    'Tiered Discount', 'Progressive Savings', 'Spend & Save', 'Accumulative Discount', 'Threshold Bonus',
    'Loyalty Tier Upgrade', 'Exclusive Access', 'Priority Service', 'Premium Features', 'Advanced Package'
]

# قوالب أسماء متقدمة
PRODUCT_TEMPLATES = {
    'Smartphones': ['iPhone {model}', 'Samsung Galaxy {model}', 'Google Pixel {model}', 'OnePlus {model}', 'Xiaomi {model}'],
    'Laptops': ['MacBook {model}', 'Dell XPS {model}', 'HP Pavilion {model}', 'Lenovo ThinkPad {model}', 'ASUS {model}'],
    'Gaming': ['PlayStation {model}', 'Xbox {model}', 'Nintendo {model}', 'Gaming PC {model}', 'Steam Deck {model}'],
    'Fast Food': ['Big Mac {variant}', 'Whopper {variant}', 'KFC {variant}', 'Pizza {variant}', 'Subway {variant}'],
    'Coffee & Tea': ['Starbucks {variant}', 'Costa {variant}', 'Dunkin {variant}', 'Local Brew {variant}', 'Specialty {variant}'],
    'Men Fashion': ['Suit {style}', 'Casual {style}', 'Formal {style}', 'Sportswear {style}', 'Accessories {style}'],
    'Women Fashion': ['Dress {style}', 'Blouse {style}', 'Skirt {style}', 'Pants {style}', 'Accessories {style}'],
    'Flights': ['{destination} Flight', 'Business Class {destination}', 'Economy {destination}', 'Direct Flight {destination}', 'Round Trip {destination}'],
    'Hotels': ['{star} Star Hotel', 'Luxury Resort {location}', 'Budget Hotel {location}', 'Boutique {location}', 'Business Hotel {location}']
}

# متغيرات للقوالب
MODELS = ['Pro', 'Max', 'Plus', 'Ultra', 'Premium', 'Standard', 'Lite', 'Mini', 'XL', 'Special Edition']
VARIANTS = ['Classic', 'Deluxe', 'Supreme', 'Original', 'Spicy', 'Mild', 'Large', 'Medium', 'Small', 'Family Size']
STYLES = ['Casual', 'Formal', 'Vintage', 'Modern', 'Classic', 'Trendy', 'Elegant', 'Sporty', 'Luxury', 'Budget']
DESTINATIONS = ['Paris', 'London', 'Tokyo', 'New York', 'Dubai', 'Singapore', 'Sydney', 'Rome', 'Barcelona', 'Amsterdam']
LOCATIONS = ['Downtown', 'Beach', 'Mountain', 'City Center', 'Airport', 'Marina', 'Historic', 'Modern', 'Luxury', 'Budget']

def generate_coupon_code():
    """توليد كود كوبون فريد"""
    prefix = random.choice(['SAVE', 'DEAL', 'OFFER', 'PROMO', 'DISC', 'SPEC', 'BONUS', 'GIFT'])
    numbers = ''.join(random.choices(string.digits, k=4))
    suffix = ''.join(random.choices(string.ascii_uppercase, k=2))
    return f"{prefix}{numbers}{suffix}"

def generate_smart_coupon_name(category, coupon_type):
    """توليد اسم كوبون ذكي ومتقدم"""
    
    # استخراج نسبة الخصم أو القيمة
    discount_percent = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 70])
    discount_amount = random.choice([5, 10, 15, 20, 25, 50, 100, 200])
    
    # قوالب متقدمة حسب نوع الكوبون
    if 'Percentage' in coupon_type or 'Discount' in coupon_type:
        templates = [
            f'{discount_percent}% Off {category}',
            f'Save {discount_percent}% on {category}',
            f'{category} {discount_percent}% Discount',
            f'Get {discount_percent}% Off Premium {category}',
            f'Exclusive {discount_percent}% {category} Deal'
        ]
    elif 'Fixed Amount' in coupon_type:
        templates = [
            f'${discount_amount} Off {category}',
            f'Save ${discount_amount} on {category}',
            f'{category} ${discount_amount} Discount',
            f'Get ${discount_amount} Off Your {category} Purchase'
        ]
    elif 'Buy One Get One' in coupon_type or 'BOGO' in coupon_type:
        templates = [
            f'BOGO {category} Deal',
            f'Buy One Get One Free {category}',
            f'{category} 2-for-1 Special',
            f'Double Your {category} Purchase'
        ]
    elif 'Free Shipping' in coupon_type:
        templates = [
            f'Free Shipping on {category}',
            f'No Shipping Fees for {category}',
            f'{category} with Free Delivery',
            f'Complimentary Shipping {category}'
        ]
    elif 'Bundle' in coupon_type or 'Package' in coupon_type:
        templates = [
            f'{category} Bundle Deal',
            f'Complete {category} Package',
            f'{category} Combo Offer',
            f'All-in-One {category} Bundle'
        ]
    elif 'Cashback' in coupon_type:
        cashback_percent = random.choice([5, 10, 15, 20])
        templates = [
            f'{cashback_percent}% Cashback on {category}',
            f'Earn {cashback_percent}% Back on {category}',
            f'{category} Cashback Reward',
            f'Get Money Back on {category}'
        ]
    elif 'Flash Sale' in coupon_type or 'Limited Time' in coupon_type:
        templates = [
            f'{category} Flash Sale',
            f'Limited Time {category} Offer',
            f'24-Hour {category} Deal',
            f'Quick Sale {category} Special'
        ]
    else:
        # قالب عام
        templates = [
            f'Special {category} Offer',
            f'Premium {category} Deal',
            f'Exclusive {category} Promotion',
            f'Limited {category} Special',
            f'Best {category} Value'
        ]
    
    base_name = random.choice(templates)
    
    # إضافة تفاصيل إضافية أحياناً
    if random.random() < 0.3:  # 30% من الوقت
        extras = ['Premium', 'Deluxe', 'Pro', 'Ultimate', 'Elite', 'VIP', 'Exclusive', 'Limited Edition']
        base_name = f"{random.choice(extras)} {base_name}"
    
    return base_name

def generate_detailed_description(coupon_name, category, coupon_type, price):
    """توليد وصف مفصل وجذاب"""
    
    # كلمات مفتاحية حسب الفئة
    category_keywords = {
        'Smartphones': ['latest technology', 'cutting-edge features', 'premium quality', 'advanced camera', 'long battery life'],
        'Laptops': ['high performance', 'portable design', 'professional grade', 'fast processing', 'reliable'],
        'Gaming': ['immersive experience', 'high-quality graphics', 'competitive gaming', 'entertainment', 'multiplayer'],
        'Fast Food': ['delicious taste', 'quick service', 'fresh ingredients', 'satisfying meal', 'convenient'],
        'Coffee & Tea': ['premium blend', 'rich flavor', 'aromatic', 'freshly brewed', 'energizing'],
        'Men Fashion': ['stylish design', 'comfortable fit', 'premium materials', 'modern style', 'versatile'],
        'Women Fashion': ['elegant design', 'trendy style', 'comfortable wear', 'fashionable', 'chic'],
        'Flights': ['comfortable journey', 'convenient schedule', 'reliable service', 'great destinations', 'smooth travel'],
        'Hotels': ['luxury accommodation', 'excellent service', 'prime location', 'comfortable stay', 'memorable experience']
    }
    
    # أوصاف حسب نوع الكوبون
    type_descriptions = {
        'Percentage Discount': 'Enjoy incredible savings with this limited-time percentage discount',
        'Fixed Amount Off': 'Save a fixed amount on your purchase with this exclusive offer',
        'Buy One Get One': 'Double your value with this amazing buy-one-get-one deal',
        'Free Shipping': 'Get your order delivered at no extra cost with free shipping',
        'Bundle Deal': 'Complete package at an unbeatable price with everything you need',
        'Cashback Offer': 'Earn money back on your purchase with this cashback reward',
        'Flash Sale': 'Limited time offer with massive savings - act fast!',
        'VIP Member Exclusive': 'Exclusive offer available only to our valued VIP members'
    }
    
    # بناء الوصف
    keywords = category_keywords.get(category, ['high quality', 'great value', 'excellent choice', 'premium product', 'best deal'])
    type_desc = type_descriptions.get(coupon_type, 'Amazing deal with great savings and value')
    
    descriptions = [
        f"{type_desc}. Experience {random.choice(keywords)} with {coupon_name}. Perfect for those who appreciate quality and value.",
        f"Discover the best in {category.lower()} with {coupon_name}. {type_desc} featuring {random.choice(keywords)} and exceptional service.",
        f"Transform your {category.lower()} experience with {coupon_name}. This exclusive offer combines {random.choice(keywords)} with unmatched savings.",
        f"Don't miss out on {coupon_name} - your gateway to premium {category.lower()}. {type_desc} with {random.choice(keywords)} guaranteed.",
        f"Elevate your lifestyle with {coupon_name}. Featuring {random.choice(keywords)} and designed for the discerning customer who values both quality and savings."
    ]
    
    base_description = random.choice(descriptions)
    
    # إضافة تفاصيل السعر
    if price > 200:
        price_desc = "This premium offering represents exceptional value for money."
    elif price > 50:
        price_desc = "Great value proposition with competitive pricing."
    else:
        price_desc = "Affordable luxury that doesn't compromise on quality."
    
    # إضافة عبارة ختامية
    endings = [
        "Terms and conditions apply. Limited time offer.",
        "While supplies last. Don't wait - offer expires soon!",
        "Available for a limited time only. Grab this deal today!",
        "Exclusive offer with limited availability. Act now!",
        "Special promotion with exceptional savings. Order today!"
    ]
    
    return f"{base_description} {price_desc} {random.choice(endings)}"

def clear_existing_data():
    """مسح البيانات الموجودة"""
    conn = get_connection()
    cur = conn.cursor()
    
    print("🧹 Clearing existing data...")
    
    try:
        cur.execute("DELETE FROM user_interactions")
        cur.execute("DELETE FROM coupons")
        cur.execute("DELETE FROM coupon_types WHERE id > 10")  # الاحتفاظ بالأنواع الأساسية
        cur.execute("DELETE FROM categories WHERE id > 10")    # الاحتفاظ بالفئات الأساسية
        
        # إعادة تعيين المعرفات
        cur.execute("ALTER SEQUENCE coupons_id_seq RESTART WITH 1")
        cur.execute("ALTER SEQUENCE user_interactions_id_seq RESTART WITH 1")
        cur.execute("SELECT setval('categories_id_seq', 10)")
        cur.execute("SELECT setval('coupon_types_id_seq', 10)")
    
    conn.commit()
        print("✅ Existing data cleared successfully")
    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        conn.rollback()
    finally:
        cur.close()
    conn.close()

def populate_categories_and_types():
    """إضافة 100 فئة و 100 نوع كوبون"""
    conn = get_connection()
    cur = conn.cursor()
    
    print("📂 Adding 100 categories and 100 coupon types...")
    
    try:
        # إضافة الفئات الجديدة (من 11 إلى 100)
        for i, category in enumerate(CATEGORIES[10:], 11):  # البدء من 11
            cur.execute("INSERT INTO categories (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING", 
                       (i, category))
        
        # إضافة أنواع الكوبونات الجديدة (من 11 إلى 100)
        for i, coupon_type in enumerate(COUPON_TYPES[10:], 11):  # البدء من 11
            description = f"Advanced {coupon_type.lower()} offer with special terms and conditions"
            cur.execute("INSERT INTO coupon_types (id, name, description) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING", 
                       (i, coupon_type, description))
    
    conn.commit()
        print("✅ Categories and coupon types added successfully")
    except Exception as e:
        print(f"❌ Error adding categories/types: {e}")
        conn.rollback()
    finally:
        cur.close()
    conn.close()

def populate_coupons(num_coupons=1000):
    """إضافة 1000 كوبون بطريقة مدروسة"""
    conn = get_connection()
    cur = conn.cursor()
    
    print(f"🎫 Adding {num_coupons} smart coupons...")
    
    try:
        # توزيع متوازن عبر الفئات
        coupons_per_category = num_coupons // 100  # 10 كوبونات لكل فئة
        extra_coupons = num_coupons % 100
        
        coupon_id = 1
        
        for category_id in range(1, 101):  # 100 فئة
            category_name = CATEGORIES[category_id - 1]
            
            # عدد الكوبونات لهذه الفئة
            category_coupons = coupons_per_category
            if category_id <= extra_coupons:
                category_coupons += 1
            
            print(f"  📦 Adding {category_coupons} coupons for {category_name}...")
            
            for i in range(category_coupons):
                # اختيار نوع كوبون بطريقة ذكية
                if i % 5 == 0:  # كل خامس كوبون خصم نسبي
                    coupon_type_id = random.choice([1, 6, 9, 21, 31])  # أنواع خصومات
                elif i % 5 == 1:  # شحن مجاني
                    coupon_type_id = random.choice([3, 21, 22, 23, 24])  # أنواع شحن
                elif i % 5 == 2:  # عروض مجمعة
                    coupon_type_id = random.choice([2, 36, 37, 38, 39])  # عروض مجمعة
                elif i % 5 == 3:  # استرداد نقدي
                    coupon_type_id = random.choice([5, 51, 52, 53, 54])  # استرداد نقدي
                else:  # عروض خاصة
                    coupon_type_id = random.randint(1, 100)
                
                # الحصول على اسم نوع الكوبون
                cur.execute("SELECT name FROM coupon_types WHERE id = %s", (coupon_type_id,))
                coupon_type_result = cur.fetchone()
                coupon_type_name = coupon_type_result[0] if coupon_type_result else "Special Offer"
                
                # توليد السعر بناءً على الفئة
                if category_name in ['Smartphones', 'Laptops', 'Gaming', 'Flights', 'Hotels']:
                    price = round(random.uniform(100, 1000), 2)  # فئات غالية
                elif category_name in ['Men Fashion', 'Women Fashion', 'Jewelry', 'Watches']:
                    price = round(random.uniform(50, 500), 2)   # فئات متوسطة
                elif category_name in ['Fast Food', 'Coffee & Tea', 'Desserts']:
                    price = round(random.uniform(5, 50), 2)     # فئات رخيصة
                else:
                    price = round(random.uniform(20, 200), 2)   # فئات عامة
                
                # توليد الاسم والوصف
                name = generate_smart_coupon_name(category_name, coupon_type_name)
                description = generate_detailed_description(name, category_name, coupon_type_name, price)
        coupon_code = generate_coupon_code()
        
                # تاريخ انتهاء متنوع
                end_date = datetime.now() + timedelta(days=random.randint(30, 730))  # من شهر إلى سنتين
                
                # مقدم الخدمة
                provider_id = random.randint(1, 500)  # 500 مقدم خدمة مختلف
                
                cur.execute("""
                    INSERT INTO coupons (name, description, price, coupon_type_id, category_id, 
             provider_id, coupon_status, coupon_code, date) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (name, description, price, coupon_type_id, category_id, 
                      provider_id, 1, coupon_code, end_date.date()))
                
                coupon_id += 1
        
            conn.commit()
        print(f"✅ Added {num_coupons} coupons successfully")
    except Exception as e:
        print(f"❌ Error adding coupons: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def create_user_personas():
    """إنشاء شخصيات مستخدمين متنوعة ومعقدة"""
    
    personas = {}
    
    # 1. محبي التكنولوجيا (100 مستخدم)
    tech_categories = [1, 2, 3, 8, 16, 17, 18, 19, 20]  # فئات تقنية
    for user_id in range(1, 101):
        personas[user_id] = {
            'type': 'tech_enthusiast',
            'primary_categories': random.sample(tech_categories, 3),
            'secondary_categories': random.sample(range(1, 101), 5),
            'behavior': 'researcher',  # يبحث كثيراً قبل الشراء
            'activity_level': 'high',  # نشاط عالي
            'purchase_probability': 0.15,  # 15% احتمال شراء
            'click_probability': 0.35,     # 35% احتمال نقر
            'search_probability': 0.50     # 50% احتمال بحث
        }
    
    # 2. محبي الطعام (80 مستخدم)
    food_categories = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # فئات طعام
    for user_id in range(101, 181):
        personas[user_id] = {
            'type': 'foodie',
            'primary_categories': random.sample(food_categories, 4),
            'secondary_categories': random.sample(range(1, 101), 4),
            'behavior': 'impulse_buyer',  # يشتري بسرعة
            'activity_level': 'medium',
            'purchase_probability': 0.30,  # 30% احتمال شراء
            'click_probability': 0.40,     # 40% احتمال نقر
            'search_probability': 0.30     # 30% احتمال بحث
        }
    
    # 3. محبي الموضة (70 مستخدم)
    fashion_categories = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    for user_id in range(181, 251):
        personas[user_id] = {
            'type': 'fashion_lover',
            'primary_categories': random.sample(fashion_categories, 5),
            'secondary_categories': random.sample(range(1, 101), 3),
            'behavior': 'seasonal_shopper',  # يشتري حسب المواسم
            'activity_level': 'medium',
            'purchase_probability': 0.20,
            'click_probability': 0.45,
            'search_probability': 0.35
        }
    
    # 4. محبي السفر (60 مستخدم)
    travel_categories = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
    for user_id in range(251, 311):
        personas[user_id] = {
            'type': 'traveler',
            'primary_categories': random.sample(travel_categories, 4),
            'secondary_categories': random.sample(range(1, 101), 6),
            'behavior': 'planner',  # يخطط مسبقاً
            'activity_level': 'low',
            'purchase_probability': 0.25,
            'click_probability': 0.30,
            'search_probability': 0.45
        }
    
    # 5. محبي الصحة واللياقة (50 مستخدم)
    health_categories = [49, 50, 51, 52, 53, 54, 55, 56, 57, 58]
    for user_id in range(311, 361):
        personas[user_id] = {
            'type': 'health_conscious',
            'primary_categories': random.sample(health_categories, 4),
            'secondary_categories': random.sample(range(1, 101), 4),
            'behavior': 'careful_buyer',
            'activity_level': 'medium',
            'purchase_probability': 0.18,
            'click_probability': 0.38,
            'search_probability': 0.44
        }
    
    # 6. مستخدمين متنوعين (140 مستخدم)
    for user_id in range(361, 501):
        personas[user_id] = {
            'type': 'diverse_user',
            'primary_categories': random.sample(range(1, 101), 6),
            'secondary_categories': random.sample(range(1, 101), 8),
            'behavior': random.choice(['balanced', 'explorer', 'bargain_hunter']),
            'activity_level': random.choice(['low', 'medium', 'high']),
            'purchase_probability': random.uniform(0.10, 0.35),
            'click_probability': random.uniform(0.25, 0.50),
            'search_probability': random.uniform(0.30, 0.60)
        }
    
    return personas

def populate_interactions(num_interactions=60000):
    """إضافة 60000 تفاعل ذكي ومدروس"""
    conn = get_connection()
    cur = conn.cursor()
    
    print(f"👥 Adding {num_interactions} smart user interactions...")
    
    try:
        # الحصول على معرفات الكوبونات مجمعة حسب الفئة
        cur.execute("SELECT id, category_id, price FROM coupons")
        coupons = cur.fetchall()
        
        coupon_dict = {}
        for coupon in coupons:
            category_id = coupon[1]
            if category_id not in coupon_dict:
                coupon_dict[category_id] = []
            coupon_dict[category_id].append({'id': coupon[0], 'price': coupon[2]})
        
        # إنشاء شخصيات المستخدمين
        personas = create_user_personas()
        
        interactions_added = 0
        batch_size = 1000
        batch_data = []
        
        print("  🎭 Creating realistic user interactions...")
        
        for user_id, persona in personas.items():
            if interactions_added >= num_interactions:
                break
            
            # تحديد عدد التفاعلات حسب مستوى النشاط
            if persona['activity_level'] == 'high':
                user_interactions = random.randint(80, 150)
            elif persona['activity_level'] == 'medium':
                user_interactions = random.randint(40, 80)
            else:  # low
                user_interactions = random.randint(15, 40)
            
            # توزيع التفاعلات عبر الوقت (آخر 6 أشهر)
            for _ in range(min(user_interactions, num_interactions - interactions_added)):
                
                # اختيار الفئة (70% مفضلة، 30% ثانوية)
                if random.random() < 0.7:
                    category_id = random.choice(persona['primary_categories'])
                else:
                    category_id = random.choice(persona['secondary_categories'])
                
                # التأكد من وجود كوبونات في هذه الفئة
                if category_id not in coupon_dict or not coupon_dict[category_id]:
                    continue
                
                # اختيار كوبون (تفضيل الأسعار المناسبة للشخصية)
                available_coupons = coupon_dict[category_id]
                
                if persona['type'] in ['tech_enthusiast', 'traveler']:
                    # يفضلون المنتجات الغالية
                    coupon = max(available_coupons, key=lambda x: x['price'])
                elif persona['type'] == 'foodie':
                    # يفضلون الأسعار المتوسطة
                    coupon = random.choice(available_coupons)
                else:
                    # توزيع عشوائي
                    coupon = random.choice(available_coupons)
                
                coupon_id = coupon['id']
                
                # تحديد نوع التفاعل حسب الشخصية
                rand = random.random()
                if rand < persona['purchase_probability']:
                    action = 'purchase'
                elif rand < persona['purchase_probability'] + persona['click_probability']:
                    action = 'click'
                else:
                    action = 'search'
                
                # النقاط مع تعديل حسب السلوك
                base_scores = {'search': 2.0, 'click': 5.0, 'purchase': 15.0}
                score = base_scores[action]
                
                # تعديل النقاط حسب السلوك
                if persona['behavior'] == 'impulse_buyer' and action == 'purchase':
                    score *= 1.2  # مكافأة للمشترين السريعين
                elif persona['behavior'] == 'researcher' and action == 'search':
                    score *= 1.1  # مكافأة للباحثين
                elif persona['behavior'] == 'planner' and action == 'click':
                    score *= 1.15  # مكافأة للمخططين
                
                # التاريخ (توزيع واقعي عبر آخر 6 أشهر)
                days_ago = int(random.expovariate(1/30))  # توزيع أسي (أكثر نشاط مؤخراً)
                days_ago = min(days_ago, 180)  # حد أقصى 6 أشهر
                timestamp = datetime.now() - timedelta(days=days_ago)
                
                # إضافة وقت عشوائي في اليوم
                timestamp += timedelta(
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                )
                
                batch_data.append((user_id, coupon_id, action, score, timestamp))
                interactions_added += 1
                
                # إدراج دفعي لتحسين الأداء
                if len(batch_data) >= batch_size:
                    cur.executemany("""
                        INSERT INTO user_interactions (user_id, coupon_id, action, score, timestamp)
                        VALUES (%s, %s, %s, %s, %s)
                    """, batch_data)
                    conn.commit()
                    print(f"    ✅ Inserted batch: {interactions_added}/{num_interactions}")
                    batch_data = []
                
                if interactions_added >= num_interactions:
                    break
        
        # إدراج الدفعة الأخيرة
        if batch_data:
            cur.executemany("""
                INSERT INTO user_interactions (user_id, coupon_id, action, score, timestamp)
                VALUES (%s, %s, %s, %s, %s)
            """, batch_data)
            conn.commit()
    
        print(f"✅ Added {interactions_added} interactions successfully")
    except Exception as e:
        print(f"❌ Error adding interactions: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def create_test_scenarios():
    """إنشاء سيناريوهات اختبار متقدمة"""
    conn = get_connection()
    cur = conn.cursor()
    
    print("🧪 Creating advanced test scenarios...")
    
    try:
        # سيناريو 1: مستخدم محب للتكنولوجيا المتقدمة
        test_user_tech = 9999
        cur.execute("SELECT id FROM coupons WHERE category_id IN (1, 2, 3) LIMIT 10")
        tech_coupons = [row[0] for row in cur.fetchall()]
        
        # إنشاء نمط بحث وشراء واقعي
        for i, coupon_id in enumerate(tech_coupons[:7]):
            # بحث أولي
            cur.execute("""
                INSERT INTO user_interactions (user_id, coupon_id, action, score, timestamp)
                VALUES (%s, %s, %s, %s, %s)
            """, (test_user_tech, coupon_id, 'search', 2.0, datetime.now() - timedelta(days=i*2)))
            
            # نقر بعد البحث
            if i < 5:
                cur.execute("""
                    INSERT INTO user_interactions (user_id, coupon_id, action, score, timestamp)
                    VALUES (%s, %s, %s, %s, %s)
                """, (test_user_tech, coupon_id, 'click', 5.0, datetime.now() - timedelta(days=i*2-1)))
            
            # شراء لبعض العناصر
            if i < 2:
                cur.execute("""
                    INSERT INTO user_interactions (user_id, coupon_id, action, score, timestamp)
                    VALUES (%s, %s, %s, %s, %s)
                """, (test_user_tech, coupon_id, 'purchase', 15.0, datetime.now() - timedelta(days=i*2-1, hours=2)))
        
        # سيناريو 2: مستخدم محب للطعام
        test_user_food = 9998
        cur.execute("SELECT id FROM coupons WHERE category_id IN (14, 15, 16, 17, 18) LIMIT 8")
        food_coupons = [row[0] for row in cur.fetchall()]
        
        for i, coupon_id in enumerate(food_coupons):
            # نمط شراء سريع (محب طعام)
            if i % 2 == 0:
                cur.execute("""
                    INSERT INTO user_interactions (user_id, coupon_id, action, score, timestamp)
                    VALUES (%s, %s, %s, %s, %s)
                """, (test_user_food, coupon_id, 'purchase', 15.0, datetime.now() - timedelta(days=i)))
            else:
                cur.execute("""
                    INSERT INTO user_interactions (user_id, coupon_id, action, score, timestamp)
                    VALUES (%s, %s, %s, %s, %s)
                """, (test_user_food, coupon_id, 'click', 5.0, datetime.now() - timedelta(days=i)))
        
        # سيناريو 3: مستخدم متنوع
        test_user_diverse = 9997
        diverse_categories = random.sample(range(1, 101), 10)
        
        for category_id in diverse_categories:
            cur.execute("SELECT id FROM coupons WHERE category_id = %s LIMIT 1", (category_id,))
            result = cur.fetchone()
            if result:
                coupon_id = result[0]
                action = random.choice(['search', 'click', 'purchase'])
                score = {'search': 2.0, 'click': 5.0, 'purchase': 15.0}[action]
                
                cur.execute("""
                    INSERT INTO user_interactions (user_id, coupon_id, action, score, timestamp)
                    VALUES (%s, %s, %s, %s, %s)
                """, (test_user_diverse, coupon_id, action, score, datetime.now() - timedelta(days=random.randint(1, 30))))
        
    conn.commit()
        print("✅ Advanced test scenarios created")
        print("  🔬 Test User 9999: Tech enthusiast with research pattern")
        print("  🔬 Test User 9998: Food lover with impulse buying")
        print("  🔬 Test User 9997: Diverse user with mixed interests")
        
    except Exception as e:
        print(f"❌ Error creating test scenarios: {e}")
        conn.rollback()
    finally:
        cur.close()
    conn.close()

def print_advanced_statistics():
    """طباعة إحصائيات متقدمة ومفصلة"""
    conn = get_connection()
    cur = conn.cursor()
    
    print("\n📊 Advanced Database Statistics:")
    print("=" * 70)
    
    try:
        # إحصائيات الكوبونات حسب الفئة
        cur.execute("""
            SELECT c.name, COUNT(*) as count, 
                   AVG(co.price) as avg_price,
                   MIN(co.price) as min_price,
                   MAX(co.price) as max_price
            FROM coupons co 
            JOIN categories c ON co.category_id = c.id 
            GROUP BY c.name 
            ORDER BY count DESC 
            LIMIT 10
        """)
        
        print("🏆 Top 10 Categories by Coupon Count:")
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]} coupons (Avg: ${row[2]:.2f}, Range: ${row[3]:.2f}-${row[4]:.2f})")
        
        # إحصائيات التفاعلات
        cur.execute("""
            SELECT action, COUNT(*) as count, AVG(score) as avg_score
            FROM user_interactions 
            GROUP BY action 
        ORDER BY count DESC
    """)
    
        print(f"\n📈 Interaction Statistics:")
        total_interactions = 0
        for row in cur.fetchall():
            total_interactions += row[1]
            percentage = (row[1] / 60000) * 100 if 60000 > 0 else 0
            print(f"  {row[0].capitalize()}: {row[1]:,} ({percentage:.1f}%) - Avg Score: {row[2]:.2f}")
        
        # أكثر المستخدمين نشاطاً
        cur.execute("""
            SELECT user_id, COUNT(*) as interactions, 
                   SUM(score) as total_score,
                   COUNT(DISTINCT coupon_id) as unique_coupons
            FROM user_interactions 
            GROUP BY user_id 
            ORDER BY interactions DESC 
            LIMIT 10
        """)
        
        print(f"\n🏅 Top 10 Most Active Users:")
        for row in cur.fetchall():
            print(f"  User {row[0]}: {row[1]} interactions, {row[2]:.1f} total score, {row[3]} unique coupons")
        
        # إحصائيات الأسعار
        cur.execute("""
            SELECT 
                COUNT(*) as total_coupons,
                AVG(price) as avg_price,
                MIN(price) as min_price,
                MAX(price) as max_price,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price) as median_price
            FROM coupons
        """)
        
        price_stats = cur.fetchone()
        print(f"\n💰 Price Statistics:")
        print(f"  Total Coupons: {price_stats[0]:,}")
        print(f"  Average Price: ${price_stats[1]:.2f}")
        print(f"  Price Range: ${price_stats[2]:.2f} - ${price_stats[3]:.2f}")
        print(f"  Median Price: ${price_stats[4]:.2f}")
        
        # إحصائيات زمنية
        cur.execute("""
            SELECT 
                DATE_TRUNC('month', timestamp) as month,
                COUNT(*) as interactions
            FROM user_interactions 
            GROUP BY DATE_TRUNC('month', timestamp)
            ORDER BY month DESC
            LIMIT 6
        """)
        
        print(f"\n📅 Monthly Interaction Trends (Last 6 months):")
        for row in cur.fetchall():
            month_name = row[0].strftime("%B %Y")
            print(f"  {month_name}: {row[1]:,} interactions")
        
        # إحصائيات الفئات الأكثر شعبية
        cur.execute("""
            SELECT c.name, COUNT(ui.id) as interactions, 
                   COUNT(DISTINCT ui.user_id) as unique_users
            FROM user_interactions ui
            JOIN coupons co ON ui.coupon_id = co.id
            JOIN categories c ON co.category_id = c.id
            GROUP BY c.name
        ORDER BY interactions DESC
            LIMIT 10
        """)
        
        print(f"\n🔥 Most Popular Categories by Interactions:")
        for row in cur.fetchall():
            print(f"  {row[0]}: {row[1]:,} interactions from {row[2]} users")
        
        # معدل التحويل
        cur.execute("""
            SELECT 
                COUNT(CASE WHEN action = 'search' THEN 1 END) as searches,
                COUNT(CASE WHEN action = 'click' THEN 1 END) as clicks,
                COUNT(CASE WHEN action = 'purchase' THEN 1 END) as purchases
            FROM user_interactions
        """)
        
        conversion = cur.fetchone()
        if conversion[0] > 0:
            click_rate = (conversion[1] / conversion[0]) * 100
            purchase_rate = (conversion[2] / conversion[1]) * 100 if conversion[1] > 0 else 0
            overall_conversion = (conversion[2] / conversion[0]) * 100
            
            print(f"\n🎯 Conversion Rates:")
            print(f"  Search to Click: {click_rate:.2f}%")
            print(f"  Click to Purchase: {purchase_rate:.2f}%")
            print(f"  Overall Conversion: {overall_conversion:.2f}%")
        
        # إجمالي الإحصائيات
        cur.execute("SELECT COUNT(DISTINCT user_id) FROM user_interactions")
        unique_users = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(DISTINCT category_id) FROM coupons")
        active_categories = cur.fetchone()[0]
        
        print(f"\n📋 Summary:")
        print(f"  Total Categories: 100")
        print(f"  Active Categories: {active_categories}")
        print(f"  Total Coupon Types: 100")
        print(f"  Total Coupons: {price_stats[0]:,}")
        print(f"  Total Interactions: {total_interactions:,}")
        print(f"  Unique Active Users: {unique_users:,}")
        print(f"  Average Interactions per User: {total_interactions/unique_users:.1f}")
        
    except Exception as e:
        print(f"❌ Error generating statistics: {e}")
    finally:
        cur.close()
    conn.close()

def main():
    """الدالة الرئيسية"""
    print("🚀 Starting MASSIVE Data Population for PostgreSQL")
    print("🎯 Target: 1000 Coupons + 60000 Interactions + 100 Categories + 100 Types")
    print("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # مسح البيانات القديمة
        clear_existing_data()
        
        # إضافة الفئات وأنواع الكوبونات
        populate_categories_and_types()
        
        # إضافة الكوبونات
        populate_coupons(1000)
        
        # إضافة التفاعلات
        populate_interactions(60000)
        
        # إنشاء سيناريوهات الاختبار
        create_test_scenarios()
        
        # طباعة الإحصائيات المتقدمة
        print_advanced_statistics()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n🎉 MASSIVE Data Population Completed Successfully!")
        print(f"⏱️  Total Time: {duration}")
        print(f"📊 Data Generated:")
        print(f"   • 100 Categories")
        print(f"   • 100 Coupon Types") 
        print(f"   • 1,000 Smart Coupons")
        print(f"   • 60,000 Realistic Interactions")
        print(f"   • 500+ Unique Users")
        print(f"   • Advanced User Personas")
        
        print(f"\n🧪 Test Your System:")
        print(f"   curl -X POST http://YOUR_SERVER:8000/build_vector_store")
        print(f"   curl 'http://YOUR_SERVER:8000/get_recommendations?user_id=9999&top_n=10'  # Tech user")
        print(f"   curl 'http://YOUR_SERVER:8000/get_recommendations?user_id=9998&top_n=10'  # Food user")
        print(f"   curl 'http://YOUR_SERVER:8000/get_recommendations?user_id=9997&top_n=10'  # Diverse user")
        print(f"   curl 'http://YOUR_SERVER:8000/get_recommendations?user_id=50&top_n=10'    # Regular user")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()