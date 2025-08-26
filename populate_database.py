#!/usr/bin/env python3

import sqlite3
import random
from datetime import datetime, timedelta
import string

# إعداد قاعدة بيانات SQLite
def get_connection():
    conn = sqlite3.connect('recommendation.db')
    return conn

def create_tables():
    """إنشاء الجداول في SQLite"""
    conn = get_connection()
    cur = conn.cursor()
    
    # إنشاء الجداول
    cur.execute('''
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        )
    ''')
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS coupon_types (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT
        )
    ''')
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS coupons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            price REAL NOT NULL,
            coupon_type_id INTEGER,
            category_id INTEGER,
            provider_id INTEGER NOT NULL,
            coupon_status INTEGER DEFAULT 1,
            coupon_code TEXT,
            date TEXT DEFAULT CURRENT_DATE,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (coupon_type_id) REFERENCES coupon_types(id),
            FOREIGN KEY (category_id) REFERENCES categories(id)
        )
    ''')
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS user_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            coupon_id INTEGER,
            action TEXT NOT NULL,
            score REAL NOT NULL,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (coupon_id) REFERENCES coupons(id)
        )
    ''')
    
    # إدراج البيانات الأساسية
    categories = [
        (1, 'Electronics'), (2, 'Food'), (3, 'Fashion'), (4, 'Travel'), (5, 'Entertainment'),
        (6, 'Health & Beauty'), (7, 'Sports & Fitness'), (8, 'Books & Education'), 
        (9, 'Home & Garden'), (10, 'Automotive')
    ]
    
    coupon_types = [
        (1, 'Discount', 'Percentage discount on single item'),
        (2, 'Bundle Deal', 'Special pricing for multiple items'),
        (3, 'Free Shipping', 'Free shipping on orders'),
        (4, 'Buy One Get One', 'Buy one item, get another free'),
        (5, 'Cashback', 'Get cashback on purchase'),
        (6, 'Flash Sale', 'Limited time discount offer'),
        (7, 'Loyalty Reward', 'Reward for loyal customers'),
        (8, 'First Time Buyer', 'Special offer for new customers'),
        (9, 'Seasonal Sale', 'Seasonal discount promotion'),
        (10, 'Bulk Purchase', 'Discount for buying in bulk')
    ]
    
    cur.executemany('INSERT OR IGNORE INTO categories VALUES (?, ?)', categories)
    cur.executemany('INSERT OR IGNORE INTO coupon_types VALUES (?, ?, ?)', coupon_types)
    
    conn.commit()
    cur.close()
    conn.close()

# بيانات أساسية للتوليد
CATEGORIES = [
    {'id': 1, 'name': 'Electronics'},
    {'id': 2, 'name': 'Food'},
    {'id': 3, 'name': 'Fashion'},
    {'id': 4, 'name': 'Travel'},
    {'id': 5, 'name': 'Entertainment'},
    {'id': 6, 'name': 'Health & Beauty'},
    {'id': 7, 'name': 'Sports & Fitness'},
    {'id': 8, 'name': 'Books & Education'},
    {'id': 9, 'name': 'Home & Garden'},
    {'id': 10, 'name': 'Automotive'}
]

COUPON_TYPES = [
    {'id': 1, 'name': 'Discount'},
    {'id': 2, 'name': 'Bundle Deal'},
    {'id': 3, 'name': 'Free Shipping'},
    {'id': 4, 'name': 'Buy One Get One'},
    {'id': 5, 'name': 'Cashback'},
    {'id': 6, 'name': 'Flash Sale'},
    {'id': 7, 'name': 'Loyalty Reward'},
    {'id': 8, 'name': 'First Time Buyer'},
    {'id': 9, 'name': 'Seasonal Sale'},
    {'id': 10, 'name': 'Bulk Purchase'}
]

# قوالب أسماء الكوبونات حسب الفئة
COUPON_TEMPLATES = {
    'Electronics': [
        '{percent}% Off {product}', 'Buy {product} Get Free {accessory}', 
        '{product} Flash Sale', 'Premium {product} Bundle', '{product} Cashback Deal',
        'Limited {product} Offer', '{product} Student Discount', 'Refurbished {product} Sale'
    ],
    'Food': [
        '{percent}% Off {item}', 'Free Delivery on {item}', '{item} Combo Deal',
        'Buy 2 Get 1 {item}', '{item} Happy Hour', 'Family {item} Pack',
        '{item} Weekend Special', 'Healthy {item} Options'
    ],
    'Fashion': [
        '{percent}% Off {item}', 'Designer {item} Sale', '{item} Collection',
        'Buy 2 Get 1 {item}', 'Seasonal {item} Clearance', 'Premium {item} Deal',
        '{item} Fashion Week', 'Vintage {item} Sale'
    ],
    'Travel': [
        '{percent}% Off {destination}', 'Free {service} with Booking', '{destination} Package',
        'Last Minute {destination}', '{destination} Adventure', 'Luxury {destination} Deal',
        '{destination} Family Package', 'Business {destination} Travel'
    ],
    'Entertainment': [
        '{percent}% Off {event}', 'Free Popcorn with {event}', '{event} VIP Experience',
        'Group {event} Discount', '{event} Season Pass', 'Premium {event} Access',
        '{event} Student Deal', 'Weekend {event} Special'
    ],
    'Health & Beauty': [
        '{percent}% Off {product}', 'Free {service} with {product}', '{product} Spa Package',
        'Organic {product} Deal', '{product} Wellness Bundle', 'Professional {product} Treatment',
        '{product} Beauty Box', 'Natural {product} Collection'
    ],
    'Sports & Fitness': [
        '{percent}% Off {equipment}', 'Free Training with {equipment}', '{equipment} Pro Package',
        '{equipment} Beginner Set', '{equipment} Championship Deal', 'Premium {equipment} Bundle',
        '{equipment} Team Discount', 'Fitness {equipment} Challenge'
    ],
    'Books & Education': [
        '{percent}% Off {subject}', 'Free Shipping on {subject}', '{subject} Study Bundle',
        '{subject} Complete Course', '{subject} Professional Certification', 'Advanced {subject} Program',
        '{subject} Student Package', 'Digital {subject} Library'
    ],
    'Home & Garden': [
        '{percent}% Off {item}', 'Free Installation {item}', '{item} Home Makeover',
        '{item} Seasonal Collection', '{item} Smart Home Bundle', 'Eco-Friendly {item}',
        '{item} Designer Series', 'Professional {item} Service'
    ],
    'Automotive': [
        '{percent}% Off {service}', 'Free {service} Check', '{service} Premium Package',
        '{service} Express Deal', '{service} Professional Care', 'Complete {service} Solution',
        '{service} Warranty Package', 'Emergency {service} Deal'
    ]
}

# منتجات وخدمات حسب الفئة
CATEGORY_ITEMS = {
    'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'TV', 'Camera', 'Tablet', 'Smartwatch', 'Gaming Console'],
    'Food': ['Pizza', 'Burger', 'Sushi', 'Pasta', 'Salad', 'Coffee', 'Dessert', 'Sandwich'],
    'Fashion': ['Dress', 'Shoes', 'Jacket', 'Jeans', 'T-Shirt', 'Bag', 'Watch', 'Sunglasses'],
    'Travel': ['Paris Trip', 'Beach Resort', 'City Tour', 'Mountain Hiking', 'Cruise', 'Safari', 'Hotel Stay', 'Flight'],
    'Entertainment': ['Movie Tickets', 'Concert', 'Theater Show', 'Sports Event', 'Comedy Show', 'Music Festival', 'Art Exhibition', 'Gaming'],
    'Health & Beauty': ['Facial Treatment', 'Massage', 'Skincare', 'Makeup', 'Hair Care', 'Spa Day', 'Wellness Program', 'Fitness Class'],
    'Sports & Fitness': ['Gym Membership', 'Yoga Classes', 'Running Shoes', 'Fitness Equipment', 'Sports Gear', 'Personal Training', 'Swimming Pool', 'Tennis Lessons'],
    'Books & Education': ['Programming Course', 'Language Learning', 'Business Books', 'Science Textbooks', 'Online Course', 'Certification Program', 'E-books', 'Audiobooks'],
    'Home & Garden': ['Furniture', 'Garden Tools', 'Home Decor', 'Kitchen Appliances', 'Lighting', 'Plants', 'Cleaning Service', 'Interior Design'],
    'Automotive': ['Oil Change', 'Car Wash', 'Tire Service', 'Engine Repair', 'Car Insurance', 'Vehicle Inspection', 'Auto Parts', 'Roadside Assistance']
}

def generate_coupon_code():
    """توليد كود كوبون عشوائي"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

def generate_coupon_name(category_name):
    """توليد اسم كوبون واقعي"""
    templates = COUPON_TEMPLATES.get(category_name, ['{percent}% Off {item}'])
    items = CATEGORY_ITEMS.get(category_name, ['Product'])
    
    template = random.choice(templates)
    item = random.choice(items)
    percent = random.choice([10, 15, 20, 25, 30, 40, 50])
    
    # استبدال المتغيرات
    name = template.replace('{percent}', str(percent))
    name = name.replace('{product}', item)
    name = name.replace('{item}', item)
    name = name.replace('{equipment}', item)
    name = name.replace('{subject}', item)
    name = name.replace('{service}', item)
    name = name.replace('{destination}', item)
    name = name.replace('{event}', item)
    name = name.replace('{accessory}', random.choice(['Case', 'Charger', 'Stand', 'Cover']))
    
    return name

def generate_description(category_name, coupon_name):
    """توليد وصف واقعي للكوبون"""
    descriptions = {
        'Electronics': [
            f"Get the latest technology with amazing discounts on {coupon_name}",
            f"Limited time offer on premium electronics - {coupon_name}",
            f"Upgrade your tech setup with this exclusive {coupon_name} deal"
        ],
        'Food': [
            f"Delicious meals at unbeatable prices - {coupon_name}",
            f"Satisfy your cravings with our special {coupon_name} offer",
            f"Fresh ingredients and great taste in every {coupon_name}"
        ],
        'Fashion': [
            f"Stay stylish with our trendy {coupon_name} collection",
            f"Fashion-forward designs at incredible prices - {coupon_name}",
            f"Express your style with premium {coupon_name} deals"
        ],
        'Travel': [
            f"Explore amazing destinations with {coupon_name} packages",
            f"Create unforgettable memories with {coupon_name} adventures",
            f"Discover the world with exclusive {coupon_name} offers"
        ],
        'Entertainment': [
            f"Enjoy premium entertainment with {coupon_name} experiences",
            f"Fun and excitement await with {coupon_name} deals",
            f"Make every moment special with {coupon_name} offers"
        ],
        'Health & Beauty': [
            f"Pamper yourself with luxurious {coupon_name} treatments",
            f"Look and feel your best with {coupon_name} services",
            f"Professional care and premium products in {coupon_name}"
        ],
        'Sports & Fitness': [
            f"Achieve your fitness goals with {coupon_name} programs",
            f"Stay active and healthy with {coupon_name} deals",
            f"Professional training and quality equipment in {coupon_name}"
        ],
        'Books & Education': [
            f"Expand your knowledge with {coupon_name} resources",
            f"Learn new skills and advance your career with {coupon_name}",
            f"Quality education and expert instruction in {coupon_name}"
        ],
        'Home & Garden': [
            f"Transform your living space with {coupon_name} solutions",
            f"Create the perfect home environment with {coupon_name}",
            f"Quality products and professional service in {coupon_name}"
        ],
        'Automotive': [
            f"Keep your vehicle running smoothly with {coupon_name} services",
            f"Professional automotive care with {coupon_name} deals",
            f"Reliable service and quality parts in {coupon_name}"
        ]
    }
    
    category_descriptions = descriptions.get(category_name, [f"Great deal on {coupon_name}"])
    return random.choice(category_descriptions)

def clear_existing_data():
    """مسح البيانات الموجودة"""
    conn = get_connection()
    cur = conn.cursor()
    
    print("🧹 Clearing existing data...")
    
    try:
        cur.execute("DELETE FROM user_interactions")
        cur.execute("DELETE FROM coupons")
        cur.execute("ALTER SEQUENCE coupons_id_seq RESTART WITH 1")
        cur.execute("ALTER SEQUENCE user_interactions_id_seq RESTART WITH 1")
        
        conn.commit()
        print("✅ Existing data cleared")
    except Exception as e:
        print(f"❌ Error clearing data: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

def populate_coupons(num_coupons=6000):
    """إضافة كوبونات بطريقة مدروسة"""
    conn = get_connection()
    cur = conn.cursor()
    
    print(f"📦 Adding {num_coupons} coupons...")
    
    # توزيع متوازن للفئات
    coupons_per_category = num_coupons // len(CATEGORIES)
    
    coupon_id = 1
    
    for category in CATEGORIES:
        category_id = category['id']
        category_name = category['name']
        
        print(f"  📂 Adding {coupons_per_category} coupons for {category_name}...")
        
        for i in range(coupons_per_category):
            # توزيع متوازن لأنواع الكوبونات
            coupon_type_id = ((i % len(COUPON_TYPES)) + 1)
            
            # أسعار واقعية حسب الفئة
            if category_name in ['Electronics', 'Automotive']:
                price = round(random.uniform(50, 500), 2)
            elif category_name in ['Travel', 'Home & Garden']:
                price = round(random.uniform(30, 300), 2)
            elif category_name in ['Fashion', 'Sports & Fitness']:
                price = round(random.uniform(20, 200), 2)
            else:
                price = round(random.uniform(5, 100), 2)
            
            name = generate_coupon_name(category_name)
            description = generate_description(category_name, name)
            coupon_code = generate_coupon_code()
            
            # تاريخ انتهاء عشوائي في المستقبل
            end_date = datetime.now() + timedelta(days=random.randint(30, 365))
            
            cur.execute("""
                INSERT INTO coupons (name, description, price, coupon_type_id, category_id, 
                                   provider_id, coupon_status, coupon_code, date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (name, description, price, coupon_type_id, category_id, 
                  random.randint(1, 100), 1, coupon_code, end_date.date()))
            
            coupon_id += 1
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"✅ Added {num_coupons} coupons successfully")

def create_realistic_user_profiles():
    """إنشاء ملفات مستخدمين واقعية"""
    
    # أنماط مستخدمين مختلفة
    user_profiles = {
        # مستخدمين محبي التكنولوجيا
        'tech_lovers': {
            'user_ids': list(range(1, 21)),  # 20 مستخدم
            'preferred_categories': [1],  # Electronics
            'secondary_categories': [8, 7],  # Books, Sports
            'behavior': 'heavy_researcher'  # يبحث كثير قبل الشراء
        },
        
        # محبي الطعام
        'food_enthusiasts': {
            'user_ids': list(range(21, 41)),  # 20 مستخدم
            'preferred_categories': [2],  # Food
            'secondary_categories': [6, 5],  # Health & Beauty, Entertainment
            'behavior': 'impulse_buyer'  # يشتري بسرعة
        },
        
        # محبي الموضة
        'fashion_lovers': {
            'user_ids': list(range(41, 61)),  # 20 مستخدم
            'preferred_categories': [3],  # Fashion
            'secondary_categories': [6, 9],  # Health & Beauty, Home & Garden
            'behavior': 'seasonal_shopper'  # يشتري حسب المواسم
        },
        
        # محبي السفر
        'travelers': {
            'user_ids': list(range(61, 81)),  # 20 مستخدم
            'preferred_categories': [4],  # Travel
            'secondary_categories': [5, 1],  # Entertainment, Electronics
            'behavior': 'planner'  # يخطط مسبقاً
        },
        
        # مستخدمين متنوعين
        'diverse_users': {
            'user_ids': list(range(81, 151)),  # 70 مستخدم
            'preferred_categories': list(range(1, 11)),  # كل الفئات
            'secondary_categories': list(range(1, 11)),
            'behavior': 'balanced'  # متوازن
        }
    }
    
    return user_profiles

def populate_interactions(num_interactions=6000):
    """إضافة تفاعلات واقعية ومدروسة"""
    conn = get_connection()
    cur = conn.cursor()
    
    print(f"👥 Adding {num_interactions} user interactions...")
    
    # الحصول على معرفات الكوبونات
    cur.execute("SELECT id, category_id FROM coupons")
    coupons = cur.fetchall()
    coupon_dict = {coupon[1]: [] for coupon in coupons}  # تجميع حسب الفئة
    
    for coupon in coupons:
        coupon_dict[coupon[1]].append(coupon[0])
    
    user_profiles = create_realistic_user_profiles()
    
    interaction_id = 1
    interactions_added = 0
    
    for profile_name, profile in user_profiles.items():
        print(f"  👤 Creating interactions for {profile_name}...")
        
        for user_id in profile['user_ids']:
            if interactions_added >= num_interactions:
                break
            
            # عدد التفاعلات لكل مستخدم حسب نمطه
            if profile['behavior'] == 'heavy_researcher':
                user_interactions = random.randint(15, 30)
            elif profile['behavior'] == 'impulse_buyer':
                user_interactions = random.randint(8, 15)
            elif profile['behavior'] == 'seasonal_shopper':
                user_interactions = random.randint(10, 20)
            elif profile['behavior'] == 'planner':
                user_interactions = random.randint(12, 25)
            else:  # balanced
                user_interactions = random.randint(5, 15)
            
            # توزيع التفاعلات
            for _ in range(min(user_interactions, num_interactions - interactions_added)):
                
                # اختيار الفئة (80% مفضلة، 20% ثانوية)
                if random.random() < 0.8:
                    category_id = random.choice(profile['preferred_categories'])
                else:
                    category_id = random.choice(profile['secondary_categories'])
                
                # اختيار كوبون من الفئة
                if category_id in coupon_dict and coupon_dict[category_id]:
                    coupon_id = random.choice(coupon_dict[category_id])
                else:
                    continue
                
                # نوع التفاعل حسب السلوك
                if profile['behavior'] == 'heavy_researcher':
                    # يبحث كثير، ينقر أحياناً، يشتري قليلاً
                    action = random.choices(['search', 'click', 'purchase'], 
                                         weights=[70, 25, 5])[0]
                elif profile['behavior'] == 'impulse_buyer':
                    # يشتري بسرعة
                    action = random.choices(['search', 'click', 'purchase'], 
                                         weights=[30, 30, 40])[0]
                elif profile['behavior'] == 'seasonal_shopper':
                    # متوازن
                    action = random.choices(['search', 'click', 'purchase'], 
                                         weights=[50, 35, 15])[0]
                elif profile['behavior'] == 'planner':
                    # يبحث ويخطط
                    action = random.choices(['search', 'click', 'purchase'], 
                                         weights=[60, 30, 10])[0]
                else:  # balanced
                    action = random.choices(['search', 'click', 'purchase'], 
                                         weights=[50, 30, 20])[0]
                
                # النقاط
                score = {'search': 2.0, 'click': 5.0, 'purchase': 15.0}[action]
                
                # التاريخ (آخر 90 يوم)
                timestamp = datetime.now() - timedelta(days=random.randint(0, 90))
                
                cur.execute("""
                    INSERT INTO user_interactions (user_id, coupon_id, action, score, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, coupon_id, action, score, timestamp))
                
                interactions_added += 1
                
                if interactions_added >= num_interactions:
                    break
            
            if interactions_added >= num_interactions:
                break
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"✅ Added {interactions_added} interactions successfully")

def create_test_scenarios():
    """إنشاء سيناريوهات اختبار محددة"""
    conn = get_connection()
    cur = conn.cursor()
    
    print("🧪 Creating test scenarios...")
    
    # سيناريو 1: مستخدم محب للإلكترونيات
    test_user_1 = 999
    cur.execute("SELECT id FROM coupons WHERE category_id = 1 LIMIT 5")
    electronics_coupons = [row[0] for row in cur.fetchall()]
    
    for coupon_id in electronics_coupons[:3]:
        cur.execute("""
            INSERT INTO user_interactions (user_id, coupon_id, action, score, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (test_user_1, coupon_id, 'search', 2.0, datetime.now()))
        
        cur.execute("""
            INSERT INTO user_interactions (user_id, coupon_id, action, score, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (test_user_1, coupon_id, 'click', 5.0, datetime.now()))
    
    # شراء واحد
    cur.execute("""
        INSERT INTO user_interactions (user_id, coupon_id, action, score, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (test_user_1, electronics_coupons[0], 'purchase', 15.0, datetime.now()))
    
    # سيناريو 2: مستخدم محب للطعام
    test_user_2 = 998
    cur.execute("SELECT id FROM coupons WHERE category_id = 2 LIMIT 5")
    food_coupons = [row[0] for row in cur.fetchall()]
    
    for coupon_id in food_coupons[:4]:
        cur.execute("""
            INSERT INTO user_interactions (user_id, coupon_id, action, score, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (test_user_2, coupon_id, 'purchase', 15.0, datetime.now()))
    
    conn.commit()
    cur.close()
    conn.close()
    
    print("✅ Test scenarios created")
    print("  🔬 Test User 999: Electronics lover")
    print("  🔬 Test User 998: Food enthusiast")

def print_statistics():
    """طباعة إحصائيات البيانات"""
    conn = get_connection()
    cur = conn.cursor()
    
    print("\n📊 Database Statistics:")
    print("=" * 50)
    
    # إحصائيات الكوبونات
    cur.execute("""
        SELECT c.name, COUNT(*) as count 
        FROM coupons co 
        JOIN categories c ON co.category_id = c.id 
        GROUP BY c.name 
        ORDER BY count DESC
    """)
    
    print("📦 Coupons by Category:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} coupons")
    
    # إحصائيات التفاعلات
    cur.execute("""
        SELECT action, COUNT(*) as count 
        FROM user_interactions 
        GROUP BY action 
        ORDER BY count DESC
    """)
    
    print("\n👥 Interactions by Type:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]} interactions")
    
    # أكثر المستخدمين نشاطاً
    cur.execute("""
        SELECT user_id, COUNT(*) as interactions, SUM(score) as total_score
        FROM user_interactions 
        GROUP BY user_id 
        ORDER BY interactions DESC 
        LIMIT 5
    """)
    
    print("\n🏆 Top 5 Most Active Users:")
    for row in cur.fetchall():
        print(f"  User {row[0]}: {row[1]} interactions, {row[2]} total score")
    
    # إجمالي الإحصائيات
    cur.execute("SELECT COUNT(*) FROM coupons")
    total_coupons = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM user_interactions")
    total_interactions = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(DISTINCT user_id) FROM user_interactions")
    unique_users = cur.fetchone()[0]
    
    print(f"\n📈 Total Statistics:")
    print(f"  Total Coupons: {total_coupons}")
    print(f"  Total Interactions: {total_interactions}")
    print(f"  Unique Users: {unique_users}")
    
    cur.close()
    conn.close()

def main():
    """الدالة الرئيسية"""
    print("🚀 Starting Large Data Population with SQLite")
    print("=" * 60)
    
    create_tables()
    
    try:
        # مسح البيانات القديمة
        clear_existing_data()
        
        # إضافة الكوبونات
        populate_coupons(6000)
        
        # إضافة التفاعلات
        populate_interactions(6000)
        
        # إنشاء سيناريوهات الاختبار
        create_test_scenarios()
        
        # طباعة الإحصائيات
        print_statistics()
        
        print("\n🎉 Data population completed successfully!")
        print("\n🧪 Test Commands:")
        print("  curl -X POST http://YOUR_SERVER:8000/build_vector_store")
        print("  curl 'http://YOUR_SERVER:8000/get_recommendations?user_id=999&top_n=10'")
        print("  curl 'http://YOUR_SERVER:8000/get_recommendations?user_id=998&top_n=10'")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()