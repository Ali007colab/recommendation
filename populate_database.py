import mysql.connector
import random
from datetime import datetime, timedelta
import string

# إعداد الاتصال بقاعدة البيانات
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='',  # أو كلمة المرور الخاصة بك
        database='recommendation_db'
    )

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
    {'id': 1, 'name': 'Discount', 'description': 'Percentage discount on single item'},
    {'id': 2, 'name': 'Bundle Deal', 'description': 'Special pricing for multiple items'},
    {'id': 3, 'name': 'Free Shipping', 'description': 'Free shipping on orders'},
    {'id': 4, 'name': 'Buy One Get One', 'description': 'Buy one item, get another free'},
    {'id': 5, 'name': 'Cashback', 'description': 'Get cashback on purchase'},
    {'id': 6, 'name': 'Flash Sale', 'description': 'Limited time discount offer'},
    {'id': 7, 'name': 'Loyalty Reward', 'description': 'Reward for loyal customers'},
    {'id': 8, 'name': 'First Time Buyer', 'description': 'Special offer for new customers'},
    {'id': 9, 'name': 'Seasonal Sale', 'description': 'Seasonal discount promotion'},
    {'id': 10, 'name': 'Bulk Purchase', 'description': 'Discount for buying in bulk'}
]


COUPON_TEMPLATES = {
    'Electronics': [
        'Gaming Laptop {discount}% Off',
        'Smartphone Bundle Deal',
        'Wireless Headphones Sale',
        'Smart TV Cashback Offer',
        'Tablet Flash Sale',
        'Camera Equipment Discount',
        'Smart Watch Special Deal',
        'Gaming Console Bundle',
        'Laptop Accessories Deal',
        'Phone Case Bulk Buy',
        'Bluetooth Speaker Sale',
        'Smartwatch Flash Deal',
        'Gaming Mouse Discount',
        'Wireless Charger Bundle',
        'VR Headset Special Offer'
    ],
    'Food': [
        'Pizza {discount}% Off',
        'Burger Combo Deal',
        'Healthy Food Bundle',
        'Restaurant Cashback',
        'Fast Food Flash Sale',
        'Organic Products Discount',
        'Beverage Special Offer',
        'Dessert Bundle Deal',
        'Breakfast Combo Sale',
        'Dinner Package Discount',
        'Coffee Shop Bundle',
        'Ice Cream Flash Sale',
        'Sandwich Combo Deal',
        'Juice Bar Special',
        'Bakery Items Discount'
    ],
    'Fashion': [
        'Designer Clothes {discount}% Off',
        'Shoes Bundle Deal',
        'Summer Collection Sale',
        'Accessories Cashback',
        'Fashion Flash Sale',
        'Winter Wear Discount',
        'Jewelry Special Offer',
        'Handbag Bundle Deal',
        'Sportswear Sale',
        'Formal Wear Discount',
        'Sunglasses Flash Deal',
        'Watch Collection Sale',
        'Belt and Wallet Bundle',
        'Scarf Special Offer',
        'Hat Collection Discount'
    ],
    'Travel': [
        'Flight Tickets {discount}% Off',
        'Hotel Stay Bundle',
        'Vacation Package Deal',
        'Car Rental Cashback',
        'Travel Insurance Sale',
        'Cruise Discount',
        'Adventure Tour Special',
        'City Break Bundle',
        'Beach Resort Deal',
        'Mountain Trip Discount',
        'Bus Ticket Flash Sale',
        'Train Journey Bundle',
        'Airport Transfer Deal',
        'Travel Gear Special',
        'Luggage Discount Offer'
    ],
    'Entertainment': [
        'Movie Tickets {discount}% Off',
        'Concert Bundle Deal',
        'Gaming Subscription Sale',
        'Streaming Service Cashback',
        'Theme Park Flash Sale',
        'Sports Event Discount',
        'Theater Show Special',
        'Music Festival Bundle',
        'Comedy Show Deal',
        'Art Exhibition Discount',
        'Bowling Night Special',
        'Karaoke Bundle Deal',
        'Arcade Games Flash Sale',
        'Mini Golf Discount',
        'Escape Room Special'
    ],
    'Health & Beauty': [
        'Skincare {discount}% Off',
        'Makeup Bundle Deal',
        'Spa Treatment Sale',
        'Fitness Supplement Cashback',
        'Beauty Products Flash Sale',
        'Wellness Package Discount',
        'Cosmetics Special Offer',
        'Hair Care Bundle',
        'Perfume Sale',
        'Health Check Discount',
        'Massage Therapy Special',
        'Nail Care Bundle',
        'Anti-aging Flash Sale',
        'Vitamin Supplement Deal',
        'Aromatherapy Discount'
    ],
    'Sports & Fitness': [
        'Gym Membership {discount}% Off',
        'Sports Equipment Bundle',
        'Fitness Gear Sale',
        'Protein Supplement Cashback',
        'Workout Clothes Flash Sale',
        'Sports Shoes Discount',
        'Fitness Class Special',
        'Athletic Wear Bundle',
        'Exercise Equipment Deal',
        'Sports Nutrition Discount',
        'Yoga Mat Flash Sale',
        'Swimming Gear Bundle',
        'Running Shoes Special',
        'Fitness Tracker Deal',
        'Sports Drink Discount'
    ],
    'Books & Education': [
        'Textbooks {discount}% Off',
        'Online Course Bundle',
        'Educational Software Sale',
        'Book Collection Cashback',
        'E-learning Flash Sale',
        'Academic Materials Discount',
        'Professional Course Special',
        'Study Guide Bundle',
        'Educational Toys Deal',
        'Language Learning Discount',
        'Digital Library Access',
        'Certification Course Bundle',
        'Reference Books Flash Sale',
        'Audio Books Special',
        'Educational Games Discount'
    ],
    'Home & Garden': [
        'Furniture {discount}% Off',
        'Garden Tools Bundle',
        'Home Decor Sale',
        'Kitchen Appliances Cashback',
        'Cleaning Supplies Flash Sale',
        'Outdoor Furniture Discount',
        'Home Improvement Special',
        'Gardening Equipment Bundle',
        'Interior Design Deal',
        'Home Security Discount',
        'Plant Collection Flash Sale',
        'Lighting Fixtures Bundle',
        'Storage Solutions Special',
        'Bedding Set Deal',
        'Cookware Discount Offer'
    ],
    'Automotive': [
        'Car Parts {discount}% Off',
        'Auto Service Bundle',
        'Car Accessories Sale',
        'Vehicle Maintenance Cashback',
        'Auto Parts Flash Sale',
        'Car Care Products Discount',
        'Tire Replacement Special',
        'Auto Insurance Bundle',
        'Car Wash Deal',
        'Vehicle Upgrade Discount',
        'Engine Oil Flash Sale',
        'Car Electronics Bundle',
        'Dashboard Accessories Special',
        'Car Audio System Deal',
        'Vehicle Safety Discount'
    ]
}

def generate_coupon_code():
    """توليد كود كوبون عشوائي"""
    letters = ''.join(random.choices(string.ascii_uppercase, k=3))
    numbers = ''.join(random.choices(string.digits, k=3))
    return f"{letters}{numbers}"

def generate_coupon_name_and_description(category_name, coupon_type_name):
    """توليد اسم ووصف الكوبون"""
    templates = COUPON_TEMPLATES.get(category_name, ['Special Offer'])
    template = random.choice(templates)
    
    # إضافة نسبة خصم عشوائية إذا كان في القالب
    if '{discount}' in template:
        discount = random.choice([10, 15, 20, 25, 30, 35, 40, 50])
        name = template.format(discount=discount)
    else:
        name = template
    
    # توليد وصف مناسب
    descriptions = {
        'Discount': f'Get special discount on {category_name.lower()} items with amazing savings',
        'Bundle Deal': f'Special bundle pricing for {category_name.lower()} products - buy more save more',
        'Free Shipping': f'Free shipping on {category_name.lower()} orders above minimum purchase',
        'Buy One Get One': f'Buy one {category_name.lower()} item, get another absolutely free',
        'Cashback': f'Get instant cashback on {category_name.lower()} purchases',
        'Flash Sale': f'Limited time flash sale on premium {category_name.lower()} items',
        'Loyalty Reward': f'Exclusive {category_name.lower()} deals for our valued loyal customers',
        'First Time Buyer': f'Special welcome discount on {category_name.lower()} for new customers',
        'Seasonal Sale': f'Seasonal {category_name.lower()} promotion with huge savings',
        'Bulk Purchase': f'Volume discount on {category_name.lower()} items for bulk orders'
    }
    
    description = descriptions.get(coupon_type_name, f'Amazing special offer on {category_name.lower()} products')
    
    return name, description

def generate_price(category_name):
    """توليد سعر مناسب للفئة"""
    price_ranges = {
        'Electronics': (50, 1500),
        'Food': (5, 100),
        'Fashion': (20, 500),
        'Travel': (100, 2000),
        'Entertainment': (10, 200),
        'Health & Beauty': (15, 300),
        'Sports & Fitness': (25, 800),
        'Books & Education': (10, 150),
        'Home & Garden': (30, 1000),
        'Automotive': (50, 2000)
    }
    
    min_price, max_price = price_ranges.get(category_name, (10, 100))
    return round(random.uniform(min_price, max_price), 2)

def clear_tables():
    """مسح البيانات الموجودة"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🗑️ مسح البيانات الموجودة...")
    

    tables = ['user_interactions', 'coupons', 'coupon_types', 'categories']
    
    for table in tables:
        cursor.execute(f"DELETE FROM {table}")
        print(f"   ✅ تم مسح جدول {table}")
    
    conn.commit()
    cursor.close()
    conn.close()

def populate_categories():
    """ملء جدول الفئات"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("📂 إضافة الفئات...")
    
    for category in CATEGORIES:
        cursor.execute(
            "INSERT INTO categories (id, name) VALUES (%s, %s)",
            (category['id'], category['name'])
        )
    
    conn.commit()
    cursor.close()
    conn.close()
    print(f"   ✅ تم إضافة {len(CATEGORIES)} فئة")

def populate_coupon_types():
    """ملء جدول أنواع الكوبونات"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("🏷️ إضافة أنواع الكوبونات...")
    
    for coupon_type in COUPON_TYPES:
        cursor.execute(
            "INSERT INTO coupon_types (id, name, description) VALUES (%s, %s, %s)",
            (coupon_type['id'], coupon_type['name'], coupon_type['description'])
        )
    
    conn.commit()
    cursor.close()
    conn.close()
    print(f"   ✅ تم إضافة {len(COUPON_TYPES)} نوع كوبون")

def populate_coupons(count=2000):
    """ملء جدول الكوبونات"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print(f"🎫 إضافة {count} كوبون...")
    
    for i in range(1, count + 1):
        category = random.choice(CATEGORIES)
        coupon_type = random.choice(COUPON_TYPES)
        
        name, description = generate_coupon_name_and_description(
            category['name'], coupon_type['name']
        )
        
        price = generate_price(category['name'])
        provider_id = random.randint(1, 100)
        coupon_status = random.choice([0, 1])  # 0 = inactive, 1 = active
        coupon_code = generate_coupon_code()
        
        # تاريخ عشوائي في المستقبل
        start_date = datetime.now()
        end_date = start_date + timedelta(days=random.randint(30, 365))
        date = end_date.strftime('%Y-%m-%d')
        
        cursor.execute("""
            INSERT INTO coupons 
            (id, name, description, price, coupon_type_id, category_id, 
             provider_id, coupon_status, coupon_code, date) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            i, name, description, price, coupon_type['id'], category['id'],
            provider_id, coupon_status, coupon_code, date
        ))
        
        if i % 500 == 0:
            print(f"   📊 تم إضافة {i} كوبون...")
            conn.commit()
    
    conn.commit()
    cursor.close()
    conn.close()
    print(f"   ✅ تم إضافة {count} كوبون بنجاح")

def populate_user_interactions_balanced(count=2000):
    """ملء جدول التفاعلات مع توزيع متوازن للفئات"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print(f"👥 إضافة {count} تفاعل بتوزيع متوازن...")
    
    balanced_users = {}
    

    for user_id in range(1, 51): 
        if user_id <= 10:
            primary_category = ((user_id - 1) % 10) + 1
            secondary_categories = random.sample([c for c in range(1, 11) if c != primary_category], 2)
            balanced_users[user_id] = {
                'categories': [primary_category] + secondary_categories,
                'weights': [0.5, 0.3, 0.2]  # 50% رئيسي، 30% ثانوي أول، 20% ثانوي ثاني
            }
        elif user_id <= 30:
            # مستخدمين مع فئتين متساويتين + فئة ثالثة
            categories = random.sample(range(1, 11), 3)
            balanced_users[user_id] = {
                'categories': categories,
                'weights': [0.4, 0.35, 0.25]  # توزيع أكثر توازناً
            }
        else:
            num_categories = random.randint(3, 4)
            categories = random.sample(range(1, 11), num_categories)
            if num_categories == 3:
                weights = [0.4, 0.35, 0.25]
            else:
                weights = [0.3, 0.25, 0.25, 0.2]
            
            balanced_users[user_id] = {
                'categories': categories,
                'weights': weights
            }
    
    for user_id in range(51, 101):
        num_categories = random.randint(2, 4)
        categories = random.sample(range(1, 11), num_categories)
        # توزيع متوازن للأوزان
        if num_categories == 2:
            weights = [0.6, 0.4]
        elif num_categories == 3:
            weights = [0.45, 0.35, 0.2]
        else:
            weights = [0.35, 0.25, 0.25, 0.15]
        
        balanced_users[user_id] = {
            'categories': categories,
            'weights': weights
        }
    
    actions = ['search', 'click', 'purchase']
    action_scores = {'search': 2.0, 'click': 5.0, 'purchase': 15.0}
    
    for i in range(1, count + 1):
        user_id = random.choice(list(balanced_users.keys()))
        user_prefs = balanced_users[user_id]
        
        category_id = random.choices(user_prefs['categories'], weights=user_prefs['weights'])[0]
        
        cursor.execute(
            "SELECT id FROM coupons WHERE category_id = %s ORDER BY RAND() LIMIT 1",
            (category_id,)
        )
        result = cursor.fetchone()
        
        if result:
            coupon_id = result[0]
            
            action_weights = [0.35, 0.40, 0.25]
            action = random.choices(actions, weights=action_weights)[0]
            score = action_scores[action]
            
            days_ago = random.randint(1, 30)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
            
            cursor.execute("""
                INSERT INTO user_interactions 
                (user_id, coupon_id, action, score, timestamp) 
                VALUES (%s, %s, %s, %s, %s)
            """, (user_id, coupon_id, action, score, timestamp))
        
        if i % 500 == 0:
            print(f"   📊 تم إضافة {i} تفاعل...")
            conn.commit()
    
    conn.commit()
    cursor.close()
    conn.close()
    print(f"   ✅ تم إضافة {count} تفاعل بتوزيع متوازن بنجاح")

def generate_statistics():
    """إحصائيات البيانات المولدة"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print("\n📊 إحصائيات البيانات المتوازنة:")
    
    cursor.execute("""
        SELECT c.name, COUNT(co.id) as count 
        FROM categories c 
        LEFT JOIN coupons co ON c.id = co.category_id 
        GROUP BY c.id, c.name 
        ORDER BY count DESC
    """)
    
    print("   🎫 الكوبونات لكل فئة:")
    for category, count in cursor.fetchall():
        print(f"      {category}: {count} كوبون")
    
    cursor.execute("""
        SELECT ui.user_id, COUNT(*) as interactions, 
               SUM(CASE WHEN ui.action = 'purchase' THEN 1 ELSE 0 END) as purchases,
               SUM(CASE WHEN ui.action = 'click' THEN 1 ELSE 0 END) as clicks,
               SUM(CASE WHEN ui.action = 'search' THEN 1 ELSE 0 END) as searches
        FROM user_interactions ui
        WHERE ui.user_id <= 20
        GROUP BY ui.user_id 
        ORDER BY ui.user_id
    """)
    
    print("\n   👥 المستخدمين المتوازنين (للاختبار):")
    for user_id, interactions, purchases, clicks, searches in cursor.fetchall():
        print(f"      المستخدم {user_id}: {interactions} تفاعل ({purchases} مشتريات، {clicks} نقرات، {searches} بحث)")
    

    cursor.execute("""
        SELECT cat.name, COUNT(ui.id) as interactions,
               SUM(CASE WHEN ui.action = 'purchase' THEN 1 ELSE 0 END) as purchases
        FROM categories cat
        JOIN coupons c ON cat.id = c.category_id
        JOIN user_interactions ui ON c.id = ui.coupon_id
        GROUP BY cat.id, cat.name
        ORDER BY interactions DESC
    """)
    
    print("\n   📈 التفاعلات لكل فئة (متوازنة):")
    for category, interactions, purchases in cursor.fetchall():
        print(f"      {category}: {interactions} تفاعل ({purchases} مشتريات)")
    

    cursor.execute("""
        SELECT ui.user_id, cat.name, COUNT(ui.id) as interactions,
               ROUND(COUNT(ui.id) * 100.0 / SUM(COUNT(ui.id)) OVER (PARTITION BY ui.user_id), 1) as percentage
        FROM user_interactions ui
        JOIN coupons c ON ui.coupon_id = c.id
        JOIN categories cat ON c.category_id = cat.id
        WHERE ui.user_id IN (1, 2, 3, 11, 21)
        GROUP BY ui.user_id, cat.id, cat.name
        ORDER BY ui.user_id, interactions DESC
    """)
    
    print("\n   🎯 توزيع الفئات لمستخدمين مختارين:")
    current_user = None
    for user_id, category, interactions, percentage in cursor.fetchall():
        if current_user != user_id:
            if current_user is not None:
                print()
            print(f"      المستخدم {user_id}:")
            current_user = user_id
        print(f"        {category}: {interactions} تفاعل ({percentage}%)")
    
    cursor.close()
    conn.close()

def main():
    """الدالة الرئيسية"""
    print("🚀 بدء ملء قاعدة البيانات بتوزيع متوازن...")
    print("=" * 60)
    
    try:

        clear_tables()
        

        populate_categories()
        populate_coupon_types()
        populate_coupons(2000)
        populate_user_interactions_balanced(2000)
        

        generate_statistics()
        
        print("\n" + "=" * 60)
        print("✅ تم ملء قاعدة البيانات بتوزيع متوازن بنجاح!")
        print("\n🧪 للاختبار المحسن:")
        print("   1. شغل: POST http://localhost:8000/analyze")
        print("   2. اختبر User 1: GET http://localhost:8000/get_recommendations?user_id=1&top_n=10")
        print("   3. قيم الجودة: GET http://localhost:8000/evaluate_similarity?user_id=1")
        print("   4. اختبر User 11: GET http://localhost:8000/get_recommendations?user_id=11&top_n=10")
        print("   5. اختبر User 21: GET http://localhost:8000/get_recommendations?user_id=21&top_n=10")
        
        print("\n📋 التوزيع المتوازن الجديد:")
        print("   • User 1-10: فئة رئيسية (50%) + فئتين ثانويتين (30%, 20%)")
        print("   • User 11-30: ثلاث فئات متوازنة (40%, 35%, 25%)")
        print("   • User 31-50: 3-4 فئات متقاربة")
        print("   • النتيجة المتوقعة: توزيع أكثر توازناً في التوصيات")
        
    except Exception as e:
        print(f"❌ خطأ: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()