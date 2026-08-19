# Grouping-Products-And-Warehouses-Based-on-Inventory-Patterns
This is the final project on Machine Learning
# PROJECT TITLE: WAREHOUSE INVENTORY OPTIMIZATION AND SEGMENTATION

#; Project Overview
This project analyzes a warehouse inventory dataset containing **1000 product records** across multiple warehouses.  
The goal is to clean messy inventory data, uncover operational insights, and use unsupervised machine learning to segment products and warehouses for better stock management and cost reduction.

The dataset tracks product details such as **Product ID, Product Name, Category, Warehouse, Location, Quantity, Price, Supplier, Status, and Last Restocked date**.  
The raw data is messy with incorrect data types and missing values.

# Problem Statement
Warehouses often struggle with 3 key problems:
1. **Overstock**: Capital tied up in products that do not sell
2. **Stockouts**: Lost sales due to "Out of Stock" on popular items
3. **Inefficient Warehousing**: Lack of segmentation leads to treating all products and warehouses the same

This results in wasted space, expired stock, and poor reordering decisions.

## Aims/Objectives of the Project
1. **Data Cleaning**: Fix data types, handle missing values, and standardize text
2. **Exploratory Data Analysis**: Identify top categories, slow-moving stock, and warehouse performance
3. **Unsupervised Segmentation**: Group products and warehouses by behavior to find hidden patterns
4. **Actionable Insights**: Provide recommendations on what to restock, discount, or relocate
# Data Understanding

**Data Source**: [Kaggle](https://github.com/eyowhite/Messy-dataset/blob/main/warehouse_messy_data.csv)

The dataset contains warehouse inventory records. It has **1000 rows** and **10 columns**.  

### **Column Descriptions:**
1. **Product ID**:Unique identifier for each product in the inventory.
2. **Product Name**:Name of the product. Examples: "Gadget Y", "Widget A". May have inconsistent capitalization and spacing.
3. **Category**:The product category or type. Used to group similar products together.
4. **Warehouse**: Name of the warehouse where the product is stored.
5. **Location**:Specific location/shelf/bin within the warehouse.
6. **Quantity**:Current stock quantity of the product.
7. **Price**:Unit price of the product in currency.
8. **Supplier**:Name of the supplier or vendor who provides the product.
9. **Status**:Stock status of the product. Expected values: "In Stock", "Out of Stock".
10. **Last Restocked**:Date when the product was last restocked.
=
Head:
   Product ID Product Name     Category    Warehouse Location     Quantity  \
0        1102    gadget y   ELECTRONICS  Warehouse 2  Aisle 1          300   
1        1435    gadget y   ELECTRONICS  Warehouse 2  Aisle 4  two hundred   
2        1860    widget a      CLOTHING  Warehouse 2  Aisle 3          100   
3        1270    gadget z          TOYS  Warehouse 2  Aisle 4           50   
4        1106    widget a     FURNITURE  Warehouse 3  Aisle 3  two hundred   

   Price    Supplier        Status Last Restocked  
0   9.99  Supplier C      In Stock            NaN  
1  19.99  Supplier C  Out of Stock            NaN  
2  19.99  Supplier B      In Stock     20/12/2022  
3  49.99  Supplier B      In Stock     20/12/2022  
4   9.99  Supplier D  Out of Stock     25/04/2023  
INFO:
<class 'pandas.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 10 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   Product ID      1000 non-null   int64  
 1   Product Name    1000 non-null   str    
 2   Category        1000 non-null   str    
 3   Warehouse       1000 non-null   str    
 4   Location        1000 non-null   str    
 5   Quantity        842 non-null    str    
 6   Price           793 non-null    float64
 7   Supplier        1000 non-null   str    
 8   Status          1000 non-null   str    
 9   Last Restocked  800 non-null    str    
dtypes: float64(1), int64(1), str(8)
memory usage: 78.3 KB
DESCRIBE NUMERIC:
        Product ID       Price
count  1000.000000  793.000000
mean   1503.929000   28.085839
std     289.998108   14.686312
min    1000.000000    9.990000
25%    1242.750000   19.990000
50%    1505.000000   29.990000
75%    1757.250000   49.990000
max    1998.000000   49.990000
DESCRIBE CATEGORICAL:
       Product Name   Category    Warehouse Location Quantity    Supplier  \
count          1000       1000         1000     1000      842        1000   
unique            6          4            3        5        5           4   
top       gadget y   FURNITURE  Warehouse 1  Aisle 3      300  Supplier B   
freq            177        265          349      211      177         288   

          Status Last Restocked  
count       1000            800  
unique         3              4  
top     In Stock     20/12/2022  
freq         340            218  
C:\Users\frkuhn\AppData\Local\Temp\ipykernel_17256\1570128282.py:22: Pandas4Warning: For backward compatibility, 'str' dtypes are included by select_dtypes when 'object' dtype is specified. This behavior is deprecated and will be removed in a future version. Explicitly pass 'str' to `include` to select them, or to `exclude` to remove them and silence this warning.
See https://pandas.pydata.org/docs/user_guide/migration-3-strings.html#string-migration-select-dtypes for details on how to write code that works with pandas 2 and 3.
  print(df.describe(include='object'))
# Data Cleaning
INFO AFTER FIXING DATA TYPES:
<class 'pandas.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 10 columns):
 #   Column          Non-Null Count  Dtype         
---  ------          --------------  -----         
 0   Product ID      1000 non-null   int64         
 1   Product Name    1000 non-null   str           
 2   Category        1000 non-null   str           
 3   Warehouse       1000 non-null   str           
 4   Location        1000 non-null   str           
 5   Quantity        682 non-null    float64       
 6   Price           793 non-null    float64       
 7   Supplier        1000 non-null   str           
 8   Status          1000 non-null   str           
 9   Last Restocked  800 non-null    datetime64[us]
dtypes: datetime64[us](1), float64(2), int64(1), str(6)
memory usage: 78.3 KB
None
Product ID          0
Product Name        0
Category            0
Warehouse           0
Location            0
Quantity          318
Price             207
Supplier            0
Status              0
Last Restocked    200
dtype: int64
Product ID        0
Product Name      0
Category          0
Warehouse         0
Location          0
Quantity          0
Price             0
Supplier          0
Status            0
Last Restocked    0
dtype: int64
np.int64(0)

#Convert all the columns title then lowercase to ensure consistency
df.columns = df.columns.str.strip().str.lower()
print("Columns after cleaning:")
print(df.columns)
Columns after cleaning:
Index(['product id', 'product name', 'category', 'warehouse', 'location',
       'quantity', 'price', 'supplier', 'status', 'last restocked'],
      dtype='str')

      Head of cleaned DataFrame:
   product id product name     category    warehouse location  quantity  \
0        1102    gadget y   ELECTRONICS  Warehouse 2  Aisle 1     300.0   
1        1435    gadget y   ELECTRONICS  Warehouse 2  Aisle 4       0.0   
2        1860    widget a      CLOTHING  Warehouse 2  Aisle 3     100.0   
3        1270    gadget z          TOYS  Warehouse 2  Aisle 4      50.0   
4        1106    widget a     FURNITURE  Warehouse 3  Aisle 3       0.0   

   price    supplier        status last restocked  
0   9.99  Supplier C      In Stock     2022-12-20  
1  19.99  Supplier C  Out of Stock     2022-12-20  
2  19.99  Supplier B      In Stock     2022-12-20  
3  49.99  Supplier B      In Stock     2022-12-20  
4   9.99  Supplier D  Out of Stock     2023-04-25

# Exploratory Data Analysis
## Univariate EDA
<img width="571" height="455" alt="image" src="https://github.com/user-attachments/assets/a4c8941f-3a26-4cbb-840f-323461ec3828" />

## **Business Insights: Count of Products by Category**

### **Warehouse Operations**
- **Space Allocation**: *FURNITURE* has the most SKUs > 250. Prioritize warehouse zoning and picker routes for this category.
- **Balanced Load**: SKU distribution is even across 4 categories. No single category is overloading warehouse capacity.

### **Inventory Risk & Strategy**
- **Diversified Risk**: No category dominates > 30% of catalog. Reduces exposure if demand drops in one category.
- **Procurement Focus**: Top 2 categories by variety are *FURNITURE* and *CLOTHING*. Buying team should prioritize supplier contracts here.

### **Growth & Optimization**
- **Expansion Opportunity**: *TOYS* has the lowest SKU count ~225. Evaluate if this is low margin or an untapped growth area.
- **Assortment Health**: Balanced 4-category mix suggests a general retail model vs niche. Good for cross-selling

- <img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/096fe485-ce4f-4065-9ec4-b731e52b48ad" />

Status Breakdown:
status
In Stock        340
Out of Stock    332
Low Stock       328
Name: count, dtype: int64

Status %:
status
In Stock        34.0
Out of Stock    33.2
Low Stock       32.8
Name: count, dtype: float64
## **Findings: Count of Products by Stock Status**

#### **Status Counts**
- **In Stock**: 340 products, 34.0%
- **Out of Stock**: 332 products, 33.2%
- **Low Stock**: 328 products, 32.8%

#### **Distribution Pattern**
- **Total Products**: 1000
- **Balance**: Nearly uniform distribution across all 3 statuses
- **Range**: Only 12 products difference between highest and lowest
- **Ranking**: *In Stock > Out of Stock > Low Stock


### **Business Insights**
#### **Inventory Health**
- **High Stockout Risk**: Only 34% of products are *In Stock*. 66% are at risk or unavailable.
- **Revenue Impact**: 332 SKUs *Out of Stock* = immediate lost sales opportunities.

#### **Operational Issues**
- **Poor Inventory Balance**: Even split indicates systemic replenishment problems, not isolated issues.
- **Emergency Replenishment**: High *Low Stock* + *Out of Stock* suggests reactive vs proactive inventory management.

#### **Optimization Actions**
- **Priority 1 - Restock**: Focus on moving *Out of Stock* items back to *In Stock* to recover revenue.
- **Priority 2 - Prevent OOS**: Reorder *Low Stock* items using *Last Restocked date* to prevent them from going OOS.
- **Root Cause**: Analyze by *Category* and *Warehouse* to identify if specific segments are driving the OOS problem.
- **Segmentation**: Use *Stock Status* as a key feature. Target "High Price + Low Stock" first for maximum impact.

#### **Contrast to Project Assumption**
- **Not Overstock**: The data shows understock is the main problem, not excess inventory.
- **Forecasting Gap**: 66% not fully stocked points to poor demand forecasting or reorder point settings.

<img width="629" height="470" alt="image" src="https://github.com/user-attachments/assets/00e566f4-5077-493e-a377-8968ce2d9e1d" />

## **Findings: Distribution of Product Prices**

### **Observable Features**

#### **Distribution Pattern**
- **Shape**: Multi-modal distribution with 4 distinct peaks
- **Primary Peak**: ~30 price point, highest frequency ~425 products
- **Secondary Peaks**: 
    - ~10 price point, ~180 products
    - ~20 price point, ~210 products  
    - ~50 price point, ~240 products
- **Gaps**: Very low frequency between peaks at 15, 25-28, 35-45

#### **Data Characteristics**
- **No Normal Distribution**: Prices are clustered, not spread evenly
- **Concentration**: ~42% of products are priced around 30
- **Spread**: Prices are not continuous, suggests price banding or tiered pricing

### **Business Insights**
#### **Pricing Strategy**
- **Tiered Pricing Model**: 4 clear price tiers exist: Budget ~10, Value ~20, Standard ~30, Premium ~50
- **Core Offering**: *Price = 30* is the dominant tier. This is likely your main product line.
- **Price Anchoring**: Gaps between 30 and 50 suggest room for mid-premium products.

#### **Inventory and  Revenue Impact**
- **Revenue Concentration**: Most revenue likely comes from the ~30 price band due to high volume.
- **Portfolio Balance**: Presence in 4 tiers allows targeting different customer segments.
- **Low Competition Between Tiers**: Gaps mean less cannibalization between price points.

<img width="542" height="412" alt="image" src="https://github.com/user-attachments/assets/bf1b7336-734b-47be-b5f1-f68679345a57" />

- ## **Findings: Distribution of Products by Warehouse**

### **1. Observable Features**
#### **Plot Structure**

#### **Warehouse Distribution**
- **Warehouse 1**: 34.9% - Largest share
- **Warehouse 2**: 33.2% - Middle share  
- **Warehouse 3**: 31.9% - Smallest share

#### **Distribution Pattern**
- **Balance**: Highly even distribution across all warehouses
- **Range**: Only 3.0% difference between largest and smallest
- **Ranking**: Warehouse 1 > Warehouse 2 > Warehouse 3

### **Business Insights**
#### **Warehouse Operations**
- **Balanced Load**: Inventory is evenly spread. No warehouse is overloaded or underutilized.
- **Capacity Planning**: Similar product counts suggest similar space and staffing needs across all 3 warehouses.
- **Risk Distribution**: Risk of disruption is spread evenly. No single point of failure for >35% of stock.

#### **Logistics And Efficiency**
- **Fulfillment Strategy**: Even distribution supports regional fulfillment. Assign warehouses by geography, not by product type.
- **Transfer Costs**: Low imbalance means fewer emergency inter-warehouse transfers needed.

## Bivariate EDA

<img width="841" height="613" alt="image" src="https://github.com/user-attachments/assets/1fc26a8f-5666-46c6-a7a1-daac68ceccf9" />

## **Findings: Price Distribution by Category**

### **Observable Features**

#### **Distribution by Category**
- **Median Price**: All 4 categories have very similar medians ~26-28
- **IQR Spread**: All categories have similar box heights. IQR ranges from ~20 to ~31
- **Minimum**: All categories start around 10-14
- **Maximum / Outliers**: All 4 categories have 1 outlier at Price = 50
- **Ranking by Median**: *FURNITURE > TOYS > CLOTHING > ELECTRONICS* but differences are very small

#### **Key Pattern**
- **Low Variance Between Categories**: Price distribution is nearly identical across categories
- **No Category-Specific Pricing**: No category is clearly "premium" or "budget"
- **Consistent Outliers**: Each category has high-price outliers at 50

### **Business Insights**
#### **Pricing Strategy**
- **Uniform Pricing Model**: Company uses consistent pricing tiers across all categories instead of category-based pricing.
- **No Premium Category**: *FURNITURE* does not command significantly higher prices despite higher SKU count.
- **Tiered, Not Segmented**: The 4 price peaks from the histogram ~10, 20, 30, 50 exist within every category.

#### **Inventory And Margin Implications**
- **Margin Risk**: If *FURNITURE* has higher cost but same price as *TOYS*, margins may be squeezed.
- **Cross-Category Competition**: Products compete on features, not price, since price bands overlap completely.
- **Capital Efficiency**: No category is tying up disproportionately more capital per unit based on price alone.
- <img width="987" height="590" alt="image" src="https://github.com/user-attachments/assets/ad248884-75f9-4ef0-9d8e-2faa5174594a" />

- ## **Findings: Price vs Quantity by Category**

### **Observable Features**

#### **Distribution Pattern**
- **Relationship**: [Positive / Negative / No clear relationship]
- **Clusters**: [Describe any clusters. e.g. High Price-Low Qty, Low Price-High Qty]
- **Outliers**: [Any points in top-right: High Price + High Quantity]
- **Category Separation**: [Do categories separate? e.g. FURNITURE tends to be high price]

### **Business Insights**
#### **Inventory Risk Segmentation**
- **High Price + High Quantity**: **Overstock Risk**. High capital tied up. Prioritize discounts/promos.
- **High Price + Low Quantity**: **Stockout Risk on Valuable Items**. Ensure replenishment.
- **Low Price + High Quantity**: **Fast-moving / Volume items**. Check turnover rate.
- **Low Price + Low Quantity**: **Low priority / Clearance candidates**.

#### **Operational Actions**
- **Capital Optimization**: Focus warehouse space on HighQ-LowP items, not HighQ-HighP.
- **Reorder Strategy**: Set lower reorder points for HighP-LowQ items to avoid stockouts.
- **Category Strategy**: [Which category falls in which quadrant?]
- **Warehouse Allocation**: Cross with *warehouse* to see if high-value inventory is concentrated.


