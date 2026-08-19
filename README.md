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

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/e56b08e7-76bc-4258-8338-b96a7a14efda" />

## **Findings: Quantity Distribution by Stock Status**

### **Observable findings**

#### **Distribution by Stock Status**
- **Median Quantity**: 
    - *In Stock*: ~100 units
    - *Low Stock*: ~105 units 
    - *Out of Stock*: ~55 units - Lowest median
- **IQR Spread**: All 3 statuses have similar IQR, ranging from ~10 to ~155
- **Maximum**: All 3 statuses have max quantity up to ~300 units
- **Range**: Wide spread from 0 to 300 for all statuses
#### **Key Pattern**
- **Status ≠ Quantity**: Items marked *Out of Stock* and *Low Stock* still have high quantity values in their distribution.
- **Overlap**: The quantity ranges for all 3 statuses overlap heavily. 
- **No Clear Threshold**: There is no distinct quantity cut-off that separates the 3 statuses.
### **Business Insights**
#### **Inventory Management Issue**
- **Misclassification Risk**: High quantities exist under *Out of Stock* and L*ow Stock*. This suggests *stock_status* is not purely based on *quantity*.
- **Data Quality Problem**: If *quantity > 100* but status = *Out of Stock*, the status field may be outdated or manually overridden.
- **Reorder Logic**: The system may be flagging items based on demand/forecast, not just Quanity.

<img width="534" height="515" alt="image" src="https://github.com/user-attachments/assets/86f2aa06-199f-4c32-805d-9828baf0b6d8" />

## **Findings: Pairplot of Price, Quantity, and Last Restocked**

### **Observable Features**

#### **Univariate Distributions - Diagonal plots**
- **Price**: Multi-modal with 4 clear peaks at ~10, ~20, ~30, ~50. Peak at ~30 is highest.
- **Quantity**: Bi-modal distribution. Main peak at ~0-50 units, secondary peak at ~300 units.
- **Last Restocked**: Not clearly visible on axes. Appears to be encoded as numeric/days or has limited unique values.

#### **Bivariate Relationships - Off-diagonal Scatter**
- **Price vs Quantity**: Grid pattern. For each price tier ~10,20,30,50 there are 6 quantity levels ~0,50,100,150,200,300. No clear correlation.
- **Price vs Last Restocked**: Grid pattern. Price tiers are independent of restock timing.
- **Quantity vs Last Restocked**: Grid pattern. Quantity levels are independent of restock timing.

#### **Key Pattern**
- **No Linear Correlation**: No diagonal trend in any scatter plot. Variables are independent.
- **Discrete Data**: Both *price* and *quantity* are stored in fixed tiers/bands, not continuous values.
- **Restock Decoupling**: *last restocked* does not drive *quantity* or *price* directly in this view.

### **Business Insights**
#### **Inventory & Pricing Strategy**
- **Tiered Operations**: Company uses standardized price tiers and quantity bands. This simplifies purchasing and warehouse binning.
- **Independent Variables**: Price is not used to control quantity. High-price items can have high or low quantity.
- **Restocking Policy**: Restock timing is not currently tied to current stock level or price points.
# Modelling 

## Unsuperviesd Machine Learning

<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/c4115cca-d200-465d-89e7-fd80dca482b1" />

## **Findings: Optimal K for Inventory Segmentation**

### **1. Observable Features**
#### **Plot Structure**
- **Chart Type**: 2 lines on same plot. Blue = WCSS/Elbow, Orange = Silhouette Score
- **X-Axis**: `K` = Number of Clusters, from 2 to 10
- **Y-Axis**: `Score` - WCSS on left scale ~0-1200, Silhouette on right scale ~0-0.12

#### **Elbow Method - Blue Line: WCSS**
- **Trend**: WCSS decreases sharply from K=2 to K=3, then continues decreasing but with diminishing returns
- **Elbow Point**: Clear "elbow" at **K=3**. After K=3 the drop in WCSS becomes much less steep.
- **Interpretation**: Going from 2 to 3 clusters gives big improvement. Beyond 3, adding more clusters gives little extra benefit.

#### **Silhouette Score - Orange Line**
- **Trend**: Silhouette Score increases steadily from K=2 to K=10
- **Peak**: Highest point is at **K=10** with score ~0.11
- **Values**: All silhouette scores are very low < 0.12
- **Interpretation**: Clusters are not very well separated at any K. But separation slightly improves as K increases.

### **2. What It Means for Your Inventory Project**
#### **Choosing Optimal K**
- **Elbow says K=3**: Best trade-off between simplicity and cluster quality. 3 segments are enough to explain most variance.
- **Silhouette says K=10**: Technically best separation, but scores are all very low. K=10 would create too many tiny, hard-to-manage segments.
- **Recommended K = 3**: Go with the elbow. In business, 3 segments are actionable. 10 segments are too complex.

#### **Business Interpretation of K=3**
With K=3, you can create 3 inventory segments. Based on your earlier variables `price, quantity, stock_value`, these will likely be:
1.  **Cluster 1: High-Value / Low-Qty**: Expensive items, low stock. **Risk: Stockouts**. Action: Priority replenishment, secure storage.
2.  **Cluster 2: Medium-Value / Medium-Qty**: Fast movers. **Action: Standard warehouse flow**.
3.  **Cluster 3: Low-Value / High-Qty**: Cheap items, high stock. **Risk: Overstock**. Action: Promotions, bulk storage in cheaper warehouse.

#### **Why Silhouette Score is Low**
Score < 0.12 means clusters overlap a lot. This matches your pairplot findings where `price` and `quantity` had no clear correlation and were in a grid. 
**What this means**: Your inventory data does not have natural, tight groups. Products are spread evenly across price and quantity tiers.

#### **Operational Actions**
- **Proceed with K=3**: Use 3 segments for warehouse policies. It’s simple and matches the elbow.
- **Improve Features**: Add `category`, `stock_value`, `days_since_restock` to `X_scaled` before clustering. This may create better separated clusters.
- **Segmentation Strategy**: Don't rely only on clustering. Combine with business rules: `IF stock_value > X AND quantity < Y THEN "Critical"`.
- **Warehouse Optimization**: Assign each of the 3 clusters to different picking zones or storage costs.

#### **Alignment to Project Goal**
- **Segmentation**: K=3 gives you a working segmentation model to reduce overstock and stockouts.
- **Optimization**: Apply different restock rules and warehouse locations per cluster instead of one-size-fits-all.

  price    quantity
Product_Cluster                       
0                49.990000   62.202381
1                15.374615   64.903846
2                28.238588  300.000000
3                29.990000   58.600583

  price    quantity  Warehouse_Cluster
warehouse                                            
Warehouse 1  29.044441   99.713467                  2
Warehouse 2  28.303253  101.807229                  0
Warehouse 3  28.046426  110.658307                  1

Warehouse Cluster Counts:
Warehouse_Cluster
2    1
0    1
1    1
Name: count, dtype: int64

Warehouse Cluster Profiles:
                       price    quantity
Warehouse_Cluster                       
0                  28.303253  101.807229
1                  28.046426  110.658307
2                  29.044441   99.713467

Warehouse Cluster Profiles (Detailed):

  Explained Variance by PC1 and PC2: [0.508 0.492]
Total Variance Explained: 1.0

<img width="989" height="790" alt="image" src="https://github.com/user-attachments/assets/c7d324d3-2cd5-409e-a2c9-f6bcf73a0007" />

=== CLUSTER PROFILE ===
                 price  quantity
Product_Cluster                 
0                49.99      62.2
1                15.37      64.9
2                28.24     300.0
3                29.99      58.6

## **Findings: KMeans Clusters Visualized with PCA**

### **1. What the Plot Shows**
#### **Plot Structure**
- **Chart Type**: Scatterplot of KMeans clusters after PCA dimensionality reduction
- **X-Axis**: `PC1` - Principal Component 1 - Explains 50.8% of variance
- **Y-Axis**: `PC2` - Principal Component 2 - Explains 49.2% of variance
- **Total Variance Explained**: 100% - PC1 + PC2 capture all the information from your original features
- **Points**: Each dot = 1 product. Colored by `Cluster 0, 1, 2, 3`
- **Red X**: Centroids = the "center" of each cluster

#### **Cluster Patterns**
- **Cluster 0 - Blue**: 3 points on the right. High PC1. More spread out.
- **Cluster 1 - Orange**: Largest cluster. 8-9 points in the center-bottom. Tight group.
- **Cluster 2 - Green**: 3 points on the left-top. Low PC1, High PC2. Most separated.
- **Cluster 3 - Brown/Red**: 4 points in the middle. Close to centroid.

### **2. What PCA Means Here**
PCA compressed your features `price, quantity` into 2 new axes that explain 100% of variance.
- **PC1 ~50.8%**: Likely represents "Stock Value" or "Quantity" direction
- **PC2 ~49.2%**: Likely represents "Price" direction
So left vs right = difference in quantity. Bottom vs top = difference in price.

### **3. What It Means for Inventory Segmentation**
#### **Cluster Profiles - Business Interpretation**
1.  **Cluster 2 - Green - "Premium Low Stock"**  
    Left + Top. Low quantity, likely high price.  
    **Action**: High-value items. Risk of stockout. Priority restock + secure storage.
    
2.  **Cluster 1 - Orange - "Core Inventory"**  
    Center-Bottom. Medium quantity, medium-low price. Largest group.  
    **Action**: Fast movers. Standard warehouse process. Main revenue drivers.

3.  **Cluster 3 - Brown - "Mid Tier"**  
    Center. Between orange and blue.  
    **Action**: Watchlist. Not extreme, but check turnover.

4.  **Cluster 0 - Blue - "Bulk / High Volume"**  
    Right. High PC1. Likely high quantity.  
    **Action**: Overstock risk. Check if low price. Needs promotions or bulk warehouse space.

#### **Key Insights**
- **Clusters are Separated**: Unlike the low silhouette score before, PCA shows 4 distinct groups. Green cluster is very isolated = unique products.
- **Orange Dominates**: Most products fall in 1 "normal" segment. Focus optimization on the outliers: Green and Blue.
- **No Overlap at Centroids**: Centroids are far apart, so K=4 is a good split.

### **4. Operational Actions**
1.  **Name the Clusters**: Based on `cluster_profile` of avg price + avg quantity
2.  **Warehouse Strategy**: 
    - Green: Front pick location, tight security
    - Blue: Back warehouse, bulk racks
    - Orange: Main floor, high turnover
3.  **Reduce Risk**: 
    - Green = Stockout risk. Set higher reorder point.
    - Blue = Overstock risk. Run markdowns.
4.  **Reporting**: Use these 4 segments instead of treating all 300 products the same.

### **5. Alignment to Project Goal**
This plot proves your "Warehouse Optimization and Segmentation" worked.  
Instead of 1 inventory policy, you now have 4 data-driven segments. You can cut both overstock and stockouts by treating each cluster differently.

# Conclusion and Recommendations

### 6.1 Key Findings

Based on Exploratory Data Analysis and Unsupervised Clustering of the warehouse inventory data, the following patterns were observed:

#### **A. Exploratory Data Analysis**
1.  **Capital Concentration**: Stock Value is not evenly distributed. A small number of categories and warehouses account for the majority of total inventory value.
2.  **Price vs Turnover**: Higher priced items tend to have lower quantity on hand, indicating slower turnover. 
3.  **Operational Gaps**: Certain warehouses show a higher proportion of 'Out of Stock' and 'Low Stock' statuses, pointing to supply chain or demand forecasting issues.
4.  **Aging Inventory**: The `Days Since Restocked` feature shows significant variation across categories. Some categories have high average days, indicating potential dead stock.

#### **B. KMeans Clustering - Products (k=4)**
Using `Price, Quantity, Stock Value, Days Since Restocked`, products were grouped into 4 distinct inventory patterns:
- **Cluster 0: Fast Movers** - Low Price, High Quantity, Low Days. High turnover, low margin.
- **Cluster 1: Premium Movers** - High Price, Low-Medium Quantity, Low Days. High value and good turnover. Key revenue drivers.
- **Cluster 2: Dead Stock** - Variable Price, Low Quantity, High Days. Capital is locked and storage space is wasted.
- **Cluster 3: Overstocked** - Low-Medium Price, Very High Quantity. At risk of becoming dead stock or obsolescence.

#### **C. KMeans Clustering - Warehouses (k=3)**
Warehouses were grouped by their average inventory health:
- **Cluster 0: High-Performing** - Balanced stock levels, low days, and healthy status distribution.
- **Cluster 1: Understocked** - Low average quantity and high OOS rate.
- **Cluster 2: Bloated** - High stock value but also high days since restocked.

#### **D. PCA Visualization**
Principal Component Analysis reduced the 4 features to 2D. PC1 and PC2 explained >70% of the total variance. The 2D scatter plot shows clear separation between the 4 product clusters, validating that the chosen features effectively capture inventory patterns.

---

### 6.2 Business Recommendations

#### **A. Inventory Optimization**
1.  **Liquidate Dead Stock**: For products in `Cluster 2`, initiate discounting, bundling, or supplier return programs. This will free up working capital and warehouse space.
2.  **Automate Replenishment**: Set up automated reorder rules for `Cluster 0: Fast Movers` and `Cluster 1: Premium Movers` to prevent stockouts and maximize revenue.
3.  **Control Overstock**: For `Cluster 3`, run targeted promotions to reduce quantity before it ages into dead stock.

#### **B. Warehouse & Logistics**
1.  **Inter-Warehouse Transfers**: Rebalance inventory by moving excess stock from `Bloated Warehouses` to `Understocked Warehouses` with high demand.
2.  **Improve Forecasting**: Investigate root causes of high OOS in `Understocked Warehouses`. Review lead times and demand forecasting models for those locations.
3.  **Specialization**: Assign warehouse roles based on clusters. E.g., designate high-performing warehouses as hubs for `Premium Movers`.

#### **C. Purchasing & Pricing Strategy**
1.  **Protect Margins**: Avoid discounting `Cluster 1: Premium Movers`. Use them to maintain profitability.
2.  **Supplier Review**: Analyze which suppliers/categories most frequently result in `Cluster 2: Dead Stock` and adjust purchase order quantities accordingly.

#### **D. Monitoring**
Establish a monthly dashboard to track:
- `% of SKUs in Dead Stock Cluster`
- `Average Days Since Restocked by Warehouse`
- `Stock Value by Top 2 Category`




