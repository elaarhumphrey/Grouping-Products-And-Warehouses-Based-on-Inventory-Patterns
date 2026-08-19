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
