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
