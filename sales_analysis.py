# Step 1: Import the necessary libraries
import pandas as pd
import numpy as np

# Let's CREATE our own sample sales data that is perfect for analysis
# This is a common technique for practice and ensures it works offline

print("Creating a sample sales dataset...")

# Create a list of dates for the past 3 years
dates = pd.date_range(start='2021-01-01', end='2023-12-31', freq='D')
np.random.seed(42)  # For reproducible results

# Generate sample data
data = {
    'Order_ID': [f'ORD_{i:05d}' for i in range(1, 1001)],
    'Order_Date': np.random.choice(dates, 1000),
    'Customer_ID': [f'CUST_{np.random.randint(100, 500):03d}' for _ in range(1000)],
    'Product_Category': np.random.choice(['Office Supplies', 'Technology', 'Furniture'], 1000),
    'Product': np.random.choice(['Paper', 'Notebooks', 'Desks', 'Chairs', 'Phones', 'Computers'], 1000),
    'Sales_Amount': np.round(np.random.uniform(10, 2000, 1000), 2),
    'Quantity': np.random.randint(1, 10, 1000),
    'Profit': np.round(np.random.uniform(-50, 500, 1000), 2),  # Sometimes losses happen!
    'Region': np.random.choice(['East', 'West', 'South', 'Central'], 1000)
}

# Create the DataFrame
df = pd.DataFrame(data)

# Calculate a new column: Profit per Item
df['Profit_Per_Item'] = np.round(df['Profit'] / df['Quantity'], 2)

# Save this data to a CSV file in our project folder
df.to_csv('SampleSuperstore.csv', index=False)

# Step 4: Print a success message and show the first 5 rows of data
print("Sample dataset created and saved successfully as 'SampleSuperstore.csv'!")
print("\nHere's a preview of your data:")
print(df.head())
print(f"\nThe dataset has {len(df)} rows and {len(df.columns)} columns.")

# --- STEP 5: Initial Data Exploration ---
print("\n" + "="*50)
print("STEP 5: Initial Data Exploration")
print("="*50)

# 1. Load the dataset we just created
print("1. Loading the dataset 'SampleSuperstore.csv'...")
df = pd.read_csv('SampleSuperstore.csv')

# 2. Check the basic information about the DataFrame
print("\n2. Dataset Info:")
print(f"   - Shape: {df.shape} (Rows, Columns)")
print(f"   - Columns: {list(df.columns)}")

# 3. Display the first 5 rows to see the data
print("\n3. First 5 rows of the dataset:")
print(df.head())

# 4. Get a quick statistical summary of numerical columns
print("\n4. Statistical Summary (Numerical Columns):")
print(df[['Sales_Amount', 'Quantity', 'Profit', 'Profit_Per_Item']].describe())

# 5. Check for any missing values in the dataset
print("\n5. Checking for missing values:")
print(df.isnull().sum())

print("\nInitial exploration complete! Ready for the next step.")


# --- STEP 6: Data Cleaning and Preparation ---
print("\n" + "="*50)
print("STEP 6: Data Cleaning and Preparation")
print("="*50)

# 1. Convert 'Order_Date' from a string to a DateTime object
print("1. Converting 'Order_Date' to DateTime format...")
df['Order_Date'] = pd.to_datetime(df['Order_Date'])

# 2. Create a new 'Year' column from the Order_Date
print("2. Creating a new 'Year' column...")
df['Year'] = df['Order_Date'].dt.year

# 3. Create a new 'Month' column from the Order_Date
print("3. Creating a new 'Month' column...")
df['Month'] = df['Order_Date'].dt.month

# 4. Verify the new columns and data types
print("\n4. Updated Dataset Info:")
print(f"   - New shape: {df.shape} (Rows, Columns)")
print(f"   - New columns: {list(df.columns)}")
print(f"   - Data types:\n{df.dtypes}")

# 5. Show a preview of the data with the new columns
print("\n5. Preview with new date-related columns:")
print(df[['Order_ID', 'Order_Date', 'Year', 'Month', 'Sales_Amount']].head())

print("\nData cleaning and preparation complete! Ready for analysis.")


# --- STEP 7: Exploratory Data Analysis (EDA) ---
print("\n" + "="*50)
print("STEP 7: Exploratory Data Analysis")
print("="*50)

# Import the visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for the plots
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# 1. Analyze Total Sales by Product Category
print("1. Analyzing Total Sales by Product Category...")
sales_by_category = df.groupby('Product_Category')['Sales_Amount'].sum().sort_values(ascending=False)
print(sales_by_category)

# 2. Create a Bar Chart for Sales by Category
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
sales_by_category.plot(kind='bar', color=['skyblue', 'lightgreen', 'lightcoral'])
plt.title('Total Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales Amount ($)')
plt.xticks(rotation=45)

# 3. Analyze Total Profit by Region
print("\n2. Analyzing Total Profit by Region...")
profit_by_region = df.groupby('Region')['Profit'].sum().sort_values(ascending=False)
print(profit_by_region)

# 4. Create a Bar Chart for Profit by Region
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
profit_by_region.plot(kind='bar', color=['gold', 'lightblue', 'lightpink', 'lightgreen'])
plt.title('Total Profit by Region')
plt.xlabel('Region')
plt.ylabel('Total Profit ($)')
plt.xticks(rotation=45)

# Adjust layout and display the plots
plt.tight_layout()
plt.savefig('sales_profit_analysis.png')  # Save the plot as an image file
print("\n3. Visualization saved as 'sales_profit_analysis.png'")
# plt.show()  # This would display the plot in a window, but can be skipped for now

# 5. Find the Top 5 Most Profitable Products
print("\n4. Top 5 Most Profitable Products:")
top_5_profitable = df.groupby('Product')['Profit'].sum().nlargest(5)
print(top_5_profitable)

# 6. Check the overall Profit Margin
print("\n5. Overall Profit Margin Analysis:")
total_sales = df['Sales_Amount'].sum()
total_profit = df['Profit'].sum()
profit_margin = (total_profit / total_sales) * 100
print(f"   - Total Sales: ${total_sales:,.2f}")
print(f"   - Total Profit: ${total_profit:,.2f}")
print(f"   - Profit Margin: {profit_margin:.2f}%")

print("\nBasic exploratory analysis complete! Check your project folder for the chart.")


# --- STEP 8: Machine Learning - Customer Segmentation ---
print("\n" + "="*60)
print("STEP 8: Machine Learning - Customer Segmentation")
print("="*60)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("1. Preparing data for customer segmentation...")

# First, create a summary of each customer's behavior
customer_summary = df.groupby('Customer_ID').agg({
    'Sales_Amount': 'sum',       # Total money spent (Monetary)
    'Order_ID': 'count',         # Number of orders (Frequency)
    'Profit': 'mean'             # Average profit per order
}).rename(columns={'Sales_Amount': 'Total_Spent', 
                   'Order_ID': 'Order_Count',
                   'Profit': 'Avg_Profit'})

# Reset index to make Customer_ID a column again
customer_summary = customer_summary.reset_index()

# Show what the new customer data looks like
print("\n   Customer Behavior Summary (First 5 customers):")
print(customer_summary.head())

# Select the features for clustering: Total spent and order count
X = customer_summary[['Total_Spent', 'Order_Count']]

# Scale the data (important for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("2. Applying K-Means clustering algorithm...")

# Apply K-Means to find 3 customer segments
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_summary['Customer_Segment'] = kmeans.fit_predict(X_scaled)

# See how many customers are in each segment
segment_counts = customer_summary['Customer_Segment'].value_counts().sort_index()
print(f"\n   Customers per segment: {dict(segment_counts)}")

# Analyze the characteristics of each segment
segment_analysis = customer_summary.groupby('Customer_Segment').agg({
    'Total_Spent': 'mean',
    'Order_Count': 'mean',
    'Avg_Profit': 'mean',
    'Customer_ID': 'count'
}).round(2)

print("\n3. Customer Segment Analysis:")
print(segment_analysis)

# Create a visualization of the segments
plt.figure(figsize=(10, 6))
scatter = plt.scatter(customer_summary['Total_Spent'], 
                     customer_summary['Order_Count'],
                     c=customer_summary['Customer_Segment'], 
                     cmap='viridis', alpha=0.7)
plt.colorbar(scatter, label='Customer Segment')
plt.title('Customer Segmentation based on Spending & Frequency')
plt.xlabel('Total Amount Spent ($)')
plt.ylabel('Number of Orders')
plt.grid(True)
plt.savefig('customer_segmentation.png')
print("\n4. Customer segmentation visualization saved as 'customer_segmentation.png'")

print("\n5. Interpreting the segments:")
print("   - Segment 0: Low spending, low frequency (Occasional shoppers)")
print("   - Segment 1: High spending, high frequency (Best customers)")
print("   - Segment 2: Medium spending, medium frequency (Regular customers)")

# Merge this segment information back to the original dataframe
df = df.merge(customer_summary[['Customer_ID', 'Customer_Segment']], 
              on='Customer_ID', how='left')

print("\nCustomer segmentation complete! Machine learning model applied successfully.")


# --- STEP 9: Final Analysis Summary & Export ---
print("\n" + "="*60)
print("STEP 9: Final Analysis Summary & Export")
print("="*60)

print("Generating comprehensive business insights...")

# 1. Key Performance Indicators (KPIs)
total_customers = df['Customer_ID'].nunique()
total_orders = df['Order_ID'].nunique()
avg_order_value = df['Sales_Amount'].mean()
best_segment_size = customer_summary[customer_summary['Customer_Segment'] == 1].shape[0]

print("\n" + "🔑 KEY BUSINESS INSIGHTS")
print("-" * 40)
print(f"• Total Customers: {total_customers}")
print(f"• Total Orders: {total_orders}")
print(f"• Average Order Value: ${avg_order_value:.2f}")
print(f"• Best Customers Segment Size: {best_segment_size} customers")
print(f"• Overall Profit Margin: {profit_margin:.2f}%")

# 2. Top Performing Categories
top_category = sales_by_category.index[0]
top_category_sales = sales_by_category.iloc[0]

print(f"\n📈 TOP PERFORMERS")
print("-" * 40)
print(f"• Highest Selling Category: {top_category} (${top_category_sales:,.2f})")
print(f"• Most Profitable Region: {profit_by_region.index[0]} (${profit_by_region.iloc[0]:,.2f})")
print(f"• Top Product: {top_5_profitable.index[0]} (${top_5_profitable.iloc[0]:,.2f})")

# 3. Customer Segmentation Summary
best_customers = customer_summary[customer_summary['Customer_Segment'] == 1]
avg_best_customer_value = best_customers['Total_Spent'].mean()

print(f"\n👥 CUSTOMER SEGMENTATION ANALYSIS")
print("-" * 40)
print(f"• Best Customers: {len(best_customers)} customers")
print(f"• Average Value of Best Customer: ${avg_best_customer_value:.2f}")
print(f"• These customers represent your most valuable segment!")

# 4. Business Recommendations
print(f"\n💡 RECOMMENDATIONS FOR GROWTH")
print("-" * 40)
print("1. Focus marketing efforts on the 'Best Customers' segment")
print("2. Increase inventory for top-performing product categories")
print("3. Study successful strategies in your highest-profit region")
print("4. Create loyalty programs to move customers to higher segments")

# 5. Export the Final Analyzed Dataset
final_filename = 'analyzed_sales_data.csv'
df.to_csv(final_filename, index=False)
print(f"\n💾 EXPORT COMPLETE")
print("-" * 40)
print(f"Final analyzed dataset saved as: {final_filename}")
print(f"This file contains all original data plus:")
print("   - Year/Month columns")
print("   - Customer segmentation labels")
print("   - Ready for further analysis or reporting")

print("\n" + "="*60)
print("🎉 PROJECT COMPLETE! 🎉")
print("="*60)
print("I have successfully:")
print("✓ Cleaned and prepared data")
print("✓ Performed exploratory data analysis")
print("✓ Created data visualizations")
print("✓ Applied machine learning (K-Means clustering)")
print("✓ Generated actionable business insights")
print("✓ Created a portfolio-ready project!")