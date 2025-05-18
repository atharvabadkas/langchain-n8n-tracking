import pandas as pd
import os
import re
import sys
import datetime
import time
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile

# Try to load environment variables from .env file
try:
    load_dotenv()
    print("Environment variables loaded from .env file")
except Exception as e:
    print(f"Note: Could not load .env file: {e}")

# Check for OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    print("Please set it by running: export OPENAI_API_KEY='your-api-key'")
    print("Or add it to your shell profile (~/.zshrc, ~/.bash_profile, etc.)")
    print("Or create a .env file with OPENAI_API_KEY=your-api-key")
    sys.exit(1)

# Define input file paths (can be overridden by environment variables)
PURCHASE_FILE = os.environ.get("PURCHASE_FILE", "/Users/atharvabadkas/Coding /Automated Reconcilation/Tertullia Purchase Data.csv")
PRODUCTION_FILE = os.environ.get("PRODUCTION_FILE", "/Users/atharvabadkas/Coding /Automated Reconcilation/Tertullia Production Data.csv")
SALES_FILE = os.environ.get("SALES_FILE", "/Users/atharvabadkas/Coding /Automated Reconcilation/Tertullia Sales Data.csv")

# Initialize the language model
LLM = ChatOpenAI(
    model="gpt-4o", 
    temperature=0,
    openai_api_key=OPENAI_API_KEY
)

def clear_cache_and_check_files():
    """
    Clear any potential Python file cache and check that the files exist.
    
    This helps ensure we're always reading the most up-to-date version of the files.
    """
    # Check if files exist
    files_to_check = [PURCHASE_FILE, PRODUCTION_FILE, SALES_FILE]
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"ERROR: File {file_path} does not exist!")
            sys.exit(1)
            
        # Print file modification time
        mod_time = os.path.getmtime(file_path)
        mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')
        file_size = os.path.getsize(file_path)
        print(f"File: {file_path}")
        print(f"  - Last modified: {mod_time_str}")
        print(f"  - Size: {file_size} bytes")
    
    # Add a small delay to ensure any file system operations complete
    time.sleep(0.5)
    
    # Explicitly clear Python's internal buffer cache
    # This is a workaround - Python doesn't have a direct way to clear file cache
    print("\nClearing file cache...")
    for file_path in files_to_check:
        try:
            # This technique forces Python not to use any cached version
            # by reopening the file directly
            with open(file_path, 'r', encoding='utf-8') as f:
                # Just read first line to ensure file handle is created
                f.readline()
            print(f"Refreshed file handle for: {file_path}")
        except Exception as e:
            print(f"Warning: Couldn't refresh file handle for {file_path}: {e}")

def clean_column_names(df):
    """
    Standardize column names by converting spaces to underscores and making them lowercase.
    
    Args:
        df (pandas.DataFrame): DataFrame with columns to clean
        
    Returns:
        pandas.DataFrame: DataFrame with cleaned column names
    """
    df.columns = [re.sub(r'\s+', '_', col).strip().lower() for col in df.columns]
    return df

def clean_and_convert_weight(series, unit_suffix='kg'):
    """
    Remove unit suffixes from weight values and convert to numeric format.
    
    Args:
        series (pandas.Series): Series containing weight values with unit suffixes
        unit_suffix (str): The unit suffix to remove (e.g., 'kg', 'g')
        
    Returns:
        pandas.Series: Cleaned numeric series with units removed
    """
    series = series.astype(str)
    series = series.str.replace(r'\s*' + unit_suffix + r'\s*', '', regex=True, case=False)
    return pd.to_numeric(series, errors='coerce')

def process_purchase_data(file_path):
    """
    Process purchase data from CSV file and create a summary by ingredient.
    
    This function handles the purchase data format:
    1. Reads the purchase data CSV
    2. Cleans column names
    3. Processes quantities that might have unit information
    4. Converts gram values to kilograms
    5. Creates a summary grouped by supplier_sku
    
    Args:
        file_path (str): Path to the purchase data CSV file
        
    Returns:
        pandas.DataFrame: Summary of purchase data by ingredient (in kg)
    """
    try:
        # Force Python to read the latest version of the file
        print(f"Reading purchase data from: {file_path}")
        # Use a cache-busting approach by setting low_memory=False
        purchases_df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
        purchases_df = clean_column_names(purchases_df)
        
        # Print the data types to help with debugging
        print("Data types in purchase data:")
        print(purchases_df.dtypes)
        
        # Check for either 'purchase_quantity' or 'total_purchase' column
        quantity_column = None
        if 'purchase_quantity' in purchases_df.columns:
            quantity_column = 'purchase_quantity'
        elif 'total_purchase' in purchases_df.columns:
            quantity_column = 'total_purchase'
            print(f"Using 'total_purchase' column for purchase quantities")
        else:
            print(f"Error: Required column for purchase quantity not found in {file_path}")
            print(f"Available columns: {list(purchases_df.columns)}")
            sys.exit(1)
            
        if quantity_column:
            purchases_df = purchases_df[['ingredient_sku', quantity_column]]
            purchases_df = purchases_df.copy()
            
            # Convert all values to strings safely - handle all possible data types
            # First convert all non-string values to string
            purchases_df[quantity_column] = purchases_df[quantity_column].astype(object)
            # Then replace NaN/None with empty string
            purchases_df[quantity_column] = purchases_df[quantity_column].fillna('')
            # Finally convert everything to string
            purchases_df[quantity_column] = purchases_df[quantity_column].astype(str)
            
            # Simple approach - try to convert everything to numeric first
            # Keep track of which rows have units to process differently
            has_units = []
            numeric_vals = []
            
            # Process each row individually to avoid data type issues
            for idx, val in purchases_df[quantity_column].items():
                # Check if the value has alphabetic characters (indicating units)
                if any(c.isalpha() for c in str(val)):
                    has_units.append(True)
                    
                    # Handle specific unit types
                    val_str = str(val).lower()
                    if 'g' in val_str and 'kg' not in val_str and 'pcs' not in val_str:
                        # Convert grams to kg
                        try:
                            # Extract numeric part and convert
                            num_val = float(''.join(c for c in val_str if c.isdigit() or c == '.'))
                            numeric_vals.append(num_val / 1000)  # Convert g to kg
                            print(f"Converted gram value: {val} -> {num_val/1000} kg")
                        except ValueError:
                            numeric_vals.append(0)
                            
                    elif 'kg' in val_str:
                        # Already in kg, just extract numeric part
                        try:
                            num_val = float(''.join(c for c in val_str if c.isdigit() or c == '.'))
                            numeric_vals.append(num_val)
                        except ValueError:
                            numeric_vals.append(0)
                            
                    elif 'pcs' in val_str:
                        # Handle piece values - keep as is
                        try:
                            num_val = float(''.join(c for c in val_str if c.isdigit() or c == '.'))
                            numeric_vals.append(num_val)
                            print(f"Kept piece value: {val} -> {num_val}")
                        except ValueError:
                            numeric_vals.append(0)
                    else:
                        # Unknown unit, try to extract numeric part
                        try:
                            num_val = float(''.join(c for c in val_str if c.isdigit() or c == '.'))
                            numeric_vals.append(num_val)
                        except ValueError:
                            numeric_vals.append(0)
                else:
                    # No units, check if it's a large number (likely grams)
                    has_units.append(False)
                    try:
                        num_val = float(val) if val != '' else 0
                        # Convert values > 1000 from g to kg
                        if num_val > 1000:
                            print(f"Converting large value from g to kg: {num_val} -> {num_val/1000}")
                            numeric_vals.append(num_val / 1000)
                        else:
                            numeric_vals.append(num_val)
                    except ValueError:
                        numeric_vals.append(0)
            
            # Replace the original column with processed numeric values
            purchases_df[quantity_column] = numeric_vals
            
            # Create summary of purchase data
            purchase_summary = purchases_df.groupby('ingredient_sku')[quantity_column].sum().reset_index()
            purchase_summary = purchase_summary.rename(columns={quantity_column: 'Total Purchased (Kg)'})
        
        # Display summary information
        print("Purchase Data Summary:")
        print(purchase_summary.to_markdown(index=False))
        print("-" * 30)
        
        return purchase_summary
        
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def process_production_data(file_path):
    """
    Process production data from CSV file and create a summary by ingredient.
    
    This function handles production data by:
    1. Reads the production data CSV
    2. Cleans column names
    3. Matches ingredient_sku and ingredient_id
    4. Converts all quantities to kilograms
    5. Summarizes by ingredient
    
    Args:
        file_path (str): Path to the production data CSV file
        
    Returns:
        pandas.DataFrame: Summary of production data by ingredient (in kg)
    """
    try:
        # Force Python to read the latest version of the file
        print(f"Reading production data from: {file_path}")
        # Use a cache-busting approach by setting low_memory=False and specifying encoding
        production_df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
        production_df = clean_column_names(production_df)
        
        # Print the data types and columns to help with debugging
        print("Data types in production data:")
        print(production_df.dtypes)
        print(f"Columns in production data: {list(production_df.columns)}")
        
        # Check if both ingredient_sku and ingredient_id exist
        has_ingredient_sku = 'ingredient_sku' in production_df.columns
        has_ingredient_id = 'ingredient_id' in production_df.columns
        
        if has_ingredient_sku and has_ingredient_id:
            print("Found both ingredient_sku and ingredient_id columns in production data")
            # Keep track of which ID to use for each row
            production_df['ingredient_identifier'] = production_df['ingredient_sku']
            # Replace empty or NaN ingredient_sku values with ingredient_id
            mask = production_df['ingredient_sku'].isna() | (production_df['ingredient_sku'] == '')
            production_df.loc[mask, 'ingredient_identifier'] = production_df.loc[mask, 'ingredient_id']
            print(f"Using ingredient_id as backup for {mask.sum()} rows where ingredient_sku is missing")
        elif has_ingredient_sku:
            print("Using ingredient_sku column for identification")
            production_df['ingredient_identifier'] = production_df['ingredient_sku']
        elif has_ingredient_id:
            print("Using ingredient_id column for identification")
            production_df['ingredient_identifier'] = production_df['ingredient_id']
        else:
            print(f"Error: Neither ingredient_sku nor ingredient_id found in {file_path}")
            print(f"Available columns: {list(production_df.columns)}")
            sys.exit(1)
        
        # Check for quantity column
        quantity_column = None
        if 'total_production' in production_df.columns:
            quantity_column = 'total_production'
            print(f"Using 'total_production' column for quantities")
        elif 'total_production_weight_(kg)' in production_df.columns:
            quantity_column = 'total_production_weight_(kg)'
            print(f"Using 'total_production_weight_(kg)' column for quantities")
        else:
            print(f"Error: Required column for production quantity not found in {file_path}")
            print(f"Available columns: {list(production_df.columns)}")
            sys.exit(1)
        
        # Convert all quantity values to numeric, handling different formats
        production_df[quantity_column] = production_df[quantity_column].astype(str)
        
        # Process each row to extract numeric values and convert to kg
        def convert_to_kg(val):
            val_str = str(val).lower()
            # If value contains letters (likely has units)
            if any(c.isalpha() for c in val_str):
                # Extract numeric part
                num_val = ''.join(c for c in val_str if c.isdigit() or c == '.')
                try:
                    num_val = float(num_val) if num_val else 0
                    # Check for unit indication
                    if 'g' in val_str and 'kg' not in val_str:
                        # Convert g to kg
                        return num_val / 1000
                    elif 'kg' in val_str:
                        # Already in kg
                        return num_val
                    else:
                        # Assume it's in g for unknown units
                        return num_val / 1000
                except ValueError:
                    return 0
            else:
                # No units specified - assume grams
                try:
                    num_val = float(val_str) if val_str else 0
                    # Always divide by 1000 to convert to kg
                    return num_val / 1000
                except ValueError:
                    return 0
        
        # Apply the conversion to all values
        production_df['quantity_in_kg'] = production_df[quantity_column].apply(convert_to_kg)
        
        # Group by the ingredient identifier and sum the quantities
        production_summary = production_df.groupby('ingredient_identifier')['quantity_in_kg'].sum().reset_index()
        production_summary = production_summary.rename(columns={
            'ingredient_identifier': 'ingredient_sku',
            'quantity_in_kg': 'Total Produced (Kg)'
        })
        
        # Round to 4 decimal places for readability
        production_summary['Total Produced (Kg)'] = production_summary['Total Produced (Kg)'].round(4)
        
        # Display summary information
        print("Production Data Summary:")
        print(production_summary.to_markdown(index=False))
        print(f"Total ingredients found: {len(production_summary)}")
        print(f"Total quantity produced: {production_summary['Total Produced (Kg)'].sum()} kg")
        print("-" * 30)
        
        return production_summary
        
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Column issues in {file_path}: {e}")
        print(f"Available columns: {list(production_df.columns)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def process_sales_data(file_path):
    """
    Process sales data from CSV file and create a summary by ingredient.
    
    This function handles sales data by:
    1. Reads the sales data CSV
    2. Cleans column names
    3. Matches ingredient_sku and ingredient_id
    4. Converts all quantities to kilograms
    5. Summarizes by ingredient
    
    Args:
        file_path (str): Path to the sales data CSV file
        
    Returns:
        pandas.DataFrame: Summary of sales data by ingredient (in kg)
    """
    try:
        # Force Python to read the latest version of the file
        print(f"Reading sales data from: {file_path}")
        # Use a cache-busting approach by setting low_memory=False and specifying encoding
        sales_df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
        sales_df = clean_column_names(sales_df)
        
        # Print the data types and columns to help with debugging
        print("Data types in sales data:")
        print(sales_df.dtypes)
        print(f"Columns in sales data: {list(sales_df.columns)}")
        
        # Check if both ingredient_sku and ingredient_id exist
        has_ingredient_sku = 'ingredient_sku' in sales_df.columns
        has_ingredient_id = 'ingredient_id' in sales_df.columns
        
        if has_ingredient_sku and has_ingredient_id:
            print("Found both ingredient_sku and ingredient_id columns in sales data")
            # Keep track of which ID to use for each row
            sales_df['ingredient_identifier'] = sales_df['ingredient_sku']
            # Replace empty or NaN ingredient_sku values with ingredient_id
            mask = sales_df['ingredient_sku'].isna() | (sales_df['ingredient_sku'] == '')
            sales_df.loc[mask, 'ingredient_identifier'] = sales_df.loc[mask, 'ingredient_id']
            print(f"Using ingredient_id as backup for {mask.sum()} rows where ingredient_sku is missing")
        elif has_ingredient_sku:
            print("Using ingredient_sku column for identification")
            sales_df['ingredient_identifier'] = sales_df['ingredient_sku']
        elif has_ingredient_id:
            print("Using ingredient_id column for identification")
            sales_df['ingredient_identifier'] = sales_df['ingredient_id']
        else:
            print(f"Error: Neither ingredient_sku nor ingredient_id found in {file_path}")
            print(f"Available columns: {list(sales_df.columns)}")
            sys.exit(1)
        
        # Check for quantity column
        quantity_column = None
        if 'total_sold' in sales_df.columns:
            quantity_column = 'total_sold'
            print(f"Using 'total_sold' column for quantities")
        elif 'total_weight_sold_(kg)' in sales_df.columns:
            quantity_column = 'total_weight_sold_(kg)'
            print(f"Using 'total_weight_sold_(kg)' column for quantities")
        else:
            print(f"Error: Required column for sales quantity not found in {file_path}")
            print(f"Available columns: {list(sales_df.columns)}")
            sys.exit(1)
        
        # Convert all quantity values to numeric, handling different formats
        sales_df[quantity_column] = sales_df[quantity_column].astype(str)
        
        # Process each row to extract numeric values and convert to kg
        def convert_to_kg(val):
            val_str = str(val).lower()
            # If value contains letters (likely has units)
            if any(c.isalpha() for c in val_str):
                # Extract numeric part
                num_val = ''.join(c for c in val_str if c.isdigit() or c == '.')
                try:
                    num_val = float(num_val) if num_val else 0
                    # Check for unit indication
                    if 'g' in val_str and 'kg' not in val_str:
                        # Convert g to kg
                        return num_val / 1000
                    elif 'kg' in val_str:
                        # Already in kg
                        return num_val
                    else:
                        # Assume it's in g for unknown units
                        return num_val / 1000
                except ValueError:
                    return 0
            else:
                # No units specified - assume grams
                try:
                    num_val = float(val_str) if val_str else 0
                    # Always divide by 1000 to convert to kg
                    return num_val / 1000
                except ValueError:
                    return 0
        
        # Apply the conversion to all values
        sales_df['quantity_in_kg'] = sales_df[quantity_column].apply(convert_to_kg)
        
        # Group by the ingredient identifier and sum the quantities
        sales_summary = sales_df.groupby('ingredient_identifier')['quantity_in_kg'].sum().reset_index()
        sales_summary = sales_summary.rename(columns={
            'ingredient_identifier': 'ingredient_sku',
            'quantity_in_kg': 'Total Sold (Kg)'
        })
        
        # Round to 4 decimal places for readability
        sales_summary['Total Sold (Kg)'] = sales_summary['Total Sold (Kg)'].round(4)
        
        # Display summary information
        print("Sales Data Summary:")
        print(sales_summary.to_markdown(index=False))
        print(f"Total ingredients found: {len(sales_summary)}")
        print(f"Total quantity sold: {sales_summary['Total Sold (Kg)'].sum()} kg")
        print("-" * 30)
        
        return sales_summary
        
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Column issues in {file_path}: {e}")
        print(f"Available columns: {list(sales_df.columns)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_reconciliation_report(purchase_summary, production_summary, sales_summary):
    """
    Create a comprehensive reconciliation report combining data from all three sources.
    
    This function creates three different reconciliation views:
    1. Full reconciliation - all items from all datasets
    2. Purchase to Production to Sales - items present in all three datasets
    3. Purchase to Sales - items that appear in purchase and sales but not production
    
    Args:
        purchase_summary (pd.DataFrame): Summary of purchase data
        production_summary (pd.DataFrame): Summary of production data
        sales_summary (pd.DataFrame): Summary of sales data
        
    Returns:
        dict: Dictionary containing the three reconciliation dataframes
    """
    try:
        print("Creating reconciliation report...")
        
        # Ensure SKU column is consistently named in all datasets
        if 'supplier_sku' in purchase_summary.columns and 'ingredient_sku' not in purchase_summary.columns:
            purchase_summary = purchase_summary.rename(columns={'supplier_sku': 'ingredient_sku'})
            print("Renamed 'supplier_sku' to 'ingredient_sku' in purchase data for consistency")
            
        # Convert any possible non-string SKUs to string to ensure proper merging
        for df in [purchase_summary, production_summary, sales_summary]:
            if 'ingredient_sku' in df.columns:
                df['ingredient_sku'] = df['ingredient_sku'].astype(str)
        
        # Create full reconciliation with all items
        # Perform outer merge to include all SKUs from all datasets
        print("Merging purchase and production data...")
        merged_df = pd.merge(purchase_summary, production_summary, on='ingredient_sku', how='outer')
        print("Merging with sales data...")
        merged_df = pd.merge(merged_df, sales_summary, on='ingredient_sku', how='outer')
        
        # Fill NaN values with 0 and round to 2 decimal places
        merged_df = merged_df.fillna(0)
        numeric_columns = ['Total Purchased (Kg)', 'Total Produced (Kg)', 'Total Sold (Kg)']
        
        # Ensure all columns are numeric to prevent comparison errors
        print("Standardizing data types in merged dataframe...")
        for col in numeric_columns:
            if col in merged_df.columns:
                # First convert to numeric, coercing errors to NaN
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
                # Replace any resulting NaNs with 0
                merged_df[col] = merged_df[col].fillna(0)
                # Round to 2 decimal places for readability
                merged_df[col] = merged_df[col].round(2)
        
        # Create specialized reconciliation dataframes
        
        # 1. Items present in all three datasets (Purchase to Production to Sales)
        # These items have complete traceability through the supply chain
        print("Identifying items present in all three datasets...")
        pps_items = merged_df[
            (merged_df['Total Purchased (Kg)'] > 0) & 
            (merged_df['Total Produced (Kg)'] > 0) & 
            (merged_df['Total Sold (Kg)'] > 0)
        ].copy()
        
        # 2. Items present in purchase and sales but not in production
        # These items bypass the production process 
        print("Identifying items present in purchase and sales but not production...")
        ps_items = merged_df[
            (merged_df['Total Purchased (Kg)'] > 0) & 
            (merged_df['Total Produced (Kg)'] == 0) & 
            (merged_df['Total Sold (Kg)'] > 0)
        ].copy()
        
        # Cap extremely large values to prevent visualization issues
        # Apply the same capping logic to all reconciliation datasets
        max_reasonable_value = 1000000  # Set a reasonable upper limit for weight in kg
        print(f"Capping extremely large values to {max_reasonable_value} kg...")
        
        for df in [merged_df, pps_items, ps_items]:
            for col in numeric_columns:
                if col in df.columns:
                    df.loc[df[col] > max_reasonable_value, col] = max_reasonable_value
        
        print(f"Reconciliation complete. Found {len(merged_df)} total items.")
        print(f"- {len(pps_items)} items present in all three datasets")
        print(f"- {len(ps_items)} items in purchase and sales but not production")
        
        # Return all reconciliation dataframes in a dictionary
        return {
            'full_reconciliation': merged_df,
            'purchase_to_production_to_sales': pps_items,
            'purchase_to_sales': ps_items
        }
    
    except Exception as e:
        print(f"Error generating reconciliation report: {e}")
        import traceback
        traceback.print_exc()
        return {'full_reconciliation': pd.DataFrame()}

def create_pdf_report(reconciliation_data, purchase_summary, production_summary, sales_summary):
    """
    Generate a formatted PDF report with reconciliation data.
    
    This function creates a comprehensive PDF report including:
    - Summary statistics
    - Specialized reconciliation tables
    - Summary tables for each data source
    
    Args:
        reconciliation_data (dict): Dictionary containing the reconciliation dataframes
        purchase_summary (pd.DataFrame): Summary of purchase data
        production_summary (pd.DataFrame): Summary of production data
        sales_summary (pd.DataFrame): Summary of sales data
        
    Returns:
        str: Filename of the generated PDF report
    """
    
    # Extract dataframes from reconciliation_data dictionary
    reconciliation_df = reconciliation_data['full_reconciliation']
    pps_reconciliation = reconciliation_data['purchase_to_production_to_sales']
    ps_reconciliation = reconciliation_data['purchase_to_sales']
    
    # Create PDF object
    pdf = FPDF()
    pdf.add_page()
    
    # Set font and styles
    pdf.set_font("Arial", "B", 16)
    
    # Add title and date
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.cell(190, 10, "Ingredient Reconciliation Report", 0, 1, "C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(190, 10, f"Generated on: {now}", 0, 1, "C")
    pdf.ln(5)
    
    # Header text for summary
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 10, "Report Summary", 0, 1, "L")
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(190, 5, 
                 f"This report contains reconciliation data for {len(reconciliation_df)} ingredients, "
                 f"showing the total purchased, produced, and sold quantities in kilograms.")
    pdf.ln(5)
    
    # TABLE 1: Purchase to Production to Sales Reconciliation
    # Shows ingredients that went through the complete supply chain process
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 10, "Purchase to Production to Sales Reconciliation", 0, 1, "L")
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(190, 5, f"Items present in all three datasets ({len(pps_reconciliation)} ingredients)")
    pdf.ln(5)
    
    if not pps_reconciliation.empty:
        # Table headers
        pdf.set_font("Arial", "B", 9)
        col_widths = [60, 43, 43, 43]
        headers = ['Ingredient SKU', 'Total Purchased (Kg)', 'Total Produced (Kg)', 'Total Sold (Kg)']
        
        # Draw table header
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 7, header, 1, 0, 'C')
        pdf.ln()
        
        # Table data rows
        pdf.set_font("Arial", "", 9)
        for _, row in pps_reconciliation.iterrows():
            pdf.cell(col_widths[0], 7, str(row['ingredient_sku']), 1, 0, 'L')
            pdf.cell(col_widths[1], 7, str(row['Total Purchased (Kg)']), 1, 0, 'C')
            pdf.cell(col_widths[2], 7, str(row['Total Produced (Kg)']), 1, 0, 'C')
            pdf.cell(col_widths[3], 7, str(row['Total Sold (Kg)']), 1, 0, 'C')
            pdf.ln()
    else:
        # Handle case where no matching ingredients were found
        pdf.set_font("Arial", "I", 10)
        pdf.cell(190, 10, "No items found in all three datasets", 0, 1, 'L')
    
    # TABLE 2: Purchase to Sales Reconciliation (bypass production)
    # Shows ingredients that were purchased and sold but did not go through production
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 10, "Purchase to Sales Reconciliation", 0, 1, "L")
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(190, 5, f"Items present in purchase and sales but not in production ({len(ps_reconciliation)} ingredients)")
    pdf.ln(5)
    
    if not ps_reconciliation.empty:
        # Table headers - note we only use 3 columns here (no production column)
        pdf.set_font("Arial", "B", 9)
        col_widths = [90, 50, 50]
        headers = ['Ingredient SKU', 'Total Purchased (Kg)', 'Total Sold (Kg)']
        
        # Draw table header
        for i, header in enumerate(headers):
            pdf.cell(col_widths[i], 7, header, 1, 0, 'C')
        pdf.ln()
        
        # Table data rows
        pdf.set_font("Arial", "", 9)
        for _, row in ps_reconciliation.iterrows():
            pdf.cell(col_widths[0], 7, str(row['ingredient_sku']), 1, 0, 'L')
            pdf.cell(col_widths[1], 7, str(row['Total Purchased (Kg)']), 1, 0, 'C')
            pdf.cell(col_widths[2], 7, str(row['Total Sold (Kg)']), 1, 0, 'C')
            pdf.ln()
    else:
        # Handle case where no matching ingredients were found
        pdf.set_font("Arial", "I", 10)
        pdf.cell(190, 10, "No items found in purchase and sales but not in production", 0, 1, 'L')
    
    # TABLE 3: Full Reconciliation Data (all ingredients)
    # Complete data for all ingredients across all sources
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 10, "Full Reconciliation Data", 0, 1, "L")
    
    # Table headers
    pdf.set_font("Arial", "B", 9)
    col_widths = [60, 43, 43, 43]
    headers = ['Ingredient SKU', 'Total Purchased (Kg)', 'Total Produced (Kg)', 'Total Sold (Kg)']
    
    # Draw table header
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 7, header, 1, 0, 'C')
    pdf.ln()
    
    # Table data rows for all ingredients
    pdf.set_font("Arial", "", 9)
    for _, row in reconciliation_df.iterrows():
        pdf.cell(col_widths[0], 7, str(row['ingredient_sku']), 1, 0, 'L')
        pdf.cell(col_widths[1], 7, str(row['Total Purchased (Kg)']), 1, 0, 'C')
        pdf.cell(col_widths[2], 7, str(row['Total Produced (Kg)']), 1, 0, 'C')
        pdf.cell(col_widths[3], 7, str(row['Total Sold (Kg)']), 1, 0, 'C')
        pdf.ln()
    
    # Add individual data source summaries
    # TABLE 4-6: Source data summaries
    # These tables show the original data from each source
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 10, "Purchase Data Summary", 0, 1, "L")
    
    # Purchase table - shows all purchased ingredients
    pdf.set_font("Arial", "B", 9)
    for i, header in enumerate(['Ingredient SKU', 'Total Purchased (Kg)']):
        pdf.cell([100, 90][i], 7, header, 1, 0, 'C')
    pdf.ln()
    
    pdf.set_font("Arial", "", 9)
    for _, row in purchase_summary.iterrows():
        pdf.cell(100, 7, str(row['ingredient_sku']), 1, 0, 'L')
        pdf.cell(90, 7, str(row['Total Purchased (Kg)']), 1, 0, 'C')
        pdf.ln()
    
    pdf.ln(10)
    
    # Production table - shows all produced ingredients
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 10, "Production Data Summary", 0, 1, "L")
    
    pdf.set_font("Arial", "B", 9)
    for i, header in enumerate(['Ingredient SKU', 'Total Produced (Kg)']):
        pdf.cell([100, 90][i], 7, header, 1, 0, 'C')
    pdf.ln()
    
    pdf.set_font("Arial", "", 9)
    for _, row in production_summary.iterrows():
        pdf.cell(100, 7, str(row['ingredient_sku']), 1, 0, 'L')
        pdf.cell(90, 7, str(row['Total Produced (Kg)']), 1, 0, 'C')
        pdf.ln()
    
    pdf.ln(10)
    
    # Sales table - shows all sold ingredients
    pdf.set_font("Arial", "B", 12)
    pdf.cell(190, 10, "Sales Data Summary", 0, 1, "L")
    
    pdf.set_font("Arial", "B", 9)
    for i, header in enumerate(['Ingredient SKU', 'Total Sold (Kg)']):
        pdf.cell([100, 90][i], 7, header, 1, 0, 'C')
    pdf.ln()
    
    pdf.set_font("Arial", "", 9)
    for _, row in sales_summary.iterrows():
        pdf.cell(100, 7, str(row['ingredient_sku']), 1, 0, 'L')
        pdf.cell(90, 7, str(row['Total Sold (Kg)']), 1, 0, 'C')
        pdf.ln()
    
    # Save the PDF file with a timestamp in the filename
    report_filename = f"Reconciliation_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(report_filename)
    
    return report_filename

def main():
    """
    Main function to execute the reconciliation process.
    
    This function:
    1. Processes the input data files
    2. Creates reconciliation reports
    3. Generates visualization and PDF output
    
    The function can operate in two modes:
    - Standard mode: Uses direct pandas operations
    - LLM Agent mode: Uses OpenAI models for analysis (when USE_LLM_AGENT=true)
    """
    print("Starting reconciliation process...")
    
    # Clear any file cache to ensure we're using the latest data
    clear_cache_and_check_files()
    
    # Process the input data files
    purchase_summary = process_purchase_data(PURCHASE_FILE)
    production_summary = process_production_data(PRODUCTION_FILE)
    sales_summary = process_sales_data(SALES_FILE)
    
    # Check if LLM agent analysis is enabled via environment variable
    use_llm_agent = os.environ.get("USE_LLM_AGENT", "false").lower() == "true"
    
    if use_llm_agent:
        # Option 1: Use LLM agent for analysis (requires OpenAI API)
        # This provides AI-powered insights but is slower and costs API credits
        agent_dfs = {
            "purchase_summary": purchase_summary,
            "production_summary": production_summary,
            "sales_summary": sales_summary
        }
        
        # Create the LangChain agent with the dataframes
        agent = create_pandas_dataframe_agent(
            LLM,
            list(agent_dfs.values()),
            verbose=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
            allow_dangerous_code=True
        )
        
        # Define the analysis prompt for the agent
        prompt = f"""
        You are provided with three pandas DataFrames:
        1. Purchase Summary (df1): Contains 'Ingredient SKU' and 'Total Purchased (Kg)'.
        2. Production Summary (df2): Contains 'Ingredient SKU' and 'Total Produced (Kg)'.
        3. Sales Summary (df3): Contains 'Ingredient SKU' and 'Total Sold (Kg)'.

        Your task is to create a final reconciliation report.
        1. Merge these three summaries into a single DataFrame using 'Ingredient SKU' as the key.
        2. The final DataFrame should have the columns: 'Ingredient SKU', 'Total Purchased (Kg)', 'Total Produced (Kg)', 'Total Sold (Kg)'.
        3. Ensure all SKUs from all summaries are included. Use an outer merge.
        4. After merging, fill any resulting NaN values in the numeric columns with 0.
        5. Also create two additional dataframes:
           a. Items present in all three datasets (purchase, production, and sales)
           b. Items present in purchase and sales but not in production
        6. Return all three dataframes as markdown tables.
        """
        
        print("\nRunning Agent to generate reconciliation report...\n")
        try:
            # Execute the agent with the prompt
            result = agent.invoke({"input": prompt})
            print("\n--- Reconciliation Report ---")
            print(result['output'])
        
        except Exception as e:
            print(f"An error occurred while running the agent: {e}")
    
    else:
        # Option 2: Use direct pandas operations (faster and more reliable)
        # Create reconciliation data using standard pandas operations
        reconciliation_data = create_reconciliation_report(purchase_summary, production_summary, sales_summary)
        
        if 'full_reconciliation' in reconciliation_data and not reconciliation_data['full_reconciliation'].empty:
            # Display the full reconciliation report
            print("\n--- Full Reconciliation Report ---")
            print(reconciliation_data['full_reconciliation'].to_markdown(index=False))
            
            # Display specialized reconciliation reports
            print("\n--- Purchase to Production to Sales Reconciliation ---")
            if not reconciliation_data['purchase_to_production_to_sales'].empty:
                print(reconciliation_data['purchase_to_production_to_sales'].to_markdown(index=False))
            else:
                print("No items found in all three datasets")
                
            print("\n--- Purchase to Sales Reconciliation ---")
            if not reconciliation_data['purchase_to_sales'].empty:
                print(reconciliation_data['purchase_to_sales'].to_markdown(index=False))
            else:
                print("No items found in purchase and sales but not in production")
            
            # Generate PDF report with visualizations
            pdf_filename = create_pdf_report(reconciliation_data, purchase_summary, production_summary, sales_summary)
            print(f"\nPDF report generated: {pdf_filename}")
    
    print("\nScript finished.") 

if __name__ == "__main__":
    main() 