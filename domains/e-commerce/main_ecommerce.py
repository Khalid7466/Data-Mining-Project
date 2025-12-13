import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Setup path to import from 'modules' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from modules.tool_apriori import run_apriori

def encode_units(x):
    """
    Encodes quantity data into a binary format.
    Returns 1 if quantity > 0, otherwise returns 0.
    """
    if x > 0:
        return 1
    else:
        return 0

def main():
    """
    Main function to execute Market Basket Analysis on Online Retail data.
    Performs a comparative analysis between France and Germany.
    """

    # 1. Data Loading
    file_path = r'G:\The Last Year\Data Mining\online_retail.csv'
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        # Load dataset with specific encoding and separator
        # 'low_memory=False' is added to suppress DtypeWarnings for mixed types
        df = pd.read_csv(file_path, sep=';', encoding='ISO-8859-1', low_memory=False)
        print("Data loaded successfully.")
        
        # 1. Strip leading/trailing whitespace from all column names
        df.columns = df.columns.str.strip()

        # 2. Force rename the first column to 'BillNo' 
        df.rename(columns={df.columns[0]: 'BillNo'}, inplace=True)

        # 3. Ensure 'Itemname' is identified correctly
        # If 'Itemname' is not found, attempt to find it via case-insensitive matching
        if 'Itemname' not in df.columns:
            possible_cols = [c for c in df.columns if 'itemname' in c.lower()]
            if possible_cols:
                df.rename(columns={possible_cols[0]: 'Itemname'}, inplace=True)
        
        print(f"Verified Columns: {df.columns.tolist()}")

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. General Preprocessing
    
    # Clean item names by removing leading/trailing whitespace
    df['Itemname'] = df['Itemname'].str.strip()
    
    # Remove rows where 'BillNo' is missing
    df.dropna(axis=0, subset=['BillNo'], inplace=True)
    
    # Ensure 'BillNo' is treated as a string type
    df['BillNo'] = df['BillNo'].astype('str')
    
    # Filter out cancelled transactions (invoices containing 'C')
    df = df[~df['BillNo'].str.contains('C')]

    # Define list of countries for comparative analysis
    target_countries = ['France', 'Germany']

    # 3. Analysis Loop
    for country in target_countries:

        # Filter dataset for the specific country
        subset_df = df[df['Country'] == country]

        if subset_df.empty:
            print(f"Warning: No data found for {country}, skipping...")
            continue

        # Transform data: Pivot to 'Basket' format (Transactions x Items)
        basket = (subset_df.groupby(['BillNo', 'Itemname'])['Quantity']
                .sum().unstack().reset_index().fillna(0)
                .set_index('BillNo'))

        # Binary Encoding (Convert quantities to 0 or 1)
        basket_encoded = basket.map(encode_units)

        # Remove 'POSTAGE' column as it represents shipping fees, not a product
        if 'POSTAGE' in basket_encoded.columns:
            basket_encoded.drop('POSTAGE', inplace=True, axis=1)

        print(f"Matrix Dimensions: {basket_encoded.shape} (Transactions x Items)")

        # Calculate dataset statistics for support threshold
        num_transactions = basket_encoded.shape[0]
        # Using 5% support for the comparison to ensure results for both countries
        min_transactions = int(num_transactions * 0.05) 

        print(f"Total Transactions: {num_transactions}")
        print(f"Minimum Transactions Required (5% Support): {min_transactions}")

        # 4. Run Apriori Algorithm
        frequent_items, rules = run_apriori(basket_encoded, min_support=0.05, metric="lift", min_threshold=1)

        if not frequent_items.empty:
            print(f"\nTop 5 Best Selling Items in {country}:")
            print(frequent_items.sort_values('support', ascending=False).head())

        # 5. Output Results
        if not rules.empty:
            print(f"\nTop 5 Association Rules for {country}:")
            print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
            
            # Construct output path dynamically
            output_path = f'rules_results_{country}.csv'
            try:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                rules.to_csv(output_path, index=False)
                print(f"Success: Results saved to {output_path}")
            except OSError:
                # Fallback to local directory if path creation fails
                rules.to_csv(f'rules_results_{country}.csv', index=False)
                print(f"Success: Results saved locally as rules_results_{country}.csv")
        else:
            print(f"No association rules found for {country} with current parameters.")
    
        # 6. Visualization: Scatter plot of Support vs Confidence
        if not rules.empty:
            plt.figure(figsize=(10, 6))
            
            sns.scatterplot(
                data=rules, 
                x="support", 
                y="confidence", 
                size="lift", 
                hue="lift", 
                palette="viridis", 
                sizes=(20, 200)
            )
            
            plt.title(f"Association Rules for {country} (Support vs Confidence)")
            plt.xlabel("Support")
            plt.ylabel("Confidence")
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            
            plt.savefig(f'plot_{country}.png', bbox_inches='tight')
            plt.show()
            print(f"Plot saved as plot_{country}.png")


if __name__ == "__main__":
    main()

