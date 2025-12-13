import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def run_apriori(basket_encoded, min_support=0.05, metric="lift", min_threshold=1):
    """
    Executes the Apriori algorithm to identify frequent itemsets and derive association rules.

    Parameters:
    ----------
    basket_encoded : pd.DataFrame
        A one-hot encoded matrix (0s and 1s) where rows are transactions and columns are items.
    min_support : float
        The minimum support threshold required for an itemset to be considered frequent.
    metric : str, optional
        The metric used to evaluate the rules (default is 'lift').
    min_threshold : float, optional
        The minimum threshold for the specified metric (default is 1).

    Returns:
    -------
    frequent_items : pd.DataFrame
        A DataFrame containing the itemsets that meet the minimum support criteria.
    rules : pd.DataFrame
        A DataFrame containing the generated association rules, sorted by lift and confidence.
    """

    # Step 1: Generate Frequent Itemsets
    # 'use_colnames=True' ensures that item names are returned instead of column indices
    frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)

    # Safety Check: Return empty results if no itemsets meet the support threshold
    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()

    # Step 2: Generate Association Rules
    # Extracts rules based on the defined metric (e.g., lift) and threshold
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=min_threshold)

    # Step 3: Sort Results
    # Prioritize rules with the highest Lift and Confidence for better business insights
    rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])

    return frequent_itemsets, rules