from typing import List, Optional, Tuple, Dict, Set
import pandas as pd
import numpy as np
from fca.api_models import Context
from collections import defaultdict


def booleanize_numeric_columns(df: pd.DataFrame, num_bins: int = 3) -> pd.DataFrame:
    """
    Transforms numeric columns in a DataFrame into boolean indicator columns based on their
    binned quantile ranges. Each numeric column is divided into `num_bins` bins representing
    quantile ranges, and for each bin, a new column is created with boolean values indicating
    whether the respective value belongs to that bin.

    :param df: The input DataFrame.
    :type df: pd.DataFrame
    :param num_bins: The number of bins to divide the numeric columns into. Defaults to 3.
    :type num_bins: int, optional
    :return: A new DataFrame with boolean indicator columns for each numeric column and bin.
    :rtype: pd.DataFrame
    """
    result = pd.DataFrame(index=df.index)
    numeric_columns = df.select_dtypes(include='number').columns

    for column in numeric_columns:
        quantiles = [df[column].quantile(i / num_bins) for i in range(1, num_bins)]
        unique_quantiles = sorted(set(quantiles))
        bin_labels = [f"bin_{i + 1}" for i in range(len(unique_quantiles) + 1)]

        bin_edges = [-np.inf] + unique_quantiles + [np.inf]
        binned_data = pd.cut(df[column], bins=bin_edges, labels=bin_labels)

        for bin_label in bin_labels:
            result[f"{column}_{bin_label}"] = (binned_data == bin_label).astype(int)

    return result


def process_dataset_for_fca(df: pd.DataFrame,
                            num_bins: int = 3,
                            drop_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Processes a dataset for Formal Concept Analysis (FCA) by preparing it in a
    booleanized format. The dataset is processed by separating numeric and
    categorical columns, then booleanizing each column type appropriately.
    Numeric columns are transformed using bins, and categorical columns are
    converted to boolean representations. Columns specified in the excluded
    list will be dropped from the dataset.

    :param df: The source dataset to process.
    :type df: pd.DataFrame

    :param num_bins: The number of bins to use for booleanization of numeric
        columns, default is 3.
    :type num_bins: int, optional

    :param drop_columns: A list of columns to drop from the dataset, default is
        None (no columns will be dropped).
    :type drop_columns: Optional[List[str]]

    :return: A DataFrame where all columns are transformed to boolean types
        (binary representation) suitable for FCA.
    :rtype: pd.DataFrame
    """
    if df.empty:
        return df
    if drop_columns:
        df.drop(columns=drop_columns, errors='ignore', inplace=True)

    numeric_df = df.select_dtypes(include='number')
    categorical_df = df.select_dtypes(include=['object', 'category', 'bool'])

    boolean_numeric = booleanize_numeric_columns(numeric_df, num_bins=num_bins)
    boolean_categorical = pd.get_dummies(categorical_df,
                                         prefix_sep='=',
                                         dtype=bool)

    return pd.concat([boolean_numeric.astype(bool),
                      boolean_categorical.astype(bool)], axis=1)


def create_context_from_dataframe(df):
    """
    Creates context data from the given dataframe. The resulting context consists of
    three components: objects, attributes, and relations. Objects are generated
    from the index of the dataframe, attributes are derived from the dataframe
    columns, and relations represent the values of the dataframe in list form.

    :param df: The input dataframe from which context components are derived
               (objects, attributes, and relations).
    :type df: pandas.DataFrame

    :return: A tuple containing three elements:
             - A list of objects derived from the dataframe index.
             - A list of attributes derived from the dataframe's columns.
             - A list of relations derived from the dataframe's values.
    :rtype: tuple[list, list, list[list]]
    """
    objects = df.index.tolist()
    attributes = df.columns.tolist()
    relation = df.values.tolist()

    return objects, attributes, relation


def support(extent: int, total_objects: int) -> float:
    """
    Computes the support value given an extent and the total number of objects.

    Support serves as a measure for the proportion of a specific condition or occurrence
    relative to a total number of objects. The computed value is obtained by dividing
    the extent by the total number of objects.

    In cases where the total number of objects is zero, the function will return a
    support value of 0.0 to avoid division by zero error.

    :param extent: The extent or occurrence count of a specific condition
    :param total_objects: The total number of objects. Default is 25000
    :return: The support value as a float, calculated as `extent / total_objects`
    """
    if total_objects == 0:
        return 0.0
    return extent / total_objects


def get_and_show_concepts(df, min_support: float, min_len_of_concept: int):
    objects, attributes, relation = create_context_from_dataframe(df)
    context = Context(objects, attributes, relation)
    concepts = context.get_concepts()

    num_rows = len(df)
    concept_list = []

    for i in concepts:
        i = i.to_tuple()
        if len(i[1]) < min_len_of_concept:
            continue
        supp = support(len(i[0]), num_rows)
        if supp < min_support:
            continue
        concept_list.append((i[1], supp))

    concept_list.sort(key=lambda x: x[1], reverse=True)

    for idx, (intent, supp) in enumerate(concept_list, start=1):
        print(f"{idx}: {intent} - Support: {supp}")


def get_and_show_rules(df, min_support: float, min_confidence: float):
    objects, attributes, relation = create_context_from_dataframe(df)
    context = Context(objects, attributes, relation)
    rules = list(context.get_association_rules(min_support=min_support, min_confidence=min_confidence))
    show_rules(rules)


def format_rule(rule, index: int) -> str:
    """
    Format a single rule with support, confidence, and lift.
    """
    stat = rule.ordered_statistics[0]
    base_items = list(stat.items_base)
    add_items = list(stat.items_add)

    base_str = ", ".join(base_items)
    add_str = ", ".join(add_items)

    return (f"{index}: {base_str} -> {add_str}\n"
            f"Support: {rule.support:.4f} - Confidence: {stat.confidence:.4f} - Lift: {stat.lift:.4f}")


def show_rules(rules) -> None:
    valid_rules = [rule for rule in rules if rule.ordered_statistics[0].items_base]

    valid_rules.sort(key=lambda r: r.support, reverse=True)

    for index, rule in enumerate(valid_rules, start=1):
        print(format_rule(rule, index))


def extract_valid_rules(rules) -> List[Tuple[Tuple[str], Tuple[str]]]:
    """
    Extract and return only rules with a non-empty base, sorted by support (descending).
    """
    valid_rules = []

    for rule in rules:
        stat = rule.ordered_statistics[0]
        if stat.items_base:
            base_items = tuple(sorted(stat.items_base))
            add_items = tuple(sorted(stat.items_add))
            support = rule.support
            valid_rules.append((support, base_items, add_items))

    valid_rules.sort(reverse=True, key=lambda x: x[0])

    return [(base, add) for _, base, add in valid_rules]


def get_rules_for_rounds(min_support: float, min_confidence: float, show_repeated: bool = True) -> None:
    """
    Analyze association rules for multiple rounds and optionally show repeated rules.

    :param min_support: Minimum support threshold
    :param min_confidence: Minimum confidence threshold
    :param show_per_round: Whether to display rules per round
    :param show_repeated: Whether to display rules that appear in multiple rounds
    """
    rule_occurrences: Dict[Tuple[Tuple[str], Tuple[str]], Set[int]] = defaultdict(set)

    for round_num in range(1, 10):
        df = pd.read_csv(f"../data/essround{round_num}.csv")
        objects, attributes, relation = create_context_from_dataframe(df)
        context = Context(objects, attributes, relation)

        rules = list(context.get_association_rules(min_support=min_support, min_confidence=min_confidence))

        print(f"\n================= Round #{round_num} =================")
        show_rules(rules)

        for base, add in extract_valid_rules(rules):
            rule_occurrences[(base, add)].add(round_num)

    if show_repeated:
        print("\n==== Rules that appear in 2 or more rounds ====\n")
        index = 1
        for (base, add), rounds in rule_occurrences.items():
            if len(rounds) >= 2:
                base_str = ", ".join(base)
                add_str = ", ".join(add)
                rounds_str = ", ".join(map(str, sorted(rounds)))
                print(f"{index}: {base_str} -> {add_str} [Rounds: {rounds_str}]")
                index += 1


def print_top_conf1_rules_by_support(top_n: int = 10) -> None:
    """
    For each round, find and print the top N rules with confidence = 1.0,
    trying different support thresholds (from 0.7 down to 0.1).

    :param top_n: Number of top rules to print per round (default: 10)
    """
    support_levels = [x / 100 for x in range(70, 9, -5)]

    for round_num in range(1, 10):
        print(f"\n================= Round #{round_num} =================")
        df = pd.read_csv(f"../data/essround{round_num}.csv")
        objects, attributes, relation = create_context_from_dataframe(df)
        context = Context(objects, attributes, relation)

        found = False
        for sup in support_levels:
            rules = list(context.get_association_rules(min_support=sup, min_confidence=1.0))
            valid_rules = [r for r in rules if r.ordered_statistics[0].items_base]

            if len(valid_rules) >= top_n:
                sorted_rules = sorted(valid_rules, key=lambda r: r.support, reverse=True)[:top_n]
                for idx, rule in enumerate(sorted_rules, start=1):
                    print(format_rule(rule, idx))
                found = True
                break

        if not found:
            print(f"⚠️ Not enough rules with confidence = 1 found at any support level to print top {top_n}.")


def get_support(df: pd.DataFrame, itemset: Set[str]) -> float:
    """
    Calculate the support of an itemset in the dataframe.

    :param df: The input DataFrame with boolean or binary (0/1 or True/False) attributes.
    :param itemset: A set of attributes/items to compute support for.
    :return: Support value (fraction of rows that contain all items)
    """
    if not itemset:
        return 0.0
    condition = df[list(itemset)].all(axis=1)
    return condition.mean()


def get_confidence(df: pd.DataFrame, base: Set[str], add: Set[str]) -> float:
    """
    Calculate the confidence of a rule: base → add

    :param df: The input DataFrame with binary (1/0 or True/False) data.
    :param base: Set of base (antecedent) items.
    :param add: Set of added (consequent) items.
    :return: Confidence value (P(add | base))
    """
    if not base or not add:
        return 0.0
    support_base = get_support(df, base)
    support_combined = get_support(df, base.union(add))
    if support_base == 0:
        return 0.0
    return support_combined / support_base


def get_lift(df, base, add):
    """
    Calculate the lift of a rule: base => add

    Args:
        df (pd.DataFrame): Booleanized DataFrame.
        base (set): Antecedent itemset (if X then ...).
        add (set): Consequent itemset (... then Y).

    Returns:
        float: Lift value of the rule.
    """
    support_base = get_support(df, base)
    support_add = get_support(df, add)
    support_both = get_support(df, base | add)

    if support_base == 0 or support_add == 0:
        return 0.0

    return support_both / (support_base * support_add)


def evaluate_itemset_across_rounds(base: Set[str], add: Set[str]) -> None:
    """
    For each round, compute and print:
      - Support of the combined itemset (base ∪ add)
      - Confidence of the rule: base → add

    :param base: Set of base (antecedent) attributes
    :param add: Set of add (consequent) attributes
    :param data_path: Path to folder containing essround{i}.csv files
    """
    if not base or not add:
        print("⚠️ Both base and add must be non-empty sets.")
        return

    full_itemset = base.union(add)
    print(f" {base} → {add} ")

    for round_num in range(1, 10):
        df = pd.read_csv(f"../data/essround{round_num}.csv")

        if not full_itemset.issubset(df.columns):
            print(f"Round {round_num}: ⚠️ Missing attributes in itemset: {full_itemset - set(df.columns)}")
            continue

        support = get_support(df, full_itemset)
        confidence = get_confidence(df, base, add)
        lift = get_lift(df, base, add)

        print(f"\n================= Round #{round_num} =================")
        print(f"Support: {support:.4f} - Confidence: {confidence:.4f} - Lift: {lift:.4f}")


def analyze_rule_evolution(round: int, num_of_rules: int = 10, min_support: float = 0.5,
                           min_confidence: float = 1) -> None:
    """
    Analyze how the top N rules from a given round evolve in the subsequent rounds.
    For each rule, print how support, confidence, and lift change over time.

    :param round: The round number from which to extract rules
    :param num_of_rules: Number of rules to analyze
    :param min_support: Minimum support threshold for rule generation
    :param min_confidence: Minimum confidence threshold for rule generation
    """

    df = pd.read_csv(f"../data/essround{round}.csv")
    objects, attributes, relation = create_context_from_dataframe(df)
    context = Context(objects, attributes, relation)

    print(f"\n📊 Analyzing top {num_of_rules} rules from Round {round}")
    rules = list(context.get_association_rules(min_support=min_support, min_confidence=min_confidence))
    filtered_rules = list(extract_valid_rules(rules))

    if not filtered_rules:
        print("⚠️ No valid rules found.")
        return

    for idx, (base, add) in enumerate(filtered_rules[:num_of_rules], 1):
        base_set = set(base)
        add_set = set(add)
        print(f"\n🔍 Rule #{idx}:")
        evaluate_itemset_across_rounds(base_set, add_set)
