from typing import List, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
from fca.api_models import Context


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


def show_concepts(concepts, min_support: float, min_size_intent: int, num_of_rows: int) -> None:
    """
    Filters and displays the given concepts based on minimum support and
    minimum size of intent.

    The function iterates over a list of concepts, represented as objects
    convertible to tuples. Each concept consists of a pair where the first
    element (intent) contains a set of items, and the second element
    (extent) contains associated related items. The function filters
    concepts based on a minimum size for the intent and a minimum support
    threshold, which is computed based on the size of the extent relative
    to the total number of rows.

    Filtered concepts are displayed with their intent and calculated
    support value.

    :param concepts: List of concept objects, each convertible to a tuple
        where the first element is the extent, and the second is the intent.
    :param min_support: Float representing the minimum support threshold.
        A concept's support is calculated as the ratio of its extent size
        to the total number of rows.
    :param min_size_intent: Integer specifying the minimum size of the
        intent of a concept to be considered valid.
    :param num_of_rows: Integer representing the total number of rows
        in the dataset, used for support calculation.
    :return: None
    """
    ind = 0
    for i in concepts:
        i = i.to_tuple()
        if len(i[1]) < min_size_intent:
            continue
        supp = support(len(i[0]), num_of_rows)
        if supp < min_support:
            continue
        ind += 1
        print(f"{ind}: {i[1]} - Support : {supp}")


def show_rules(rules) -> None:
    """
    Iterates over a list of rules and prints their details including the base items,
    additional items, support, confidence, and lift values. Each rule defines an
    association between sets of items, and these details provide insights into the
    strength and importance of these associations.

    :param rules: A list of association rules where each rule contains statistical
        details such as support, confidence, and lift, along with the items involved.
    :type rules: list
    :return: None
    """
    ind = 1
    for j in rules:
        stat = j.ordered_statistics[0]
        if stat.items_base == frozenset():
            continue
        base_items = list(stat.items_base)
        add_items = list(stat.items_add)

        base_str = ", ".join(base_items) if len(base_items) > 1 else base_items[0]
        add_str = ", ".join(add_items) if len(add_items) > 1 else add_items[0]

        print(f"{ind}: {base_str} -> {add_str}")
        print(f"Support: {j.support:.4f} - Confidence : {stat.confidence:.4f} - Lift: {stat.lift:.4f}")
        ind += 1


def get_rules_for_rounds(min_support, confidence):
    for i in range(1, 10):
        print(f"================= Round # {i} =================")
        df = pd.read_csv(f"../data/essround{i}.csv")
        objects, attributes, relation = create_context_from_dataframe(df)
        context = Context(objects, attributes, relation)
        rules = list(context.get_association_rules(min_support=min_support, min_confidence=confidence))
        show_rules(rules)


def get_repeated_rules(min_support, confidence):
    rule_occurrences = defaultdict(set)

    for i in range(1, 10):
        df = pd.read_csv(f"../data/essround{i}.csv")
        objects, attributes, relation = create_context_from_dataframe(df)
        context = Context(objects, attributes, relation)
        rules = list(context.get_association_rules(min_support=min_support, min_confidence=confidence))

        for j in rules:
            stat = j.ordered_statistics[0]
            if stat.items_base == frozenset():
                continue

            base_items = tuple(sorted(stat.items_base))
            add_items = tuple(sorted(stat.items_add))
            rule_key = (base_items, add_items)
            rule_occurrences[rule_key].add(i)

    print("\n==== Rules that appear in 2 or more rounds ====\n")
    ind = 1
    for (base, add), rounds in rule_occurrences.items():
        if len(rounds) >= 2:
            base_str = ", ".join(base)
            add_str = ", ".join(add)
            rounds_str = ", ".join(map(str, sorted(rounds)))
            print(f"{ind}: {base_str} -> {add_str} [Rounds: {rounds_str}]")
            ind += 1
