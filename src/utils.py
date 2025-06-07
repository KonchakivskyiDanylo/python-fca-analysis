from typing import List, Optional, Tuple, Dict, Set
import pandas as pd
from fca.api_models import Context
from collections import defaultdict
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def load_attribute_mapping() -> dict:
    """
    Attribute mapping for better reading
    """
    df: pd.DataFrame = pd.read_csv("../data/codebook.csv")

    mapping: dict = dict(zip(df["vert"], df["question"]))
    return mapping


ATTRIBUTE_MAPPING = load_attribute_mapping()


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


def create_context_from_dataframe(df: pd.DataFrame):
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


def get_and_show_concepts(df, min_support: float, min_len_of_concept: int, use_mapping: bool = False):
    """
    Extracts and displays formal concepts from a given DataFrame using Formal Concept Analysis
    (FCA). The method identifies concepts based on their attributes and their support within
    the dataset. Users can specify a minimum support threshold and a minimum number of
    attributes to filter the identified concepts. Additionally, the function can map attribute
    names using a predefined mapping for better interpretability.

    :param df: Input dataset in the form of a pandas DataFrame from which formal concepts are
        derived.
    :type df: pandas.DataFrame

    :param min_support: Minimum support threshold for filtering concepts. Only the concepts
        with support greater than or equal to this threshold are retained.
    :type min_support: float

    :param min_len_of_concept: Minimum length of the concept intent. Only the concepts with
        intents of size greater than or equal to this value are retained.
    :type min_len_of_concept: int

    :param use_mapping: Indicates whether the attribute names in the concept intents should be
        mapped to more interpretable names using a predefined mapping. Defaults to False.
    :type use_mapping: bool

    :return: A list of tuples containing the concept intents and their corresponding support
        values. Each tuple consists of a set of attributes (intent) and a support value.
    :rtype: list[tuple[set, float]]
    """
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

    if use_mapping:
        for idx, (intent, supp) in enumerate(concept_list, start=1):
            formatted_intent = []
            for attr in intent:
                parts = attr.rsplit('_', 2)
                if len(parts) == 3:
                    short, bin_num, bin_total = parts
                    full = ATTRIBUTE_MAPPING.get(short, short)
                    formatted = f"{full}_{bin_num}_from_{bin_total}"
                else:
                    formatted = attr
                formatted_intent.append(formatted)
            print(f"{idx}: {formatted_intent} - Support: {supp:.4f}")
    else:
        for idx, (intent, supp) in enumerate(concept_list, start=1):
            print(f"{idx}: {intent} - Support: {supp}")


def get_and_show_rules(df, min_support: float, min_confidence: float, use_mapping: bool = False):
    """
    Extracts and displays association rules from a given dataframe using Formal
    Concept Analysis. The function builds a formal context based on the input
    data and computes association rules abiding by specified minimum support
    and confidence thresholds. The rules can also be visualized using mapped
    representations if specified.

    :param df: DataFrame containing the input data for creating the formal
        context.
    :type df: pandas.DataFrame
    :param min_support: Minimum support threshold for the rules. Should be a
        float value between 0 and 1.
    :type min_support: float
    :param min_confidence: Minimum confidence threshold for the rules. Should
        be a float value between 0 and 1.
    :type min_confidence: float
    :param use_mapping: A boolean flag to specify whether to use mapping for
        displaying the rules. If True, rules are displayed with mapped
        representations, otherwise they are shown in their default form.
    :type use_mapping: bool
    :return: Nothing is returned as the rules are displayed directly.
    """
    objects, attributes, relation = create_context_from_dataframe(df)
    context = Context(objects, attributes, relation)
    rules = list(context.get_association_rules(min_support=min_support, min_confidence=min_confidence))
    show_rules(rules, use_mapping=use_mapping)
    # show_rules_network(rules)


def format_item(item: str, use_mapping: bool) -> tuple[str, str | None]:
    """
    Formats a string identifier by mapping its prefix to a full attribute name and
    reconstructing it with its bin number and bin total. If the input string does not
    fit the expected format of three parts separated by underscores, it remains
    unchanged.

    :param item: The input string identifier in the format <short>_<bin_num>_<bin_total>.
    :param use_mapping: A boolean indicating whether to apply the ATTRIBUTE_MAPPING
        dictionary to expand the short form of the identifier.
    :return: The formatted string constructed from the input string or the original
        string if it does not match the expected format.
    :rtype: str
    """
    parts = item.rsplit('_', 2)
    if len(parts) == 3:
        short, bin_num, bin_total = parts
        full = ATTRIBUTE_MAPPING.get(short, short) if use_mapping else short
        bin_label = f"bin {bin_num} from {bin_total}"
        return full, bin_label
    return item, None


def format_rule(rule, index: int, use_mapping: bool = False) -> str:
    """
    Formats a rule into a string representation, including the items involved in the rule, its index,
    and its statistical metrics (support, confidence, and lift). The function processes the base
    items and added items of the rule, then formats them into a human-readable string. Optionally,
    it allows mapping the items through a custom mapping function if specified.

    :param rule: The association rule to be formatted.
    :type rule: Any
    :param index: The index of the rule to be represented, typically denoting its order.
    :type index: int
    :param use_mapping: Flag to indicate whether to apply a mapping function to the rule's items.
    :type use_mapping: bool
    :return: A string representing the formatted rule, including index, items, and metrics.
    :rtype: str
    """
    stat = rule.ordered_statistics[0]
    base_items = list(stat.items_base)
    add_items = list(stat.items_add)

    lines = [f"RULE {index}:"]

    for item in base_items:
        text, bin_label = format_item(item, use_mapping)
        if bin_label:
            lines.append(f"\"{text}\" [{bin_label}]")
        else:
            lines.append(f"\"{text}\"")

    lines.append("  ->")

    for item in add_items:
        text, bin_label = format_item(item, use_mapping)
        if bin_label:
            lines.append(f"\"{text}\" [{bin_label}]")
        else:
            lines.append(f"\"{text}\"")

    lines.append("")
    lines.append(f"[Support: {rule.support:.4f}] [Confidence: {stat.confidence:.4f}] [Lift: {stat.lift:.4f}]")
    lines.append("")

    return "\n".join(lines)


def show_rules(rules, use_mapping: bool = False) -> None:
    """
    Filters and displays association rules based on their base items and support. The function
    accepts a list of rules, filters out those without a base item, sorts the remaining rules
    by their support in descending order, and then displays them in a formatted manner.

    :param rules:
        A list of association rules to be processed. Each rule is expected to have
        an "ordered_statistics" attribute, which is a list where the first item
        contains information about the base items of the rule.
    :param use_mapping:
        A boolean indicating whether to use mapping for formatting the displayed rules.
        Defaults to False.
    :return:
        This function does not return a value. It processes and displays the
        formatted rules directly to the console.
    """
    valid_rules = [rule for rule in rules if rule.ordered_statistics[0].items_base]
    valid_rules.sort(key=lambda r: r.support, reverse=True)

    for index, rule in enumerate(valid_rules, start=1):
        print(format_rule(rule, index, use_mapping=use_mapping))


def extract_valid_rules(rules) -> List[Tuple[Tuple[str], Tuple[str]]]:
    """
    Extracts valid rules from a given list of rules, sorts them by support in
    descending order, and returns a list of tuples containing the base and added
    items of the rules.

    The input rules should be iterable, and each rule is expected to have an
    attribute called `ordered_statistics`, which is a list. The first element of
    this list should contain an `items_base` and `items_add` attribute representing
    the base and additional items for the rule. Only rules with non-empty
    `items_base` are considered valid. The valid rules are sorted by their support
    value in descending order.

    :param rules: The input rules to process. Each rule should have an
                  `ordered_statistics` attribute containing, among others, the
                  `items_base` and `items_add` details.
    :type rules: List
    :return: A list of tuples where each tuple contains two sub-tuples: the first
             sub-tuple is the base items, and the second sub-tuple is the added
             items for each valid rule. The list is sorted by support in
             descending order.
    :rtype: List[Tuple[Tuple[str], Tuple[str]]]
    """
    valid_rules = []

    for rule in rules:
        stat = rule.ordered_statistics[0]
        if stat.items_base:
            base_items = tuple(sorted(stat.items_base))
            add_items = tuple(sorted(stat.items_add))
            sup = rule.support
            valid_rules.append((sup, base_items, add_items))

    valid_rules.sort(reverse=True, key=lambda x: x[0])

    return [(base, add) for _, base, add in valid_rules]


def get_rules_for_rounds(min_support: float, min_confidence: float, show_repeated: int = 0,
                         use_mapping: bool = False) -> None:
    """
    Analyzes and outputs association rules for rounds of data, calculates occurrences
    across rounds, and optionally displays rules that appear in two or more rounds.

    :param min_support: Minimum support threshold for association rule determination.
    :type min_support: float
    :param min_confidence: Minimum confidence threshold for association rule determination.
    :type min_confidence: float
    :param show_repeated: Flag indicating whether to display rules that appear in give amount or
        more rounds. Defaults to True.
    :type show_repeated: in, optional
    :param use_mapping: Flag indicating whether to use a mapping for formatting rule items.
        Defaults to False.
    :type use_mapping: bool, optional

    :return: None
    """
    rule_occurrences: Dict[Tuple[Tuple[str], Tuple[str]], Set[int]] = defaultdict(set)

    for round_num in range(1, 10):
        df = pd.read_csv(f"../data/essround{round_num}.csv")
        # df.drop(["clsprty_1_3", "clsprty_2_3", "clsprty_3_3"], axis=1, inplace=True)
        objects, attributes, relation = create_context_from_dataframe(df)
        context = Context(objects, attributes, relation)

        rules = list(context.get_association_rules(min_support=min_support, min_confidence=min_confidence))

        print(f"\n================= Round #{round_num} =================")
        show_rules(rules, use_mapping)

        for base, add in extract_valid_rules(rules):
            rule_occurrences[(base, add)].add(round_num)

    if show_repeated:
        print(f"\n==== Rules that appear in {show_repeated} or more rounds ====\n")
        index = 1
        for (base, add), rounds in rule_occurrences.items():
            if len(rounds) >= show_repeated:
                print(f"RULE {index}:")
                for i in base:
                    text, bin_label = format_item(i, use_mapping)
                    line = f"\"{text}\""
                    if bin_label:
                        line += f" [{bin_label}]"
                    print(line)
                print("  ->")
                for i in add:
                    text, bin_label = format_item(i, use_mapping)
                    line = f"\"{text}\""
                    if bin_label:
                        line += f" [{bin_label}]"
                    print(line)
                rounds_str = ", ".join(map(str, sorted(rounds)))
                print(f"[Rounds: {rounds_str}]\n")
                index += 1


def print_top_rules_by_support(top_n: int = 10, use_mapping: bool = False) -> None:
    """
    Prints the top N association rules sorted by support with a confidence value of 1.0 for each dataset round.
    The function processes data for rounds from 1 to 9, aiming to extract and format association rules
    from each round's dataset. If the desired number of rules is not met at any support level, a warning is
    printed for that round.

    :param top_n: The number of top association rules to print for each round.
    :param use_mapping: If True, applies mapping for the rule formatting.
    :return: None
    """
    support_levels = [x / 100 for x in range(70, 9, -5)]

    for round_num in range(1, 10):
        print(f"\n================= Round #{round_num} =================")
        df = pd.read_csv(f"../data/essround{round_num}.csv")
        # df.drop(["clsprty_1_3", "clsprty_2_3", "clsprty_3_3"], axis=1, inplace=True)
        objects, attributes, relation = create_context_from_dataframe(df)
        context = Context(objects, attributes, relation)

        found = False
        for sup in support_levels:
            rules = list(context.get_association_rules(min_support=sup, min_confidence=1.0))
            valid_rules = [r for r in rules if r.ordered_statistics[0].items_base]

            if len(valid_rules) >= top_n:
                sorted_rules = sorted(valid_rules, key=lambda r: r.support, reverse=True)[:top_n]
                for idx, rule in enumerate(sorted_rules, start=1):
                    print(format_rule(rule, idx, use_mapping=use_mapping))
                found = True
                break

        if not found:
            print(f"⚠️ Not enough rules with confidence = 1 found at any support level to print top {top_n}.")


def get_support(df: pd.DataFrame, itemset: Set[str]) -> float:
    """
    Calculate the support of a given itemset in a pandas DataFrame.

    The function determines the proportion of rows in the DataFrame
    where all the items in the given itemset are present. The support
    indicates how frequently the itemset appears in the dataset.

    :param df: The pandas DataFrame containing binary data, where rows
        represent transactions and columns represent items.
    :param itemset: A set of strings representing the items for which
        the support is calculated. Each string corresponds to a column
        name in the DataFrame.
    :return: The support value as a float, representing the fraction of
        transactions containing the itemset.
    """
    if not itemset:
        return 0.0
    condition = df[list(itemset)].all(axis=1)
    return condition.mean()


def get_confidence(df: pd.DataFrame, base: Set[str], add: Set[str]) -> float:
    """
    Calculates the confidence of a rule, which is the ratio of the support
    of the combined itemset (base union add) to the support of the base
    itemset. Confidence measures how likely it is that the add itemset
    will occur given that the base itemset already occurred. If the base
    or add sets are empty, or the support of the base itemset is zero, the
    confidence is zero. This function is typically used in association
    rule mining to evaluate rules.

    :param df: A pandas DataFrame containing the transaction dataset.
    :param base: A set of base items.
    :param add: A set of additional items.
    :return: A float representing the confidence of the rule.
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
    Calculate the lift for a given combination of features.

    The lift is a statistical measure that evaluates the strength of the co-occurrence
    relationship between two sets of features. It is calculated as the ratio of the
    observed support for both features occurring together to the expected support
    if they were independent. A lift value greater than 1 indicates a positive
    association between the features, while a value less than 1 suggests a negative
    association. A value equal to 1 implies independence.

    :param df: A DataFrame containing the data from which to calculate support values.
    :param base: A set of features representing the base feature combination.
    :param add: A set of additional features to calculate the lift with respect to
        the base features.
    :return: The lift value representing the co-occurrence strength between the
        base and additional features.
    :rtype: float
    """
    support_base = get_support(df, base)
    support_add = get_support(df, add)
    support_both = get_support(df, base | add)

    if support_base == 0 or support_add == 0:
        return 0.0

    return support_both / (support_base * support_add)


def evaluate_itemset_across_rounds(base: Set[str], add: Set[str], use_mapping: bool = False,
                                   plot: bool = False) -> None:
    """
    Evaluates the correlation and significance of relationships between a base set of attributes and an
    additional set of attributes across multiple rounds of a dataset. The analysis includes computing
    support, confidence, and lift measures for each round. The operation prints warnings where
    necessary, including missing attributes in datasets or invalid input sets, and outputs the computed
    measures for valid rounds.

    :param base: The base set of attributes.
    :type base: Set[str]
    :param add: The set of attributes to evaluate in relation to the base.
    :type add: Set[str]
    :param use_mapping: Indicates whether special formatting or mapping should be applied to the item labels.
                        Defaults to False.
    :type use_mapping: bool
    :return: None
    """
    if not base or not add:
        print("⚠️ Both base and add must be non-empty sets.")
        return

        # Pretty print the evaluated rule
    for i in base:
        text, bin_label = format_item(i, use_mapping)
        line = f"\"{text}\""
        if bin_label:
            line += f" [{bin_label}]"
        print(line)
    print("  ->")
    for i in add:
        text, bin_label = format_item(i, use_mapping)
        line = f"\"{text}\""
        if bin_label:
            line += f" [{bin_label}]"
        print(line)

    full_itemset = base.union(add)
    support_values = []
    confidence_values = []
    valid_rounds = []

    for round_num in range(1, 10):
        df = pd.read_csv(f"../data/essround{round_num}.csv")
        # df.drop(["clsprty_1_3", "clsprty_2_3", "clsprty_3_3"], axis=1, inplace=True)
        if not full_itemset.issubset(df.columns):
            missing = full_itemset - set(df.columns)
            print(f"Round {round_num}: ⚠️ Missing attributes in itemset: {missing}")
            continue

        sup = get_support(df, full_itemset)
        confidence = get_confidence(df, base, add)
        lift = get_lift(df, base, add)

        print(f"\n================= Round #{round_num} =================")
        print(f"Support: {sup:.4f} - Confidence: {confidence:.4f} - Lift: {lift:.4f}")

        support_values.append(sup)
        confidence_values.append(confidence)
        valid_rounds.append(round_num)

    if plot and support_values:
        plot_support_confidence_zoomed(valid_rounds, support_values, confidence_values)
    elif not support_values:
        print("⚠️ No valid data to plot.")


def plot_support_confidence_zoomed(rounds, support_values, confidence_values):
    min_y = min(min(support_values), min(confidence_values))
    max_y = max(max(support_values), max(confidence_values))
    padding = 0.05

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, support_values, marker='o', label='Support', linestyle='-')
    plt.plot(rounds, confidence_values, marker='s', label='Confidence', linestyle='--')
    plt.title("Support & Confidence across Rounds")
    plt.xlabel("Round")
    plt.ylabel("Metric Value")
    plt.ylim(min_y - padding, max_y + padding)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_rule_evolution(round: int, num_of_rules: int = 10, min_support: float = 0.5,
                           min_confidence: float = 1, use_mapping: bool = False, plot: bool = False) -> None:
    """
    Analyzes the evolution of rules in a given dataset round and evaluates selected
    rules across multiple rounds of data. The function reads a dataset corresponding
    to a specific round, extracts the concept lattice, generates association rules,
    and evaluates valid rules among the top specified ones with given support and
    confidence thresholds.

    :param round: The round number of the dataset to analyze.
    :type round: int
    :param num_of_rules: The number of top rules to analyze. Default is 10.
    :type num_of_rules: int
    :param min_support: The minimum support threshold for association rule generation.
        Default is 0.5.
    :type min_support: float
    :param min_confidence: The minimum confidence threshold for association rule
        generation. Default is 1.
    :type min_confidence: float
    :param use_mapping: A flag to determine whether to use item/attribute mappings in
        evaluating rules. Default is False.
    :type use_mapping: bool
    :return: None. The function performs analysis and outputs results directly.
    :rtype: NoneType
    """

    df = pd.read_csv(f"../data/essround{round}.csv")
    # df.drop(["clsprty_1_3", "clsprty_2_3", "clsprty_3_3"], axis=1, inplace=True)
    objects, attributes, relation = create_context_from_dataframe(df)
    context = Context(objects, attributes, relation)

    print(f"\nAnalyzing top {num_of_rules} rules from Round {round} with minimum support {min_support}")
    rules = list(context.get_association_rules(min_support=min_support, min_confidence=min_confidence))
    filtered_rules = list(extract_valid_rules(rules))

    if not filtered_rules:
        print("⚠️ No valid rules found.")
        return

    for idx, (base, add) in enumerate(filtered_rules[:num_of_rules], 1):
        base_set = set(base)
        add_set = set(add)
        print(f"\n🔍 Rule #{idx}:")
        evaluate_itemset_across_rounds(base_set, add_set, use_mapping=use_mapping, plot=plot)


def show_rules_network(rules, show_metrics=False):
    """
    Visualizes association rules as a network graph.
    Each rule is a directed edge from antecedent(s) to consequent(s).
    Edge thickness represents confidence.
    If show_metrics=True, edge labels will show confidence and support.
    """
    G = nx.DiGraph()
    edge_labels = {}
    confidences = []
    node_roles = {}  # Track which nodes are consequents for coloring

    for rule in rules:
        stat = rule.ordered_statistics[0]
        antecedent_items = list(stat.items_base)
        consequent_items = list(stat.items_add)

        confidence = stat.confidence
        support = rule.support
        confidences.append(confidence)

        ant_str = "\n".join(sorted(antecedent_items))
        con_str = "\n".join(sorted(consequent_items))

        G.add_node(ant_str)
        G.add_node(con_str)
        G.add_edge(ant_str, con_str, confidence=confidence, support=support)

        node_roles[ant_str] = node_roles.get(ant_str, "antecedent")
        node_roles[con_str] = "consequent"

        if show_metrics:
            edge_labels[(ant_str, con_str)] = f"conf: {confidence:.2f}, sup: {support:.2f}"

    if not confidences:
        print("⚠️ No rules to display in graph.")
        return

    # Normalize edge widths by confidence
    min_conf = min(confidences)
    max_conf = max(confidences)
    edge_widths = [
        2 + 6 * ((G[u][v]['confidence'] - min_conf) / (max_conf - min_conf + 1e-6))
        for u, v in G.edges()
    ]

    # Assign colors: orange for consequents, blue for others
    node_colors = [
        "orange" if node_roles.get(n) == "consequent" else "skyblue"
        for n in G.nodes()
    ]

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=2000,
        font_size=8,
        arrows=True,
        width=edge_widths,
        edge_color='gray',
        alpha=0.8  # Slight transparency
    )

    if show_metrics:
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_color='red',
            font_size=7
        )

    plt.title('Association Rules Network Graph')
    plt.tight_layout()
    plt.show()

# df = pd.read_csv("../data/essround8.csv")
# get_and_show_rules(df, min_support=0.6, min_confidence=1, use_mapping=False)
