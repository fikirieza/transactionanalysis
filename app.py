import streamlit as st
import pandas as pd
from fpgrowth_py import fpgrowth
import graphviz

st.title('Analisis Keranjang Belanja: FP-Growth dan Visualisasi FP-Tree')

# Define the FPTreeNode class (from previous code)
class FPTreeNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.node_link = None

    def increment_count(self, count):
        self.count += count

# Function to build the FP-Tree (adapted from previous code)
def build_fp_tree(dataset, frequent_itemsets, min_support_count):
    # Create header table
    header_table = {}
    unique_frequent_items = set()
    for itemset in frequent_itemsets:
        for item in itemset:
            unique_frequent_items.add(item)

    # Count frequencies of individual frequent items
    item_counts = {}
    for transaction in dataset:
        for item in transaction:
            if item in unique_frequent_items:
                 item_counts[item] = item_counts.get(item, 0) + 1

    # Filter out items below min support count and build header table
    items_to_process = sorted([item for item, count in item_counts.items() if count >= min_support_count],
                              key=lambda item: item_counts[item], reverse=True)

    if not items_to_process:
        return None, {}

    for item in items_to_process:
        header_table[item] = [item_counts[item], None] # [count, node_link]


    # Initialize root node
    root = FPTreeNode("Null", 1, None)

    # Build the tree
    for transaction in dataset:
        current_node = root
        # Sort transaction items by frequency in descending order
        sorted_transaction = [item for item in transaction if item in items_to_process]
        sorted_transaction.sort(key=lambda item: header_table[item][0], reverse=True)

        for item in sorted_transaction:
            if item not in current_node.children:
                new_node = FPTreeNode(item, 1, current_node)
                current_node.children[item] = new_node

                # Update header table node link
                if header_table[item][1] is None:
                    header_table[item][1] = new_node
                else:
                    # Find the last node in the node link
                    current_link_node = header_table[item][1]
                    while current_link_node.node_link is not None:
                        current_link_node = current_link_node.node_link
                    current_link_node.node_link = new_node

                current_node = new_node
            else:
                current_node.children[item].increment_count(1)
                current_node = current_node.children[item]

    return root, header_table

# Function to build the Graphviz representation of the FP-Tree
def build_graphviz_tree(node, graph):
    node_label = f"{node.item}\n({node.count})"
    # Use a consistent string ID for Graphviz nodes
    node_id = str(id(node))
    graph.node(node_id, label=node_label)

    for child_item, child_node in node.children.items():
        child_label = f"{child_node.item}\n({child_node.count})"
        child_id = str(id(child_node))
        graph.node(child_id, label=child_label)
        graph.edge(node_id, child_id, label=str(child_node.count))
        build_graphviz_tree(child_node, graph)


uploaded_file = st.file_uploader("Unggah File CSV", type=["csv"])

# Allow user to input min_support and min_confidence
min_support = st.slider("Minimum Support", min_value=0.01, max_value=1.0, value=0.03, step=0.01)
min_confidence = st.slider("Minimum Confidence", min_value=0.01, max_value=1.0, value=0.7, step=0.01)


if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File berhasil dimuat!")
        st.dataframe(df.head())

        # Preprocessing: ubah kolom Nama Barang menjadi list of lists
        if 'Nama Barang' in df.columns:
            # Handle potential NaN values in 'Nama Barang'
            df['Nama Barang'] = df['Nama Barang'].fillna('')
            transaksi = df['Nama Barang'].apply(lambda x: [item.strip() for item in str(x).split(',') if item.strip()])
            # Filter out empty transactions that might result from empty strings or only commas
            daftar_transaksi = [t for t in transaksi.tolist() if t]

            if not daftar_transaksi:
                 st.error("No valid transactions found in the 'Nama Barang' column after processing. Please check your data.")
            else:
                st.info(f"Proses {len(daftar_transaksi)} transaksi...")
                # Jalankan FP-Growth
                try:
                    result = fpgrowth(daftar_transaksi, minSupRatio=min_support, minConf=min_confidence)

                    if result is not None:
                        frequent_itemsets, rules = result
                        st.subheader("Frequent Itemsets:")
                        if frequent_itemsets:
                            for itemset in frequent_itemsets:
                                st.write(itemset)
                        else:
                            st.info("No frequent itemsets found with the given minimum support.")


                        st.subheader("Association Rules:")
                        if rules:
                            for rule in rules:
                                st.write(rule)
                        else:
                            st.info("No association rules found with the given minimum confidence.")


                        # Build and visualize the FP-Tree
                        st.subheader("Visualisasi FP-Tree:")
                        min_support_count = int(min_support * len(daftar_transaksi))

                        # Ensure frequent_itemsets is not empty before building the tree
                        if frequent_itemsets:
                            try:
                                fp_tree_root, header_table = build_fp_tree(daftar_transaksi, frequent_itemsets, min_support_count)

                                if fp_tree_root and fp_tree_root.children: # Check if the tree is not just the root
                                    dot = graphviz.Digraph(comment='FP-Tree')
                                    dot.attr(rankdir='TB') # Top-to-bottom layout

                                    # Start building the graph from the root
                                    build_graphviz_tree(fp_tree_root, dot)

                                    # Display the graph in Streamlit
                                    st.graphviz_chart(dot.source)
                                elif fp_tree_root:
                                     st.info("FP-Tree built, but has no branches meeting minimum support (only root).")
                                else:
                                     st.info("Could not build FP-Tree. No frequent items meet minimum support count.")
                            except Exception as tree_e:
                                st.error(f"Error building or visualizing FP-Tree: {tree_e}")
                                st.exception(tree_e)


                        else:
                             st.info("Cannot build FP-Tree as no frequent itemsets were found.")


                    else:
                        frequent_itemsets = []
                        rules = []
                        st.warning("FP-Growth result is None. Frequent itemsets or rules not found.")
                except Exception as fpgrowth_e:
                    st.error(f"Error during FP-Growth execution: {fpgrowth_e}")
                    st.exception(fpgrowth_e)


        else:
            st.error("CSV file must contain a column named 'Nama Barang'. Please check your file.")

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty. Please upload a file with data.")
    except pd.errors.ParserError:
        st.error("Error parsing the CSV file. Please ensure it is a valid CSV format.")
    except Exception as e:
        st.error(f"An unexpected error occurred during file loading or initial processing: {e}")
        st.exception(e)
