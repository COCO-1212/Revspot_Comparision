import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Configuration ---
st.set_page_config(
    page_title="Data Comparison Dashboard",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Static Definitions ---
DATA_COLUMNS = [
    "age_group", "gender", "occupation_type", "Categorized_Designation",
    "Categorized_Zone_Revspot", "Categorized_Earnings_actual",
    "Categorized_Years_in_City" 
]
FILTER_COLUMN = "enriched"

# --- Helper Functions ---

@st.cache_data
def load_csv(uploaded_file):
    """
    Loads a CSV file, filters based on the 'enriched' column,
    and prepares data for analysis.
    """
    if uploaded_file is None:
        return None
    try:
        df = pd.read_csv(uploaded_file)
        original_rows = len(df)
        
        missing_data_cols = [col for col in DATA_COLUMNS if col not in df.columns]
        if missing_data_cols:
            st.warning(f"In '{uploaded_file.name}', the following columns are missing: {', '.join(missing_data_cols)}. Related analyses will be skipped.")

        if FILTER_COLUMN in df.columns:
            df[FILTER_COLUMN] = df[FILTER_COLUMN].astype(str).str.lower().isin(['true', '1'])
            df_filtered = df[df[FILTER_COLUMN]].copy()
            st.sidebar.info(f"Filtered '{uploaded_file.name}': Using {len(df_filtered)} of {original_rows} rows where '{FILTER_COLUMN}' is TRUE.")
            df = df_filtered
        else:
            st.sidebar.warning(f"'{FILTER_COLUMN}' column not found in '{uploaded_file.name}'. Using all {original_rows} rows.")
            df = df.copy()

        # Keep Categorized_Years_in_City as categorical - don't convert to numeric
        # All columns should be treated as categorical for consistent analysis
        
        return df
    except Exception as e:
        st.error(f"Error loading '{uploaded_file.name}': {e}")
        return None

def get_unique_values(df1, df2, column):
    """Get unique values from both dataframes for a given column"""
    if column not in df1.columns and column not in df2.columns:
        return []
    
    values = set()
    if column in df1.columns:
        values.update(df1[column].dropna().unique())
    if column in df2.columns:
        values.update(df2[column].dropna().unique())
    
    return sorted(list(values))

def filter_dataframe(df, column, selected_values):
    """Filter dataframe based on selected values for a column"""
    if column not in df.columns or not selected_values:
        return df
    return df[df[column].isin(selected_values)].copy()

def create_univariate_pivot(df, column_name):
    """Creates a pivot table for a single column with counts and percentages."""
    if column_name not in df.columns or df[column_name].isnull().all():
        return pd.DataFrame(), f"Column '{column_name}' not found or is empty."
    counts = df[column_name].value_counts(dropna=True)
    total_valid = counts.sum()
    if total_valid == 0:
        return pd.DataFrame(), f"No valid data for '{column_name}'."
    pivot = counts.reset_index()
    pivot.columns = [column_name, 'Count']
    pivot['Percentage (%)'] = (pivot['Count'] / total_valid * 100).round(1)
    return pivot.sort_values(by='Count', ascending=False), None

def create_cross_tab_pivot(df, index_col, columns_col, normalization_choice):
    """
    Creates cross-tabulation with raw counts (including totals) and normalized percentages.
    """
    if index_col not in df.columns or columns_col not in df.columns:
        return None, None, f"Missing one or both columns: '{index_col}', '{columns_col}'."
    
    # Map user-friendly choice to pandas parameter
    normalize_map = {
        'Grand Total': 'all', 
        'Column Total (Column %)': 'columns', 
        'Row Total (Row %)': 'index',
        'No Normalization': None
    }
    
    try:
        # Create pivot for raw counts with margins for totals
        count_pivot = pd.crosstab(df[index_col], df[columns_col], dropna=True, margins=True, margins_name='Total')
        
        # Create pivot for percentages based on user's choice
        if normalization_choice == 'No Normalization':
            percent_pivot = count_pivot.copy()
        else:
            percent_pivot = pd.crosstab(
                df[index_col],
                df[columns_col],
                dropna=True,
                normalize=normalize_map[normalization_choice]
            ).mul(100).round(2)
        
        if count_pivot.empty:
            return pd.DataFrame(), pd.DataFrame(), "No valid data for this combination."

        return count_pivot, percent_pivot, None
    except Exception as e:
        return None, None, f"Error creating pivot: {e}"

def align_pivots(pivot1, pivot2):
    """Aligns two pivots to have the same rows and columns for accurate comparison."""
    all_rows = pivot1.index.union(pivot2.index)
    all_cols = pivot1.columns.union(pivot2.columns)
    p1_aligned = pivot1.reindex(index=all_rows, columns=all_cols, fill_value=0)
    p2_aligned = pivot2.reindex(index=all_rows, columns=all_cols, fill_value=0)
    return p1_aligned, p2_aligned
    
def display_univariate_comparison(df1_name, df2_name, col_name, pivot1, pivot2):
    """Generates a bar chart and a side-by-side comparison table for a single column."""
    if pivot1.empty and pivot2.empty:
        st.warning(f"No data available to compare for '{col_name}'.")
        return
    plot_df = pd.concat([pivot1.assign(Source=df1_name), pivot2.assign(Source=df2_name)])
    category_order = plot_df.groupby(col_name)['Percentage (%)'].sum().sort_values(ascending=False).index
    fig = px.bar(plot_df, x=col_name, y='Percentage (%)', color='Source', barmode='group', title=f'Distribution of {col_name}', labels={'Percentage (%)': 'Percentage (%)', col_name: col_name.replace("_", " ").title()}, text='Percentage (%)', template="plotly_white", color_discrete_map={df1_name: '#1f77b4', df2_name: '#ff7f0e'}, category_orders={col_name: category_order})
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, yaxis_title="Percentage (%)", xaxis_title=None, legend_title_text="", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified")
    comp_df = pd.merge(pivot1[[col_name, 'Percentage (%)']], pivot2[[col_name, 'Percentage (%)']], on=col_name, how='outer', suffixes=(f'_{df1_name}', f'_{df2_name}')).fillna(0)
    comp_df['Difference'] = (comp_df[f'Percentage (%)_{df2_name}'] - comp_df[f'Percentage (%)_{df1_name}']).round(1)
    comp_df = comp_df.rename(columns={f'Percentage (%)_{df1_name}': f'{df1_name} (%)', f'Percentage (%)_{df2_name}': f'{df2_name} (%)'})
    comp_df = comp_df.sort_values(by=f'{df1_name} (%)', ascending=False)
    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(f"**Comparative Breakdown**")
        st.dataframe(comp_df.style.background_gradient(cmap='RdYlGn', subset=['Difference'], vmin=-20, vmax=20).format('{:.1f}', subset=['Difference']), use_container_width=True)
    st.markdown("---")

def display_cross_tab_comparison(title, df1_name, df2_name, count1, percent1, count2, percent2, index_col, cols_col, normalization_choice):
    """Displays a detailed comparison of two cross-tabulations using tabs."""
    st.markdown(f"##### {title}")
    
    # Align the pivots
    p1_aligned, p2_aligned = align_pivots(percent1, percent2)
    c1_aligned, c2_aligned = align_pivots(count1, count2)
    
    # Calculate difference only if we have percentages
    if normalization_choice != 'No Normalization':
        diff_aligned = (p2_aligned - p1_aligned).round(2)
    else:
        diff_aligned = (c2_aligned - c1_aligned)

    # Prepare captions based on normalization choice
    captions = {
        'Grand Total': "Percentages are based on the grand total of all records.",
        'Column Total (Column %)': "Each column's percentages sum to 100%. This shows the distribution within each column category.",
        'Row Total (Row %)': "Each row's percentages sum to 100%. This shows the distribution within each row category.",
        'No Normalization': "Raw counts without percentage calculation."
    }

    # Prepare tab names based on normalization
    if normalization_choice == 'No Normalization':
        tab_list = ["📊 Count Difference", f"{df1_name} (Counts)", f"{df2_name} (Counts)", "Raw Counts with Totals"]
        diff_title = 'Count Difference'
        diff_label = "Count Diff"
    else:
        tab_list = ["📊 Heatmap of Difference", f"{df1_name} (%)", f"{df2_name} (%)", "Raw Counts with Totals"]
        diff_title = 'Percentage Point Difference'
        diff_label = "Percentage Diff"
        
    tab1, tab2, tab3, tab4 = st.tabs(tab_list)

    with tab1:
        st.markdown(f"**{diff_title}**: {df2_name} minus {df1_name}")
        st.caption(captions[normalization_choice])
        
        color_scale = 'RdYlGn' if normalization_choice != 'No Normalization' else 'RdBu'
        fig_heatmap = px.imshow(
            diff_aligned, 
            text_auto=True, 
            aspect="auto", 
            color_continuous_scale=color_scale,
            labels=dict(
                x=cols_col.replace("_", " "), 
                y=index_col.replace("_", " "), 
                color=diff_label
            ), 
            title=diff_title
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
    with tab2:
        st.markdown(f"**{df1_name} {'Percentage' if normalization_choice != 'No Normalization' else 'Count'} Distribution**")
        st.caption(captions[normalization_choice])
        if normalization_choice != 'No Normalization':
            st.dataframe(p1_aligned.style.background_gradient(cmap='Blues').format("{:.1f}%"), use_container_width=True)
        else:
            st.dataframe(c1_aligned.style.background_gradient(cmap='Blues'), use_container_width=True)
            
    with tab3:
        st.markdown(f"**{df2_name} {'Percentage' if normalization_choice != 'No Normalization' else 'Count'} Distribution**")
        st.caption(captions[normalization_choice])
        if normalization_choice != 'No Normalization':
            st.dataframe(p2_aligned.style.background_gradient(cmap='Oranges').format("{:.1f}%"), use_container_width=True)
        else:
            st.dataframe(c2_aligned.style.background_gradient(cmap='Oranges'), use_container_width=True)
            
    with tab4:
        st.markdown(f"**Raw Counts for {df1_name}**")
        st.dataframe(c1_aligned, use_container_width=True)
        st.markdown(f"**Raw Counts for {df2_name}**")
        st.dataframe(c2_aligned, use_container_width=True)
    st.markdown("---")

# Remove the numeric analysis function since Categorized_Years_in_City is categorical
# def display_grouped_average_comparison - REMOVED

# --- Streamlit App Layout ---

st.title("✨ Advanced Data Comparison Dashboard")
st.markdown("Upload two CSV files to visually compare their distributions and key metrics. The dashboard automatically focuses on `enriched` data for precise insights.")

# --- Sidebar for File Uploads and Info ---
st.sidebar.header("📂 Upload & Configure")
uploaded_file1 = st.sidebar.file_uploader("Upload First CSV (Baseline)", type="csv", key="file1")
uploaded_file2 = st.sidebar.file_uploader("Upload Second CSV (Comparison)", type="csv", key="file2")

df1 = load_csv(uploaded_file1)
df2 = load_csv(uploaded_file2)

df1_name = uploaded_file1.name.replace(".csv", "") if uploaded_file1 else "File 1"
df2_name = uploaded_file2.name.replace(".csv", "") if uploaded_file2 else "File 2"

# --- Main Dashboard Area ---
if df1 is not None and df2 is not None:
    st.success("🎉 Both files are loaded and filtered. Explore the comparisons below!")

    # --- 1. Executive Summary / KPIs ---
    st.header("🚀 Executive Summary", divider='rainbow')
    total_records1 = len(df1)
    total_records2 = len(df2)
    
    # Calculate most common category in Years_in_City for both datasets
    most_common_years1 = df1['Categorized_Years_in_City'].mode()[0] if 'Categorized_Years_in_City' in df1.columns and not df1['Categorized_Years_in_City'].empty else "N/A"
    most_common_years2 = df2['Categorized_Years_in_City'].mode()[0] if 'Categorized_Years_in_City' in df2.columns and not df2['Categorized_Years_in_City'].empty else "N/A"
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label=f"Total Records in {df1_name}", value=f"{total_records1:,}")
    col1.metric(label=f"Most Common Years in City ({df1_name})", value=str(most_common_years1))
    
    col2.metric(label=f"Total Records in {df2_name}", value=f"{total_records2:,}")
    col2.metric(label=f"Most Common Years in City ({df2_name})", value=str(most_common_years2))
    
    col3.metric(label="Change in Records", value=f"{total_records2 - total_records1:+,}", delta=f"{(total_records2 - total_records1) / total_records1:.1%}" if total_records1 else "N/A")
    col3.metric(label="Years Category Change", value="Different" if most_common_years1 != most_common_years2 else "Same")
    
    st.markdown("---")

    # --- 2. Univariate (Single Column) Analysis ---
    st.header("📊 Profile Breakdowns", divider='rainbow')
    st.markdown("Compare the percentage distribution of key categorical attributes. Charts are paired with tables showing the precise differences.")
    categorical_cols = ["age_group", "gender", "occupation_type", "Categorized_Designation", "Categorized_Zone_Revspot", "Categorized_Earnings_actual", "Categorized_Years_in_City"]
    for col in categorical_cols:
        with st.expander(f"**Analysis of: {col.replace('_', ' ').title()}**", expanded=False):
            pivot1, msg1 = create_univariate_pivot(df1, col)
            pivot2, msg2 = create_univariate_pivot(df2, col)
            if msg1 and "not found" in msg1: st.warning(f"For {df1_name}: {msg1}")
            if msg2 and "not found" in msg2: st.warning(f"For {df2_name}: {msg2}")
            if not pivot1.empty or not pivot2.empty:
                display_univariate_comparison(df1_name, df2_name, col, pivot1, pivot2)
            else:
                st.info(f"Column '{col}' not found or contains no data in either file.")

    # --- 3. Enhanced Cross-Tabulation Analysis ---
    st.header("↔️ Cross-Tabulation Insights", divider='rainbow')
    st.markdown("Explore the intersection of two attributes with advanced pivot table features including filtering and percentage calculations.")
    
    cross_tab_pairs = [
        ("age_group", "Categorized_Earnings_actual"), 
        ("occupation_type", "Categorized_Earnings_actual"), 
        ("Categorized_Designation", "Categorized_Earnings_actual"), 
        ("Categorized_Zone_Revspot", "Categorized_Earnings_actual"), 
        ("age_group", "occupation_type"), 
        ("Categorized_Zone_Revspot", "Categorized_Designation"),
        ("Categorized_Years_in_City", "Categorized_Earnings_actual"),
        ("age_group", "Categorized_Years_in_City"),
        ("occupation_type", "Categorized_Years_in_City")
    ]
    
    for idx, (index_col, columns_col) in enumerate(cross_tab_pairs):
        with st.expander(f"**Analysis of: {index_col.replace('_',' ').title()} vs. {columns_col.replace('_',' ').title()}**", expanded=False):
            
            # Create unique keys for each expander's widgets
            unique_key = f"{idx}_{index_col}_{columns_col}"
            
            # Configuration section - Make it more prominent
            st.markdown("---")
            st.markdown("#### 🛠️ Analysis Configuration")
            
            config_col1, config_col2 = st.columns([1, 1])
            
            with config_col1:
                st.markdown("**📊 Percentage Calculation Method:**")
                # Normalization choice with better descriptions
                normalization_options = [
                    'Grand Total',
                    'Column Total (Column %)', 
                    'Row Total (Row %)', 
                    'No Normalization'
                ]
                
                normalization_descriptions = {
                    'Grand Total': "Show percentages of the entire dataset",
                    'Column Total (Column %)': "Each column sums to 100% - shows composition within columns",
                    'Row Total (Row %)': "Each row sums to 100% - shows composition within rows",
                    'No Normalization': "Show raw counts only"
                }
                
                normalization_choice = st.selectbox(
                    "Choose calculation method:",
                    normalization_options,
                    index=0,
                    key=f'cross_tab_norm_{unique_key}',
                    help="Select how percentages should be calculated for the cross-tabulation"
                )
                
                st.caption(f"ℹ️ {normalization_descriptions[normalization_choice]}")
            
            with config_col2:
                st.markdown("**🔍 Filter Data:**")
                
                # Get unique values for filtering
                index_values = get_unique_values(df1, df2, index_col)
                columns_values = get_unique_values(df1, df2, columns_col)
                
                # Multi-select for index column (rows)
                selected_index_values = st.multiselect(
                    f"Filter {index_col.replace('_', ' ').title()}:",
                    options=index_values,
                    default=index_values,
                    key=f'index_filter_{unique_key}',
                    help=f"Choose which {index_col.replace('_', ' ')} values to include"
                )
                
                # Multi-select for columns column
                selected_columns_values = st.multiselect(
                    f"Filter {columns_col.replace('_', ' ').title()}:",
                    options=columns_values,
                    default=columns_values,
                    key=f'columns_filter_{unique_key}',
                    help=f"Choose which {columns_col.replace('_', ' ')} values to include"
                )
            
            st.markdown("---")
            
            # Apply filters to dataframes
            df1_filtered = filter_dataframe(df1, index_col, selected_index_values)
            df1_filtered = filter_dataframe(df1_filtered, columns_col, selected_columns_values)
            
            df2_filtered = filter_dataframe(df2, index_col, selected_index_values)
            df2_filtered = filter_dataframe(df2_filtered, columns_col, selected_columns_values)
            
            # Show filtering info
            if len(selected_index_values) < len(index_values) or len(selected_columns_values) < len(columns_values):
                st.info(f"📊 **Filtered Analysis:** Using {len(df1_filtered):,} records from {df1_name} and {len(df2_filtered):,} records from {df2_name}")
            else:
                st.info(f"📊 **Complete Analysis:** Using all {len(df1_filtered):,} records from {df1_name} and {len(df2_filtered):,} records from {df2_name}")
            
            # Perform analysis on filtered data
            c1, p1, msg1 = create_cross_tab_pivot(df1_filtered, index_col, columns_col, normalization_choice)
            c2, p2, msg2 = create_cross_tab_pivot(df2_filtered, index_col, columns_col, normalization_choice)
            
            if (c1 is None) or (c2 is None):
                st.warning(f"Could not perform analysis for this pair. {msg1 or ''} {msg2 or ''}")
                continue
            if not c1.empty and not c2.empty:
                display_cross_tab_comparison(
                    f"{index_col.replace('_',' ').title()} by {columns_col.replace('_',' ').title()}", 
                    df1_name, df2_name, c1, p1, c2, p2, index_col, columns_col, normalization_choice
                )
            else:
                st.info("No common valid data to display for this cross-tabulation with current filters.")

    # --- 4. Categorical Years in City Analysis ---
    st.header("🏙️ Years in City Category Analysis", divider='rainbow') 
    st.markdown("Analyze the categorical distribution of `Categorized_Years_in_City` across different demographic groups using cross-tabulations.")
    
    # Since Years_in_City is categorical, we'll do cross-tabs instead of averages
    years_cross_tab_pairs = [
        ("Categorized_Years_in_City", "Categorized_Designation"),
        ("Categorized_Years_in_City", "occupation_type"),
        ("Categorized_Years_in_City", "age_group"),
        ("Categorized_Years_in_City", "Categorized_Earnings_actual")
    ]
    
    for idx, (index_col, columns_col) in enumerate(years_cross_tab_pairs):
        with st.expander(f"**Years in City by: {columns_col.replace('_', ' ').title()}**", expanded=False):
            unique_key = f"years_{idx}_{index_col}_{columns_col}"
            
            # Simple configuration for years analysis
            normalization_choice = st.radio(
                "Show as:",
                ('Column Total (Column %)', 'Row Total (Row %)', 'Grand Total', 'No Normalization'),
                horizontal=True,
                index=0,
                key=f'years_norm_{unique_key}',
                help="Column % shows distribution within each demographic group"
            )
            
            c1, p1, msg1 = create_cross_tab_pivot(df1, index_col, columns_col, normalization_choice)
            c2, p2, msg2 = create_cross_tab_pivot(df2, index_col, columns_col, normalization_choice)
            
            if (c1 is None) or (c2 is None):
                st.warning(f"Could not perform analysis. {msg1 or ''} {msg2 or ''}")
                continue
            if not c1.empty and not c2.empty:
                display_cross_tab_comparison(
                    f"Years in City Distribution by {columns_col.replace('_',' ').title()}", 
                    df1_name, df2_name, c1, p1, c2, p2, index_col, columns_col, normalization_choice
                )
            else:
                st.info("No data available for this analysis.")

else:
    st.info("👋 Welcome! Please upload two CSV files in the sidebar to begin the comparison.")
    st.markdown("### 📋 Expected Data Format")
    st.markdown("Your CSV files should contain these columns for optimal analysis:")
    expected_cols = ["age_group", "gender", "occupation_type", "Categorized_Designation", "Categorized_Zone_Revspot", "Categorized_Earnings_actual", "Categorized_Years_in_City", "enriched"]
    for col in expected_cols:
        st.markdown(f"• `{col}`")
    st.markdown("---")
    st.markdown("The `enriched` column should contain `True`/`False` or `1`/`0` values to filter the analysis to enriched records only.")