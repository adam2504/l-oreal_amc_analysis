import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
import re
from collections import defaultdict, Counter
import numpy as np

# Configure page
st.set_page_config(page_title="Data Analytics Dashboard", page_icon="📊", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'channel_colors' not in st.session_state:
    st.session_state.channel_colors = {}

# Color palette for channels
DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

def main():
    st.title("📊 Data Analytics Dashboard")

    # Tab layout
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Data Upload", "📋 Data Table", "📈 Conversion Paths", "📖 Documentation"])

    with tab1:
        data_upload_tab()

    with tab2:
        if st.session_state.data is not None:
            data_table_tab()
        else:
            st.info("Please upload a CSV file first")

    with tab3:
        if st.session_state.data is not None:
            conversion_paths_tab()
        else:
            st.info("Please upload a CSV file first")

    with tab4:
        documentation_tab()

def data_upload_tab():
    st.header("Upload CSV File")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file, low_memory=False)

        # Data cleaning and preprocessing
        df = preprocess_data(df)

        st.success("✅ File uploaded successfully!")

        # Basic statistics
        st.subheader("Data Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            date_min = df['start_date'].min() if 'start_date' in df.columns and df['start_date'].notna().any() else 'N/A'
            date_max = df['end_date'].max() if 'end_date' in df.columns and df['end_date'].notna().any() else 'N/A'
            st.metric("Date Range", f"{str(date_min).split('T')[0] if date_min != 'N/A' else 'N/A'} - {str(date_max).split('T')[0] if date_max != 'N/A' else 'N/A'}")

        # Channels list
        st.subheader("Channels Found")
        if 'channel' in df.columns:
            channels_list = sorted([str(c) for c in df['channel'].dropna().unique() if str(c).strip()])
            if channels_list:
                st.write("**Available Channels:**")
                channels_display = ", ".join(channels_list[:15])  # Show first 15 channels
                if len(channels_list) > 15:
                    channels_display += f" (+{len(channels_list) - 15} more)"
                st.info(channels_display)
            else:
                st.info("No channels found in the data")

        # Video information checkbox
        has_video = st.checkbox("File contains video information", value=True)

        # Filter columns based on video option
        df_filtered = df
        if not has_video:
            video_cols = [col for col in df.columns if 'video' in col.lower()]
            if video_cols:
                df_filtered = df.drop(columns=video_cols)
                st.info(f"Removed {len(video_cols)} video-related columns: {', '.join(video_cols)}")

        st.session_state.data = df_filtered

        st.subheader("Data Preview")
        st.dataframe(df_filtered.head(), use_container_width=True)

def preprocess_data(df):
    """Preprocess the uploaded data"""
    # Convert numerical columns
    numeric_cols = []
    for col in df.columns:
        if col not in ['client', 'brand', 'use_case_name', 'extract_date', 'start_date',
                      'end_date', 'analysis_level', 'granularity', 'channel', 'place_channel',
                      'path', 'description']:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_cols.append(col)
            except:
                pass

    # Calculate additional KPIs
    if 'product_sales' in df.columns and 'impressions_cost' in df.columns:
        df['roas'] = df['product_sales'] / df['impressions_cost'].replace(0, np.nan)

    if 'impressions_cost' in df.columns and 'detail_page_view' in df.columns:
        df['cpdpv'] = df['impressions_cost'] / df['detail_page_view'].replace(0, np.nan)

    if 'purchases' in df.columns and 'impressions_cost' in df.columns:
        df['cpa'] = df['impressions_cost'] / df['purchases'].replace(0, np.nan)

    # Extract channels from data for color initialization
    channels = []
    if 'channel' in df.columns:
        channels.extend(df['channel'].dropna().unique())
    if 'path' in df.columns:
        for path in df['path'].dropna():
            try:
                # Extract channels from path like "[1/SEARCH, 2/DSP]"
                matches = re.findall(r'/(\w+(?:\s+\w+)*)', str(path))
                channels.extend(matches)
            except:
                pass

    channels = list(set(channels))

    # Initialize colors if not already done
    for i, channel in enumerate(channels):
        if channel not in st.session_state.channel_colors:
            st.session_state.channel_colors[channel] = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]

    return df

def data_table_tab():
    st.header("Data Table & Filters")

    df = st.session_state.data.copy()

    # Extract path channels for filter
    path_channels = set()
    if 'path' in df.columns:
        for path in df['path'].dropna():
            try:
                matches = re.findall(r'/([A-Z\s]+)', path.upper())
                path_channels.update(matches)
            except:
                pass
    path_channels = sorted(list(path_channels))

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        analysis_level_filter = st.multiselect(
            "Analysis Level",
            options=df['analysis_level'].unique(),
            default=df['analysis_level'].unique()
        )

    with col2:
        granularity_filter = st.multiselect(
            "Granularity",
            options=df['granularity'].dropna().unique(),
            default=df['granularity'].dropna().unique()
        )

    with col3:
        channel_filter = st.multiselect(
            "Path Channel",
            options=path_channels,
            default=path_channels[:3] if path_channels else []  # Default to first 3
        )

    # Apply filters
    filtered_df = df.copy()
    if analysis_level_filter:
        filtered_df = filtered_df[filtered_df['analysis_level'].isin(analysis_level_filter)]
    if granularity_filter:
        filtered_df = filtered_df[filtered_df['granularity'].isin(granularity_filter)]
    if channel_filter:
        def contains_selected_channels(path):
            try:
                matches = re.findall(r'/([A-Z\s]+)', str(path).upper())
                return all(channel in matches for channel in channel_filter)
            except:
                return False
        filtered_df = filtered_df[filtered_df['path'].apply(contains_selected_channels)]

    # Color picker for channels
    st.subheader("Channel Colors")
    color_cols = st.columns(min(4, len(st.session_state.channel_colors)))

    for i, (channel, color) in enumerate(st.session_state.channel_colors.items()):
        with color_cols[i % 4]:
            new_color = st.color_picker(f"{channel}", color)
            if new_color != color:
                st.session_state.channel_colors[channel] = new_color

    # Display table
    st.subheader(f"Filtered Data ({len(filtered_df)} rows)")

    # Note about conversion KPIs (orange highlighting not supported in this Streamlit version)
    conversion_columns = [
        'user_purchased', 'product_sales', 'purchases', 'units_sold',
        'user_total_purchased', 'total_purchases', 'total_product_sales', 'total_units_sold',
        'ntb_purchased', 'ntb_product_sales', 'ntb_purchases', 'ntb_units_sold',
        'ntb_total_purchased', 'total_ntb_purchases', 'total_ntb_product_sales', 'total_ntb_units_sold'
    ]

    existing_conv_cols = [col for col in conversion_columns if col in filtered_df.columns]
    if existing_conv_cols:
        st.info(f"⚠️ **Conversion KPIs in this table:** {', '.join(existing_conv_cols)} - (Note: Column highlighting requires a newer Streamlit version)")

    st.dataframe(filtered_df, use_container_width=True)

    # Statistics
    if len(filtered_df) > 0:
        st.subheader("Quick Statistics")

        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            selected_metric = st.selectbox(
                "Select metric for analysis",
                options=numeric_cols,
                index=min(10, len(numeric_cols)-1)  # Default to a metric column
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Average {selected_metric}",
                         f"{filtered_df[selected_metric].mean():.2f}")
            with col2:
                st.metric(f"Max {selected_metric}",
                         f"{filtered_df[selected_metric].max():.2f}")
            with col3:
                st.metric(f"Min {selected_metric}",
                         f"{filtered_df[selected_metric].min():.2f}")

def conversion_paths_tab():
    st.header("Conversion Path Analysis")

    df = st.session_state.data.copy()
    path_df = df[df['analysis_level'] == 'Path to conversion'].copy()

    if len(path_df) == 0:
        st.warning("No conversion path data found")
        return

    # Filters for path visualization
    col1, col2 = st.columns(2)

    with col1:
        granularity_options = path_df['granularity'].unique()
        selected_granularity = st.selectbox("Select Granularity", granularity_options)

    with col2:
        max_paths = st.slider("Max paths to show", 1, 50, 10)

    filtered_paths = path_df[path_df['granularity'] == selected_granularity]

    if len(filtered_paths) == 0:
        st.warning(f"No data for {selected_granularity} granularity")
        return

    # Display top paths by a metric
    metric_options = ['purchases', 'product_sales', 'purchases', 'clicks_all_users']
    selected_metric = st.selectbox("Sort paths by", metric_options)

    # Sort and get top paths
    sorted_paths = filtered_paths.sort_values(by=selected_metric, ascending=False).head(max_paths)

    # Create Sankey chart
    create_sankey_chart(sorted_paths, selected_granularity, selected_metric)

    # Individual path details
    st.subheader("Path Details")
    selected_path_idx = st.selectbox(
        "Select a path to view details",
        options=range(len(sorted_paths)),
        format_func=lambda x: f"{x+1}: {sorted_paths.iloc[x]['description']}"
    )

    selected_path_data = sorted_paths.iloc[selected_path_idx]
    st.json(selected_path_data.to_dict())

def create_sankey_chart(df, granularity_type, metric):
    """Create a Sankey diagram for conversion paths"""

    # Parse paths and create flow data
    nodes = set()
    links = []

    for _, row in df.iterrows():
        path_str = str(row['path'])
        if path_str and path_str != 'nan':
            try:
                # Extract steps from path like "[1/SEARCH, 2/DSP, 3/CONVERSION]"
                matches = re.findall(r'(\d+)/([A-Z\s]+)', path_str.upper())
                if matches and len(matches) > 1:
                    for i in range(len(matches) - 1):
                        current = matches[i][1].strip()
                        next_step = matches[i + 1][1].strip()

                        nodes.add(current)
                        nodes.add(next_step)

                        links.append({
                            'source': current,
                            'target': next_step,
                            'value': row[metric] if pd.notna(row[metric]) else 1
                        })
            except:
                continue

    if not nodes or not links:
        st.warning("Could not parse conversion paths")
        return

    # Aggregate links
    link_dict = {}
    for link in links:
        key = (link['source'], link['target'])
        if key in link_dict:
            link_dict[key] += link['value']
        else:
            link_dict[key] = link['value']

    # Create node list and mapping
    node_list = list(nodes)
    node_map = {node: i for i, node in enumerate(node_list)}

    # Prepare Sankey data
    link_sources = []
    link_targets = []
    link_values = []

    for (source, target), value in link_dict.items():
        if source in node_map and target in node_map:
            link_sources.append(node_map[source])
            link_targets.append(node_map[target])
            link_values.append(value)

    # Create color mapping
    node_colors = []
    for node in node_list:
        if node in st.session_state.channel_colors:
            node_colors.append(st.session_state.channel_colors[node])
        else:
            node_colors.append(DEFAULT_COLORS[len(node_colors) % len(DEFAULT_COLORS)])

    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_list,
            color=node_colors
        ),
        link=dict(
            source=link_sources,
            target=link_targets,
            value=link_values,
            color=[node_colors[src] for src in link_sources]
        )
    )])

    fig.update_layout(
        title_text=f"Conversion Paths ({granularity_type}) - {metric}",
        font_size=10
    )

    st.plotly_chart(fig, use_container_width=True)

def documentation_tab():
    st.header("Documentation")

    tab1, tab2 = st.tabs(["📚 KPI Glossary", "📊 Campaign Summary"])

    with tab1:
        st.subheader("Key Performance Indicators (KPIs)")

        kpi_definitions = {
            "Impressions": "Number of times an ad is shown",
            "Clicks": "Number of times users clicked on the ad",
            "Click-Through Rate (CTR)": "Clicks / Impressions * 100",
            "Cost Per Click (CPC)": "Total cost / Clicks",
            "Cost Per Mille (CPM)": "Cost per 1,000 impressions",
            "Conversions": "Number of desired actions (purchases, signups, etc.)",
            "Conversion Rate": "Conversions / Clicks * 100",
            "Cost Per Acquisition (CPA)": "Total cost / Conversions",
            "Return on Ad Spend (ROAS)": "Revenue / Ad spend",
            "Cost Per Detail Page View (CPDPV)": "Total cost / Detail page views",
            "Average Time": "Average time spent on page/engagement",
            "Complete Views": "Number of times video was watched to completion"
        }

        for kpi, definition in kpi_definitions.items():
            with st.expander(f"**{kpi}**"):
                st.write(definition)

        st.subheader("Analysis Levels")
        st.markdown("""
        - **Media Mix**: Overall performance across channels
        - **Path to Conversion**: User journey analysis showing touchpoints leading to conversion
        - **Campaign Performance**: Specific campaign metrics
        - **Place of Channel**: Position of channel in conversion path (Beginner, Intermediate, Finisher)
        """)

    with tab2:
        if st.session_state.data is not None:
            df = st.session_state.data

            st.subheader("Campaign Summary")

            # Overall metrics
            col1, col2, col3, col4 = st.columns(4)

            total_spend = df['impressions_cost'].sum()
            total_purchases = df['purchases'].sum()
            total_sales = df['product_sales'].sum()

            with col1:
                st.metric("Total Spend", f"${total_spend:,.0f}")
            with col2:
                st.metric("Total Purchases", f"{total_purchases:,.0f}")
            with col3:
                st.metric("Total Sales", f"${total_sales:,.0f}")
            with col4:
                roas = total_sales / total_spend if total_spend > 0 else 0
                st.metric("Overall ROAS", f"{roas:.2f}")

            # Channel performance
            st.subheader("Channel Performance")
            if 'channel' in df.columns:
                channel_stats = df.groupby('channel').agg({
                    'impressions_cost': 'sum',
                    'purchases': 'sum',
                    'product_sales': 'sum'
                }).reset_index()

                channel_stats['ROAS'] = channel_stats['product_sales'] / channel_stats['impressions_cost']
                channel_stats = channel_stats.sort_values('impressions_cost', ascending=False)

                st.dataframe(channel_stats, use_container_width=True)

                # Quick chart
                fig = go.Figure(data=[
                    go.Bar(name='Spend', x=channel_stats['channel'], y=channel_stats['impressions_cost']),
                    go.Bar(name='Sales', x=channel_stats['channel'], y=channel_stats['product_sales'])
                ])
                fig.update_layout(barmode='group', title="Channel Spend vs Sales")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload data to see campaign summary")

if __name__ == "__main__":
    main()
