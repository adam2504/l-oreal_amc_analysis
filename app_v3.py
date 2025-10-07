import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
import re
from collections import defaultdict, Counter
import numpy as np
import time

# Utility functions
def extract_path_channels(df):
    """Extract unique path channels from the data."""
    path_channels = set()
    if 'path' in df.columns:
        for path in df['path'].dropna():
            try:
                matches = re.findall(r'/([A-Z\s]+)', path.upper())
                path_channels.update(matches)
            except:
                pass
    return sorted(list(path_channels))

def get_column_highlights(df):
    """Get column categories for highlighting."""
    conversion_columns = [
        'user_purchased', 'product_sales', 'purchases', 'units_sold',
        'user_total_purchased', 'total_purchases', 'total_product_sales', 'total_units_sold',
        'ntb_purchased', 'ntb_product_sales', 'ntb_purchases', 'ntb_units_sold',
        'ntb_total_purchased', 'total_ntb_purchases', 'total_ntb_product_sales', 'total_ntb_units_sold'
    ]

    consideration_columns = [
        'Clicks', 'user_detail_page_view', 'detail_page_view', 'user_detail_page_view_clicks',
        'detail_page_view_clicks', 'user_detail_page_view_views', 'detail_page_view_views',
        'user_add_to_cart', 'add_to_cart', 'user_add_to_cart_clicks', 'add_to_cart_clicks',
        'user_add_to_cart_views', 'add_to_cart_views', 'user_total_detail_page_view',
        'total_detail_page_view', 'user_total_detail_page_view_clicks', 'total_detail_page_view_clicks',
        'user_total_detail_page_view_views', 'total_detail_page_view_views', 'user_total_add_to_cart',
        'total_add_to_cart', 'user_total_add_to_cart_clicks', 'total_add_to_cart_clicks',
        'user_total_add_to_cart_views', 'total_add_to_cart_views'
    ]

    existing_conv_cols = [col for col in conversion_columns if col in df.columns]
    existing_cons_cols = [col for col in consideration_columns if col in df.columns]

    return existing_conv_cols, existing_cons_cols

def display_quick_stats(df, key_prefix=""):
    """Display quick statistics section."""
    if len(df) > 0:
        st.subheader("Quick Statistics")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            selected_metric = st.selectbox(
                "Select metric for analysis",
                options=numeric_cols,
                index=min(10, len(numeric_cols)-1),
                key=f"{key_prefix}overview_metric"
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"**Average** {selected_metric}",
                         f"{df[selected_metric].mean():.2f}")
            with col2:
                st.metric(f"**Max** {selected_metric}",
                         f"{df[selected_metric].max():.2f}")
            with col3:
                st.metric(f"**Min** {selected_metric}",
                         f"{df[selected_metric].min():.2f}")

def display_filters(df, path_channels, analysis_level_filter, key_prefix=""):
    """Display and return filter controls."""
    filters = {}

    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        granularity_options = df['granularity'].dropna().unique()
        if len(granularity_options) > 0:
            default_index = list(granularity_options).index("Format") if "Format" in granularity_options else 0
            filters['granularity'] = st.selectbox(
                "Granularity",
                options=granularity_options,
                index=default_index,
                key=f"{key_prefix}granularity"
            )
        else:
            filters['granularity'] = None

    with col2:
        if analysis_level_filter == 'Data Workspace':
            analysis_level_options = df['analysis_level'].unique()
            filters['analysis_level'] = st.selectbox(
                "Analysis Level",
                options=analysis_level_options,
                index=0 if len(analysis_level_options) > 0 else 0,
                key=f"{key_prefix}analysis_level"
            )
        else:
            brand_options = df['brand'].dropna().unique()
            if len(brand_options) > 0:
                filters['brand'] = st.selectbox(
                    "Brand",
                    options=brand_options,
                    index=0,
                    key=f"{key_prefix}brand"
                )
            else:
                filters['brand'] = None

    with col3:
        filters['channel'] = st.multiselect(
            "Path Channel",
            options=path_channels,
            default=[],
            key=f"{key_prefix}channel"
        )

    return filters

def apply_filters(df, filters, analysis_level_filter=None):
    """Apply filters to dataframe."""
    filtered_df = df.copy()

    if analysis_level_filter:
        filtered_df = filtered_df[filtered_df['analysis_level'] == analysis_level_filter]
    elif 'analysis_level' in filters:
        if filters['analysis_level']:
            filtered_df = filtered_df[filtered_df['analysis_level'] == filters['analysis_level']]

    for filter_key in ['granularity', 'brand']:
        if filter_key in filters and filters[filter_key]:
            filtered_df = filtered_df[filtered_df[filter_key] == filters[filter_key]]

    if 'channel' in filters and filters['channel']:
        def contains_selected_channels(path):
            try:
                matches = re.findall(r'/([A-Z\s]+)', str(path).upper())
                return all(channel in matches for channel in filters['channel'])
            except:
                return False
        filtered_df = filtered_df[filtered_df['path'].apply(contains_selected_channels)]

    return filtered_df

def display_sorting_controls(filtered_df, key_prefix=""):
    """Display sorting and limit controls, return sorted and limited df."""
    col1, col2 = st.columns(2)

    with col1:
        sort_options = ['None'] + list(filtered_df.select_dtypes(include=[np.number]).columns)
        sort_by = st.selectbox("Sort by column (descending)", options=sort_options, key=f"{key_prefix}sort_by")

    with col2:
        max_rows = st.slider("Maximum rows to display", min_value=5, max_value=max(10, len(filtered_df)), value=min(50, len(filtered_df)), step=5, key=f"{key_prefix}max_rows")

    st.subheader(f"Filtered Data ({len(filtered_df)} rows)")

    # Apply sorting if requested
    if sort_by != 'None':
        ascending = False  # Always descending
        filtered_df_sorted = filtered_df.sort_values(by=sort_by, ascending=ascending)
    else:
        filtered_df_sorted = filtered_df

    # Limit to max_rows for display
    display_df = filtered_df_sorted.head(max_rows)

    return display_df

def display_styled_table(df, key_prefix=""):
    """Display the dataframe with column highlighting."""
    existing_conv_cols, existing_cons_cols = get_column_highlights(df)

    def get_colors(column):
        if column.name in existing_conv_cols:
            return ['background-color: #FFA500'] * len(column)  # Orange
        elif column.name in existing_cons_cols:
            return ['background-color: #C896C8'] * len(column)  # Light purple
        else:
            return [''] * len(column)  # No highlight

    styled_df = df.style.apply(get_colors, axis=0)
    column_config = {
        col: st.column_config.Column(width=110) for col in df.columns
    }
    st.dataframe(styled_df, column_config=column_config, hide_index=True)

def display_export_controls(df, key_prefix="", default_all=True):
    """Display export controls for data."""
    st.subheader("🔽 Data Export")
    col1, col2 = st.columns(2)

    with col1:
        selected_columns = st.multiselect(
            "Select columns to export",
            options=list(df.columns),
            default=list(df.columns) if default_all else list(df.columns[:11]),
            key=f"export_column_selection_{key_prefix}"
        )

    with col2:
        export_format = st.selectbox(
            "Export format",
            options=["CSV", "Excel"],
            index=0,
            key=f"export_format_{key_prefix}"
        )

    # Create export data with selected columns
    export_df = df[selected_columns] if selected_columns else pd.DataFrame()

    if not export_df.empty:
        if export_format == "CSV":
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download CSV ({len(export_df)} rows × {len(selected_columns)} columns)",
                data=csv_data,
                file_name=f"amc_data_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"download_csv_button_{key_prefix}"
            )
        else:  # Excel
            import io
            buffer = io.BytesIO()
            export_df.to_excel(buffer, index=False, engine='openpyxl')
            excel_data = buffer.getvalue()

            st.download_button(
                label=f"📥 Download Excel ({len(export_df)} rows × {len(selected_columns)} columns)",
                data=excel_data,
                file_name=f"amc_data_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_excel_button_{key_prefix}"
            )

def display_search_controls(filtered_df_sorted, key_prefix=""):
    """Display search controls and apply search if provided."""
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input(
            "🔍 Search in data",
            placeholder="Type to filter rows...",
            key=f"{key_prefix}search",
            help="Search across all text columns in the table"
        )
    with col2:
        search_button = st.button("Search", key=f"{key_prefix}search_button", help="Apply search filter")

    # Apply search filter
    if search_term and len(search_term.strip()) > 0:
        search_filtered_df = filtered_df_sorted.copy()
        text_columns = search_filtered_df.select_dtypes(include=['object', 'string']).columns
        mask = pd.Series(False, index=search_filtered_df.index)
        for col in text_columns:
            mask |= search_filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)
        search_filtered_df = search_filtered_df[mask]

        filtered_df_sorted = search_filtered_df
        st.info(f"🔍 Found {len(filtered_df_sorted)} rows matching '{search_term}'")

    return filtered_df_sorted

def display_channel_color_pickers(key_prefix=""):
    """Display channel color picker controls."""
    st.subheader("Channel Colors")
    num_channels = len(st.session_state.channel_colors)

    if num_channels > 0:
        # Create appropriate number of columns based on number of channels
        num_cols = max(1, min(4, num_channels))
        color_cols = st.columns(num_cols)

        for i, (channel, color) in enumerate(st.session_state.channel_colors.items()):
            col_idx = i % num_cols
            with color_cols[col_idx]:
                new_color = st.color_picker(f"{channel}", color, key=f"{key_prefix}color_{channel}")
                if new_color != color:
                    st.session_state.channel_colors[channel] = new_color
    else:
        st.info("💡 Upload data to see channel color options")

def apply_video_filter(df, has_video):
    """Apply video column filtering if needed."""
    df_filtered = df
    if not has_video:
        video_cols = [col for col in df.columns if 'video' in col.lower()]
        if video_cols:
            df_filtered = df.drop(columns=video_cols)

        if 'description' in df_filtered.columns and 'reach' in df_filtered.columns:
            cols_to_keep = []
            keep_mode = True

            for col in df_filtered.columns:
                if col == 'description':
                    cols_to_keep.append(col)
                    keep_mode = False
                elif col == 'reach':
                    keep_mode = True
                    cols_to_keep.append(col)
                elif keep_mode:
                    cols_to_keep.append(col)

            df_filtered = df_filtered[cols_to_keep]

    return df_filtered

# Configure page
st.set_page_config(page_title="AMC Analytics L'Oréal v3", page_icon="📊", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'channel_colors' not in st.session_state:
    st.session_state.channel_colors = {}
if 'previous_has_video' not in st.session_state:
    st.session_state.previous_has_video = True

# Color palette for channels
DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

def main():
    st.title("📊 AMC Analytics L'Oréal v3")

    # Tab layout
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📁 Data Upload", "📈 Campaign Summary", "📋 Data Workspace", "📊 Media Mix", "🔀 Path to Conversion", "📖 Documentation"])

    with tab1:
        data_upload_tab()

    with tab2:
        if st.session_state.data is not None:
            campaign_summary_tab()
        else:
            st.info("Please upload a CSV file first")

    with tab3:
        if st.session_state.data is not None:
            data_table_tab()
        else:
            st.info("Please upload a CSV file first")

    with tab4:
        if st.session_state.data is not None:
            media_mix_tab()
        else:
            st.info("Please upload a CSV file first")

    with tab5:
        if st.session_state.data is not None:
            path_to_conversion_tab()
        else:
            st.info("Please upload a CSV file first")

    with tab6:
        documentation_tab()

def data_upload_tab():
    st.header("Upload CSV File")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your AMC analytics CSV file. Maximum size: 50MB, CSV format only."
    )

    if uploaded_file is not None:
        try:
            # Validation: Check file size
            if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
                st.error("❌ File too large! Maximum file size is 50MB.")
                return

            # Validation: Check file extension
            if not uploaded_file.name.lower().endswith('.csv'):
                st.error("❌ Please upload a CSV file only.")
                return

            # Check if it's a new upload
            current_hash = hash(uploaded_file.getvalue())
            if 'file_hash' not in st.session_state or st.session_state.file_hash != current_hash:
                st.session_state.file_hash = current_hash
                st.toast("File uploaded successfully!", icon="✅")

            # Read CSV file (do this always for data processing)
            with st.spinner("Loading CSV file..."):
                try:
                    df = pd.read_csv(uploaded_file, low_memory=False)
                except pd.errors.EmptyDataError:
                    st.error("❌ The uploaded file is empty or contains no data.")
                    return
                except pd.errors.ParserError as e:
                    st.error(f"❌ Error parsing CSV file: {str(e)}")
                    return
                except Exception as e:
                    st.error(f"❌ Error reading file: {str(e)}")
                    return

            # Basic validation
            if df.empty:
                st.error("❌ The uploaded file contains no rows.")
                return

            if len(df.columns) < 3:
                st.warning("⚠️ Warning: The file has very few columns. This might not be AMC data.")

            # Data cleaning and preprocessing
            with st.spinner("Processing data..."):
                try:
                    progress_bar = st.progress(0, text="Preprocessing data...")
                    df = preprocess_data(df)
                    progress_bar.progress(100, text="Preprocessing complete!")
                    time.sleep(0.5)  # Brief pause to show completion
                    progress_bar.empty()
                except Exception as e:
                    st.error(f"❌ Error processing data: {str(e)}")
                    return

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

        except Exception as e:
            st.error(f"❌ Unexpected error during file upload: {str(e)}")
            return

        # Video information checkbox
        has_video = st.checkbox(
            "File contains video information",
            value=True,
            help="Check if your data includes video-related KPIs. Unchecking will filter them out."
        )

        # Basic statistics
        st.subheader("Data Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            date_min = df['start_date'].min() if 'start_date' in df.columns and df['start_date'].notna().any() else 'N/A'
            date_max = df['end_date'].max() if 'end_date' in df.columns and df['end_date'].notna().any() else 'N/A'
            st.metric("Date Range", f"{str(date_min).split('T')[0] if date_min != 'N/A' else 'N/A'} to {str(date_max).split('T')[0] if date_max != 'N/A' else 'N/A'}")

        # Channels list
        col3 = st.columns(1)[0]
        with col3:
            if 'channel' in df.columns:
                channels_list = sorted([str(c) for c in df['channel'].dropna().unique() if str(c).strip()])
                if channels_list:
                    channels_display = ", ".join(channels_list)
                    st.metric("Available Channels", channels_display)
                else:
                    st.write("No channels found in the data")
            else:
                st.write("Channels not available")

        # Check for changes in video checkbox
        if has_video != st.session_state.previous_has_video:
            if has_video:
                st.toast("Video-related columns included", icon="ℹ️")
            else:
                st.toast("Video-related columns removed", icon="ℹ️")
            st.session_state.previous_has_video = has_video

        # Filter columns based on video option
        df_filtered = df
        if not has_video:
            # Remove video-related columns
            video_cols = [col for col in df.columns if 'video' in col.lower()]
            if video_cols:
                df_filtered = df.drop(columns=video_cols)

            # Remove ALL KPI columns between 'description' and 'reach' (inclusive)
            # Keep everything up to 'description', then start from 'reach' onwards
            if 'description' in df_filtered.columns and 'reach' in df_filtered.columns:
                cols_to_keep = []
                keep_mode = True  # Start in keep mode (before description)

                for col in df_filtered.columns:
                    if col == 'description':
                        cols_to_keep.append(col)  # Keep description
                        keep_mode = False  # Switch to remove mode after description
                    elif col == 'reach':
                        keep_mode = True  # Switch back to keep mode from reach
                        cols_to_keep.append(col)  # Keep reach
                    elif keep_mode:
                        cols_to_keep.append(col)  # Keep columns in keep mode

                df_filtered = df_filtered[cols_to_keep]

        st.session_state.data = df_filtered

        st.subheader("Data Preview")
        st.dataframe(df_filtered.head(), width='stretch')

@st.cache_data
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

    # Calculate KPIs according to documentation definitions

    # Awareness KPIs
    if 'reach' in df.columns:
        df['REACH'] = df['reach']

    if 'impressions' in df.columns:
        df['IMPRESSIONS'] = df['impressions']
        if 'reach' in df.columns:
            df['AVG FREQ'] = df['impressions'] / df['reach'].replace(0, np.nan)

    # Engagement KPIs
    if 'clicks' in df.columns:
        df['CLICKS'] = df['clicks']
        if 'impressions' in df.columns:
            df['CTR'] = df['clicks'] / df['impressions'].replace(0, np.nan)

    if 'impressions_cost' in df.columns:
        df['COST AMC'] = df['impressions_cost']
        if 'impressions' in df.columns:
            df['CPM'] = df['impressions_cost'] / (df['impressions'] / 1000).replace(0, np.nan)

    if 'details_page_view_clicks' in df.columns:
        df['DPV'] = df['details_page_view_clicks']
        if 'impressions_cost' in df.columns:
            df['CPDPV'] = df['impressions_cost'] / df['details_page_view_clicks'].replace(0, np.nan)

    if 'add_to_cart' in df.columns:
        df['ADD TO CART'] = df['add_to_cart']

    # Purchase KPIs
    if 'purchases' in df.columns:
        df['PURCHASES'] = df['purchases']

    if 'user_purchased' in df.columns:
        df['NBR ACHETEURS'] = df['user_purchased']

    if 'product_sales' in df.columns:
        df['REVENUE'] = df['product_sales']

    if 'purchases' in df.columns and 'details_page_view_clicks' in df.columns:
        df['CVR'] = df['purchases'] / df['details_page_view_clicks'].replace(0, np.nan)

    if 'product_sales' in df.columns and 'impressions_cost' in df.columns:
        df['ROAS'] = df['product_sales'] / df['impressions_cost'].replace(0, np.nan)

    # NTB KPIs
    if 'ntb_purchases' in df.columns:
        df['NTB'] = df['ntb_purchases']

    if 'ntb_product_sales' in df.columns:
        df['REVENUE NTB'] = df['ntb_product_sales']

    if 'ntb_purchases' in df.columns and 'impressions_cost' in df.columns:
        df['COST PER NTB'] = df['impressions_cost'] / df['ntb_purchases'].replace(0, np.nan)

    if 'ntb_purchases' in df.columns and 'details_page_view_clicks' in df.columns:
        df['CVR NTB'] = df['ntb_purchases'] / df['details_page_view_clicks'].replace(0, np.nan)

    if 'ntb_product_sales' in df.columns and 'impressions_cost' in df.columns:
        df['ROAS NTB'] = df['ntb_product_sales'] / df['impressions_cost'].replace(0, np.nan)

    if 'ntb_purchases' in df.columns and 'purchases' in df.columns:
        df['% NTB'] = df['ntb_purchases'] / df['purchases'].replace(0, np.nan)

    # Keep natural column order (KPIs at the end as they are created)

    return df

def campaign_summary_tab():
    st.header("Campaign Summary")

    df = st.session_state.data

    # Overall metrics
    col1, col2, col3, col4 = st.columns(4)

    total_spend = df['impressions_cost'].sum()
    total_purchases = df['purchases'].sum()
    total_sales = df['product_sales'].sum()

    with col1:
        st.metric("Total Spend", f"{total_spend:,.0f} €")
    with col2:
        st.metric("Total Purchases", f"{total_purchases:,.0f}")
    with col3:
        st.metric("Total Sales", f"{total_sales:,.0f} €")
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

        st.dataframe(channel_stats, width='stretch')

        # Quick chart
        fig = go.Figure(data=[
            go.Bar(name='Spend', x=channel_stats['channel'], y=channel_stats['impressions_cost']),
            go.Bar(name='Sales', x=channel_stats['channel'], y=channel_stats['product_sales'])
        ])
        fig.update_layout(barmode='group', title="Channel Spend vs Sales")
        st.plotly_chart(fig, config={'responsive': True})
    else:
        st.info("Upload data to see campaign summary")

def data_table_tab():
    st.header("Data Table & Filters")

    df = st.session_state.data.copy()

    # Show quick stats for all data
    display_quick_stats(df, "data_workspace")

    # Extract path channels for filter
    path_channels = extract_path_channels(df)

    # Filters - Data Workspace has analysis_level instead of fixed level
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        analysis_level_options = df['analysis_level'].unique()
        analysis_level_filter = st.selectbox(
            "Analysis Level",
            options=analysis_level_options,
            index=0 if len(analysis_level_options) > 0 else 0,
            key="data_workspace_analysis_level"
        )

    with col2:
        granularity_options = df['granularity'].dropna().unique()
        granularity_filter = st.selectbox(
            "Granularity",
            options=granularity_options,
            index=0 if len(granularity_options) > 0 else 0,
            key="data_workspace_granularity"
        )

    with col3:
        brand_options = df['brand'].dropna().unique()
        if len(brand_options) > 0:
            brand_filter = st.selectbox(
                "Brand",
                options=brand_options,
                index=0,
                key="data_workspace_brand"
            )
        else:
            brand_filter = None

    # Channel filter
    channel_filter = st.multiselect(
        "Path Channel",
        options=path_channels,
        default=[],
        key="data_workspace_channel"
    )

    # Apply filters
    filters = {
        'analysis_level': analysis_level_filter,
        'granularity': granularity_filter,
        'brand': brand_filter,
        'channel': channel_filter
    }
    filtered_df = apply_filters(df, filters)

    # Sorting and display controls
    display_df = display_sorting_controls(filtered_df, "data_workspace")

    # Add search functionality (unique to data workspace)
    filtered_df_search = display_search_controls(display_df, "data_workspace")
    display_df = filtered_df_search.head(len(display_df))  # Maintain max_rows limit

    # Display table
    display_styled_table(display_df, "data_workspace")

    # Export controls with all columns by default
    display_export_controls(filtered_df, "data_workspace", default_all=True)

    # Channel color pickers
    display_channel_color_pickers("data_workspace")


def media_mix_tab():
    st.header("Media Mix")

    df = st.session_state.data.copy()

    # Extract path channels
    path_channels = extract_path_channels(df)

    # Fixed analysis level
    analysis_level_filter = 'Media Mix'

    # Show quick stats for filtered data
    filtered_df_for_stats = df[df['analysis_level'] == analysis_level_filter]
    display_quick_stats(filtered_df_for_stats, "media_mix")

    # Get filters
    filters = display_filters(df, path_channels, analysis_level_filter, "media_mix")

    # Apply filters
    filtered_df = apply_filters(df, filters, analysis_level_filter)

    # Sorting and display controls
    display_df = display_sorting_controls(filtered_df, "media_mix")

    # Display table
    display_styled_table(display_df, "media_mix")

    # Export controls
    display_export_controls(display_df, "media_mix", default_all=False)

    # Generate plots button
    generate_plots = st.button("📊 Generate Charts for Each Row", key="generate_plots_media_mix")

    # Channel color pickers
    display_channel_color_pickers("media_mix")

    # Display plots if generated
    if generate_plots:
        st.subheader("📊 Charts for Each Row")
        st.info(f"Displaying charts for the {len(display_df)} rows shown above")

        for idx, row in display_df.iterrows():
            path_value = str(row.get('path', f'Row {idx}'))
            path_clean = path_value.replace('[', '').replace(']', '').strip()

            with st.expander(f"📈 Chart for Path: {path_clean}", expanded=False):
                try:
                    matches = re.findall(r'/([A-Z\s]+)', str(row['path']).upper())
                    unique_channels = list(dict.fromkeys(matches))
                    fig = create_venn_diagram(unique_channels, st.session_state.channel_colors)
                except Exception as e:
                    fig = go.Figure(data=[go.Bar(x=['DEBUG ERROR'], y=[1])])
                    fig.update_layout(title=f"Error: {str(e)[:100]}", height=300)

                st.plotly_chart(fig, config={'responsive': True}, key=f"media_mix_chart_{idx}")


def path_to_conversion_tab():
    st.header("Path to Conversion")

    df = st.session_state.data.copy()

    # Extract path channels
    path_channels = extract_path_channels(df)

    # Fixed analysis level
    analysis_level_filter = 'Path to conversion'

    # Show quick stats for filtered data
    filtered_df_for_stats = df[df['analysis_level'] == analysis_level_filter]
    display_quick_stats(filtered_df_for_stats, "path_to_conversion")

    # Get filters
    filters = display_filters(df, path_channels, analysis_level_filter, "path_to_conversion")

    # Apply filters
    filtered_df = apply_filters(df, filters, analysis_level_filter)

    # Sorting and display controls
    display_df = display_sorting_controls(filtered_df, "path_to_conversion")

    # Display table
    display_styled_table(display_df, "path_to_conversion")

    # Export controls
    display_export_controls(display_df, "path_to_conversion", default_all=False)

    # Generate plots button
    generate_plots = st.button("📊 Generate Charts for Each Row", key="generate_plots_path_to_conversion")

    # Channel color pickers
    display_channel_color_pickers("path_to_conversion")

    # Display plots if generated
    if generate_plots:
        st.subheader("📊 Charts for Each Row")
        st.info(f"Displaying charts for the {len(display_df)} rows shown above")

        for idx, row in display_df.iterrows():
            path_value = str(row.get('path', f'Row {idx}'))
            path_clean = path_value.replace('[', '').replace(']', '').strip()

            with st.expander(f"📈 Chart for Path: {path_clean}", expanded=False):
                try:
                    matches = re.findall(r'/([A-Z\s]+)', str(row['path']).upper())
                    if matches:
                        fig = create_simple_path_diagram(matches, st.session_state.channel_colors)
                    else:
                        fig = go.Figure(data=[go.Bar(x=['No path data'], y=[1])])
                        fig.update_layout(title="No conversion path found", height=300)
                except:
                    fig = go.Figure(data=[go.Bar(x=['Error'], y=[1])])
                    fig.update_layout(title="Error creating chevron diagram", height=300)

                st.plotly_chart(fig, config={'responsive': True}, key=f"path_to_conversion_chart_{idx}")


def create_simple_path_diagram(channels, color_dict):
    """
    Create a simple flow diagram with HUGE squares for each channel connected by arrows.

    FONCTIONNEMENT DÉTAILLÉ :
    =========================
    Cette fonction crée un diagramme de flux horizontal montrant le chemin de conversion :
    CANAL1 → CANAL2 → CANAL3 → ... CANAL_N

    ÉTAPES PRINCIPALES :
    ====================
    1. INITIALISATION : fig = go.Figure() (canevas plotly vide)

    2. CALCUL DU NOMBRE DE CANAUX : n_channels = len(channels)
       - Si 0 canaux : retourne un graphe vide avec message d'erreur

    3. CRÉATION DES CARRÉS ENORMES (4x plus grands) :
       - Pour chaque canal i dans channels :
       - Position x_start = i * 8.0  # Espacement énorme entre carrés
       - Dimensions du carré : largeur=6.0, hauteur=6.0 (au lieu d'1.5x1.5)
       - Couleur récupérée depuis color_dict[channel] ou palette par défaut

    4. DESSINER LE CARRÉ :
       - Utilise go.Scatter avec mode='lines' et fill='toself'
       - Définit les 4 coins du carré + couleur de fond
       - Aucun hover, pas de légende pour rester épuré

    5. AJOUTER LE TEXTE DU CANAL :
       - Annotation annotée au centre du carré : x_start + 3.0, y=0
       - Texte blanc en gras pour contraste avec couleur de fond

    6. AJOUTER LES FLÈCHES ENTRE CARRÉS :
       - Si n_channels > 1 : créer des flèches "→" entre chaque paire
       - Position : (i+1) * 8.0 - 3.5 (centré entre deux carrés)
       - Taille de police 48 (énorme pour matcher les carrés géants)

    7. RÉGLER LES DIMENSIONS DU GRAPHE :
       - Largeur calculée = n_channels * 8.0 + marges (énorme pour tout contenir)
       - Hauteur fixe = 400 (agrandie pour voir les carrés géants)
       - Axes : rangées=[-8, 8] pour cadres les carrés qui vont de y=-3 à y=3

    8. MASQUER LES AXES :
       - showticklabels=False, showgrid=False, visible=False
       - Conserver scaleanchor="x" pour aspect ratio carré si besoin

    RETOUR : fig complète prête à être affichée par st.plotly_chart()
    ==================================================================
    """
    fig = go.Figure()

    n_channels = len(channels)

    if n_channels == 0:
        fig.update_layout(title="No channels found", height=300)
        return fig

    # Create HUGE squares for each channel (4x larger !!!)
    for i, channel in enumerate(channels):
        x_start = i * 8.0  # Massive spacing between squares

        # Create GIGANTIC square shape (6x6 instead of 1.5x1.5)
        # Scaling factor: x4 width and height
        square_x = [x_start, x_start + 6.0, x_start + 6.0, x_start]
        square_y = [-3.0, -3.0, 3.0, 3.0]  # Height doubled accordingly

        color = color_dict.get(channel, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])

        # Add filled gigantic square
        fig.add_trace(go.Scatter(
            x=square_x + [square_x[0]],  # Close the square (5 points needed for closed shape)
            y=square_y + [square_y[0]],
            fill="toself",              # Fill the polygon formed by the points
            fillcolor=color,            # Use channel-specific color
            mode='lines',               # Draw only the outline, but 'fill' fills it
            line=dict(color=color, width=5), # Thicker borders for HUGE squares
            hoverinfo='skip',           # No hover tooltips (clean)
            showlegend=False            # No legend needed (single unlabeled shape)
        ))

        # Add channel text in center of GIGANTIC square
        fig.add_annotation(
            x=x_start + 3.0,  # Center of 6.0-wide square
            y=0,               # Middle vertically (within -3 to +3 range)
            text=channel,      # Channel name text
            showarrow=False,   # No arrow from annotation to point
            font=dict(size=20, color='white', weight='bold'), # Larger font for HUGE squares
            xanchor='center',  # Center text horizontally on x position
            yanchor='middle'   # Center text vertically on y position
        )

    # Add MASSIVE arrows between GIGANTIC squares if more than one channel
    if n_channels > 1:
        for i in range(n_channels - 1):
            # Position arrow exactly BETWEEN squares: midway between right edge of square i and left edge of square i+1
            # Square i ends at: i * 8.0 + 6.0
            # Square i+1 starts at: (i+1) * 8.0
            # Perfect mid-point: 8*i + 7.0 (thus offset: (i+1)*8.0 - 1.0)
            arrow_x = (i + 1) * 8.0 - 1.0  # Perfect centering between MASSIVE squares
            fig.add_annotation(
                x=arrow_x,
                y=0,  # Same y position as square centers
                text="→",  # Unicode right arrow
                showarrow=False,
                font=dict(size=48, color='black'),  # MASSIVE font size to match HUGE squares
                xanchor='center',
                yanchor='middle'
            )

    # Set MASSIVE axis ranges to fit all GIGANTIC squares and arrows
    # Axis ranges need to accommodate squares from y=-3 to y=3
    # x range: from 0 to (n_channels-1)*8.0 + 6.0 (last square goes to its right edge)
    x_axis_max = (n_channels - 1) * 8.0 + 6.0 + 4.0  # +4 additional margin
    x_axis_min = -4.0  # Left margin

    fig.update_layout(
        height=600,  # Triple height to show HUGE squares properly
        width=max(800, int(x_axis_max * 80)),  # Width calculated for all HUGE squares + arrows
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Configure axes - MASSIVE ranges to fit everything
    fig.update_xaxes(
        range=[x_axis_min, x_axis_max],  # Massive x range to fit all HUGE squares
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False
    )

    # Special y-axis range for 3 channels to make squares appear BIGGER
    # For 3 channels, we use [-2.5, 2.5] instead of [-8,8] to zoom in visually
    # This makes the 6x6 squares (which are actually [-3,+3]) appear much larger on screen
    if n_channels == 3:
        y_axis_range = [-3.5, 3.5]  # Zoom in for 3 channels - makes squares WAY bigger
    else:
        y_axis_range = [-4, 4]  # Normal zoom for 1,2,4+ channels

    fig.update_yaxes(
        range=y_axis_range,  # Adaptive y range: zoomed in for 3 channels, normal for others
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False,
        scaleanchor="x",  # Maintain square aspect ratio
        scaleratio=1
    )

    return fig  # Return complete plotly figure


def create_venn_diagram(channels, color_dict):
    """
    Create a Venn diagram visualization for channels using overlapping circles
    """
    fig = go.Figure()

    # Number of channels
    n_channels = len(channels)

    if n_channels == 0:
        fig.update_layout(title="No channels found", height=300)
        return fig

    # Position circles in a pattern that keeps them circular
    if n_channels == 1:
        # Single large circle
        centers = [[0, 0]]
        radius = 1.5
        axis_range = [-3, 3]
    elif n_channels == 2:
        # Two side-by-side circles, non-overlapping
        centers = [[-0.8, 0], [0.8, 0]]
        radius = 1.0
        axis_range = [-2.5, 2.5]
    elif n_channels == 3:
        # Triangle arrangement
        centers = [[0, 1], [-0.85, -0.45], [0.85, -0.45]]
        radius = 1.0
        axis_range = [-3, 3]
    elif n_channels == 4:
        # Square arrangement
        centers = [[0, 0.8], [-0.75, 0], [0.75, 0], [0, -0.8]]
        radius = 0.8
        axis_range = [-3, 3]
    else:
        # For 5+ channels, circular arrangement
        centers = []
        radius = 0.7 if n_channels > 6 else 0.8
        angle_step = 2 * 3.14159 / n_channels
        for i in range(n_channels):
            angle = i * angle_step
            x = 1.3 * np.cos(angle)
            y = 1.3 * np.sin(angle)
            centers.append([x, y])
        axis_range = [-2.5, 2.5]

    # Create circles for all channels
    colors = [color_dict.get(ch, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]) for i, ch in enumerate(channels)]

    for center, color in zip(centers, colors):
        circle = create_circle_shape(center, radius, color, 0.7)
        fig.add_shape(circle)

    # Add text inside each circle (not in center)
    if n_channels == 1:
        # Single circle - text in center
        fig.add_annotation(
            x=0, y=0,
            text=channels[0],
            showarrow=False,
            font=dict(size=16, color='white', weight='bold'),
            xanchor='center',
            yanchor='middle'
        )
    else:
        # Multiple circles - text inside each circle
        for i, (center, channel) in enumerate(zip(centers, channels)):
            fig.add_annotation(
                x=center[0], y=center[1],
                text=channel,
                showarrow=False,
                font=dict(size=12, color='white', weight='bold'),
                xanchor='center',
                yanchor='middle'
            )

    # Update axes - no grid, no labels, no ticks
    fig.update_xaxes(
        range=axis_range,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False
    )
    fig.update_yaxes(
        range=axis_range,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False
    )

    fig.update_layout(
        height=900,
        width=900,  # Square aspect ratio to ensure circular shapes
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'  # Transparent background
    )

    # Force equal aspect ratio to keep circles perfectly round
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig

def create_circle_shape(center, radius, color, opacity):
    """
    Create a circle shape for Venn diagram
    """
    return dict(
        type="circle",
        xref="x", yref="y",
        x0=center[0] - radius, y0=center[1] - radius,
        x1=center[0] + radius, y1=center[1] + radius,
        fillcolor=color,
        opacity=opacity,
        line=dict(width=2, color=color)
    )

def documentation_tab():
    st.subheader("Key Performance Indicators (KPIs)")

    # Awareness KPIs
    st.markdown("### Awareness")
    awareness_kpis = {
        "REACH": "Column: 'reach'",
        "IMPRESSIONS": "Column: 'impressions'",
        "AVG FREQ": "Formula: impressions / reach"
    }

    for kpi, definition in awareness_kpis.items():
        with st.expander(f"**{kpi}**"):
            st.write(definition)

    # Engagement KPIs
    st.markdown("### Engagement")
    engagement_kpis = {
        "CLICKS": "Column: 'clicks'",
        "CTR": "Formula: clicks / impressions",
        "CPM": "Formula: impression_cost / (impressions/1000)",
        "COST AMC": "Column: 'impression_cost'",
        "DPV": "Column: 'details_page_view_clicks'",
        "CPDPV": "Formula: COST AMC / DPV",
        "ADD TO CART": "Column: 'add_to_cart'"
    }

    for kpi, definition in engagement_kpis.items():
        with st.expander(f"**{kpi}**"):
            st.write(definition)

    # Purchase KPIs
    st.markdown("### Purchase")
    purchase_kpis = {
        "PURCHASES": "Column: 'purchases'",
        "NBR ACHETEURS": "Column: 'user_purchased'",
        "REVENUE": "Column: 'product_sales'",
        "CVR": "Formula: PURCHASES / DPV",
        "ROAS": "Formula: revenue / cost"
    }

    for kpi, definition in purchase_kpis.items():
        with st.expander(f"**{kpi}**"):
            st.write(definition)

    # NTB KPIs
    st.markdown("### NTB (New To Brand)")
    ntb_kpis = {
        "NTB": "Column: 'ntb_purchases'",
        "REVENUE NTB": "Column: 'ntb_product_sales'",
        "COST PER NTB": "Formula: cost / NTB",
        "CVR NTB": "Formula: NTB / DPV",
        "ROAS NTB": "Formula: revenu NTB / COST AMC",
        "% NTB": "Formula: NTB / PURCHASES"
    }

    for kpi, definition in ntb_kpis.items():
        with st.expander(f"**{kpi}**"):
            st.write(definition)

if __name__ == "__main__":
    main()
