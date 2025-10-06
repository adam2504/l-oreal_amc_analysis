import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
import re
from collections import defaultdict, Counter
import numpy as np

# Configure page
st.set_page_config(page_title="AMC Analytics L'Oréal v2", page_icon="📊", layout="wide")

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
    st.title("📊 AMC Analytics L'Oréal v2")

    # Tab layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 Data Upload", "📋 Data Workspace", "📊 Media Mix", "🔀 Path to Conversion", "📖 Documentation"])

    with tab1:
        data_upload_tab()

    with tab2:
        if st.session_state.data is not None:
            data_table_tab()
        else:
            st.info("Please upload a CSV file first")

    with tab3:
        if st.session_state.data is not None:
            media_mix_tab()
        else:
            st.info("Please upload a CSV file first")

    with tab4:
        if st.session_state.data is not None:
            path_to_conversion_tab()
        else:
            st.info("Please upload a CSV file first")

    with tab5:
        documentation_tab()

def data_upload_tab():
    st.header("Upload CSV File")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Check if it's a new upload
        current_hash = hash(uploaded_file.getvalue())
        if 'file_hash' not in st.session_state or st.session_state.file_hash != current_hash:
            st.session_state.file_hash = current_hash
            st.toast("File uploaded successfully!", icon="✅")

        # Read CSV file (do this always for data processing)
        df = pd.read_csv(uploaded_file, low_memory=False)

        # Data cleaning and preprocessing
        df = preprocess_data(df)

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

        # Video information checkbox
        has_video = st.checkbox("File contains video information", value=True)

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

        st.subheader("Campaign Summary")

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
        df['ROAS'] = df['product_sales'] / df['impressions_cost'].replace(0, np.nan)

    if 'impressions_cost' in df.columns and 'detail_page_view' in df.columns:
        df['CPDPV'] = df['impressions_cost'] / df['detail_page_view'].replace(0, np.nan)

    if 'purchases' in df.columns and 'impressions_cost' in df.columns:
        df['CPA'] = df['impressions_cost'] / df['purchases'].replace(0, np.nan)

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
        channel_filter = st.multiselect(
            "Path Channel",
            options=path_channels,
            default=[],
            key="data_workspace_channel"  # Start with no channels selected
        )

    # Apply filters
    filtered_df = df.copy()
    if analysis_level_filter:
        filtered_df = filtered_df[filtered_df['analysis_level'] == analysis_level_filter]
    if granularity_filter:
        filtered_df = filtered_df[filtered_df['granularity'] == granularity_filter]
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
            new_color = st.color_picker(f"{channel}", color, key=f"data_workspace_color_{channel}")
            if new_color != color:
                st.session_state.channel_colors[channel] = new_color

    # Add sorting options
    col1, col2, col3 = st.columns(3)

    with col1:
        sort_options = ['None'] + list(filtered_df.select_dtypes(include=[np.number]).columns)
        sort_by = st.selectbox("Sort by column", options=sort_options, key="data_workspace_sort_by")

    with col2:
        if sort_by != 'None':
            sort_order = st.selectbox("Sort order", options=["Descending", "Ascending"], key="data_workspace_sort_order")
        else:
            sort_order = "Descending"  # default

    with col3:
        max_rows = st.slider("Maximum rows to display", min_value=5, max_value=max(10, len(filtered_df)), value=min(50, len(filtered_df)), step=5, key="data_workspace_max_rows")

    st.subheader(f"Filtered Data ({len(filtered_df)} rows)")

    # Column selection for export/download
    st.subheader("🔽 Data Export")
    col1, col2 = st.columns(2)

    with col1:
        selected_columns = st.multiselect(
            "Select columns to export",
            options=list(filtered_df.columns),
            default=list(filtered_df.columns),  # All columns selected by default
            key="export_column_selection"
        )

    with col2:
        export_format = st.selectbox(
            "Export format",
            options=["CSV", "Excel"],
            index=0,
            key="export_format"
        )

    # Create export data with selected columns
    export_df = filtered_df[selected_columns] if selected_columns else pd.DataFrame()

    if not export_df.empty:
        if export_format == "CSV":
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download CSV ({len(export_df)} rows × {len(selected_columns)} columns)",
                data=csv_data,
                file_name=f"amc_data_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_csv_button"
            )
        else:  # Excel
            # For Excel export, we would need to add openpyxl to requirements
            csv_data = export_df.to_csv(index=False)  # Temporary, basic CSV for now
            st.download_button(
                label=f"📥 Download Excel ({len(export_df)} rows × {len(selected_columns)} columns)",
                data=csv_data,
                file_name=f"amc_data_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel_button",
                help="Note: Currently exports as CSV with .xlsx extension. For true Excel export, add 'openpyxl' to requirements."
            )

    st.divider()  # Visual separator

    # Apply sorting if requested
    if sort_by != 'None':
        ascending = (sort_order == "Ascending")
        filtered_df_sorted = filtered_df.sort_values(by=sort_by, ascending=ascending)
    else:
        filtered_df_sorted = filtered_df

    # Limit to max_rows for display and plots
    display_df = filtered_df_sorted.head(max_rows)

    generate_plots = st.button("📊 Generate Charts for Each Row", key="generate_plots")

    # Display the table first (always, whether plots are generated or not)
    # Highlight columns logic
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

    # Check which KPI columns exist
    existing_conv_cols = [col for col in conversion_columns if col in filtered_df.columns]
    existing_cons_cols = [col for col in consideration_columns if col in filtered_df.columns]

    # Display table with styling
    # Define color function based on column category
    def get_colors(column):
        if column.name in existing_conv_cols:
            return ['background-color: #FFA500'] * len(column)  # Orange
        elif column.name in existing_cons_cols:
            return ['background-color: #C896C8'] * len(column)  # Light purple
        else:
            return [''] * len(column)  # No highlight

    # Apply pandas styling to highlight columns
    styled_df = display_df.style.apply(get_colors, axis=0)
    # Column configuration for constrained width (max ~20 characters)
    column_config = {
        col: st.column_config.Column(width=110) for col in display_df.columns
    }
    st.dataframe(styled_df, column_config=column_config)

    # Display plots separately if generated
    if generate_plots:
        st.subheader("📊 Charts for Each Row")
        st.info(f"Displaying charts for the {len(display_df)} rows shown above")

        # Create and display plots for each row
        for idx, row in display_df.iterrows():
            # Get the path from the row
            path_value = str(row.get('path', f'Row {idx}'))
            path_clean = path_value.replace('[', '').replace(']', '').strip()

            with st.expander(f"📈 Chart for Path: {path_clean}", expanded=False):
                # Create diagrams based on analysis level
                if analysis_level_filter == 'Media Mix':
                    # Extract channels from path
                    try:
                        matches = re.findall(r'/([A-Z\s]+)', str(row['path']).upper())
                        unique_channels = list(dict.fromkeys(matches))  # Remove duplicates while preserving order

                        fig = create_venn_diagram(unique_channels, st.session_state.channel_colors)
                    except Exception as e:
                        # Fallback in case of error - show exact error
                        fig = go.Figure(data=[go.Bar(x=['DEBUG ERROR'], y=[1])])
                        fig.update_layout(title=f"Error: {str(e)[:100]}", height=300)

                elif analysis_level_filter == 'Path to conversion':
                    # Extract steps from conversion path
                    try:
                        matches = re.findall(r'/([A-Z\s]+)', str(row['path']).upper())
                        if matches:
                            fig = create_simple_path_diagram(matches, st.session_state.channel_colors)
                        else:
                            fig = go.Figure(data=[go.Bar(x=['No path data'], y=[1])])
                            fig.update_layout(title="No conversion path found", height=300)
                    except:
                        # Fallback in case of error
                        fig = go.Figure(data=[go.Bar(x=['Error'], y=[1])])
                        fig.update_layout(title="Error creating path diagram", height=300)

                else:
                    # For other analysis levels: dummy test data
                    dummy_data = np.random.randint(10, 100, size=5)
                    plot_labels = ['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4', 'Metric 5']

                    fig = go.Figure(data=[go.Bar(x=plot_labels, y=dummy_data)])
                    fig.update_layout(
                        title=f"Path: {path_clean} - KPIs",
                        xaxis_title="Metrics",
                        yaxis_title="Values",
                        width=600,
                        height=300,
                        showlegend=False
                    )

                st.plotly_chart(fig, config={'responsive': True}, key=f"data_workspace_chart_{idx}")

    # Statistics
    if len(filtered_df) > 0:
        st.subheader("Quick Statistics")

        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            selected_metric = st.selectbox(
                "Select metric for analysis",
                options=numeric_cols,
                index=min(10, len(numeric_cols)-1),  # Default to a metric column
                key="data_workspace_metric"
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


def media_mix_tab():
    st.header("Media Mix")

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

    # Fixed analysis level
    analysis_level_filter = 'Media Mix'

    # Statistics (moved up)
    if len(df[df['analysis_level'] == analysis_level_filter]) > 0:
        filtered_df_for_stats = df[df['analysis_level'] == analysis_level_filter]
        st.subheader("Quick Statistics")

        numeric_cols = filtered_df_for_stats.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            selected_metric = st.selectbox(
                "Select metric for analysis",
                options=numeric_cols,
                index=min(10, len(numeric_cols)-1),  # Default to a metric column
                key="media_mix_overview_metric"
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Average {selected_metric}",
                         f"{filtered_df_for_stats[selected_metric].mean():.2f}")
            with col2:
                st.metric(f"Max {selected_metric}",
                         f"{filtered_df_for_stats[selected_metric].max():.2f}")
            with col3:
                st.metric(f"Min {selected_metric}",
                         f"{filtered_df_for_stats[selected_metric].min():.2f}")

    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        granularity_options = df['granularity'].dropna().unique()
        if len(granularity_options) > 0:
            # Default to "format" if available, otherwise first option
            if "Format" in granularity_options:
                default_index = list(granularity_options).index("Format")
            else:
                default_index = 0
            granularity_filter = st.selectbox(
                "Granularity",
                options=granularity_options,
                index=default_index,
                key="media_mix_granularity"
            )
        else:
            granularity_filter = None

    with col2:
        brand_options = df['brand'].dropna().unique()
        if len(brand_options) > 0:
            brand_filter = st.selectbox(
                "Brand",
                options=brand_options,
                index=0,
                key="media_mix_brand"
            )
        else:
            brand_filter = None

    with col3:
        channel_filter = st.multiselect(
            "Path Channel",
            options=path_channels,
            default=[],
            key="media_mix_channel"  # Start with no channels selected
        )

    # Apply filters
    filtered_df = df.copy()
    if analysis_level_filter:
        filtered_df = filtered_df[filtered_df['analysis_level'] == analysis_level_filter]
    if granularity_filter:
        filtered_df = filtered_df[filtered_df['granularity'] == granularity_filter]
    if brand_filter:
        filtered_df = filtered_df[filtered_df['brand'] == brand_filter]
    if channel_filter:
        def contains_selected_channels(path):
            try:
                matches = re.findall(r'/([A-Z\s]+)', str(path).upper())
                return all(channel in matches for channel in channel_filter)
            except:
                return False
        filtered_df = filtered_df[filtered_df['path'].apply(contains_selected_channels)]

    # Add sorting options and max rows
    col1, col2 = st.columns(2)

    with col1:
        sort_options = ['None'] + list(filtered_df.select_dtypes(include=[np.number]).columns)
        sort_by = st.selectbox("Sort by column (descending)", options=sort_options, key="media_mix_sort_by")

    with col2:
        if sort_by != 'None':
            # Always sort in descending order when a column is selected
            sort_order = "Descending"
        else:
            sort_order = "Descending"  # default

        max_rows = st.slider("Maximum rows to display", min_value=5, max_value=max(10, len(filtered_df)), value=min(50, len(filtered_df)), step=5, key="media_mix_max_rows")



    st.subheader(f"Filtered Data ({len(filtered_df)} rows)")

    # Apply sorting if requested
    if sort_by != 'None':
        ascending = (sort_order == "Ascending")
        filtered_df_sorted = filtered_df.sort_values(by=sort_by, ascending=ascending)
    else:
        filtered_df_sorted = filtered_df

    # Limit to max_rows for display and plots
    display_df = filtered_df_sorted.head(max_rows)

    # Display the table first (always, whether plots are generated or not)
    # Highlight columns logic
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

    # Check which KPI columns exist
    existing_conv_cols = [col for col in conversion_columns if col in filtered_df.columns]
    existing_cons_cols = [col for col in consideration_columns if col in filtered_df.columns]

    # Display table with styling
    # Define color function based on column category
    def get_colors(column):
        if column.name in existing_conv_cols:
            return ['background-color: #FFA500'] * len(column)  # Orange
        elif column.name in existing_cons_cols:
            return ['background-color: #C896C8'] * len(column)  # Light purple
        else:
            return [''] * len(column)  # No highlight

    # Apply pandas styling to highlight columns
    styled_df = display_df.style.apply(get_colors, axis=0)
    # Column configuration for constrained width (max ~20 characters)
    column_config = {
        col: st.column_config.Column(width=110) for col in display_df.columns
    }
    st.dataframe(styled_df, column_config=column_config)

    # Column selection for export/download
    st.subheader("🔽 Data Export")
    col1, col2 = st.columns(2)

    with col1:
        selected_columns = st.multiselect(
            "Select columns to export",
            options=list(display_df.columns),
            default=list(display_df.columns[:11]),  # All columns selected by default
            key="export_column_selection_media_mix"
        )

    with col2:
        export_format = st.selectbox(
            "Export format",
            options=["CSV", "Excel"],
            index=0,
            key="export_format_media_mix"
        )

    # Create export data with selected columns
    export_df = display_df[selected_columns] if selected_columns else pd.DataFrame()

    if not export_df.empty:
        if export_format == "CSV":
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download CSV ({len(export_df)} rows × {len(selected_columns)} columns)",
                data=csv_data,
                file_name=f"amc_media_mix_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_csv_button_media_mix"
            )
        else:  # Excel
            # For Excel export, we would need to add openpyxl to requirements
            csv_data = export_df.to_csv(index=False)  # Temporary, basic CSV for now
            st.download_button(
                label=f"📥 Download Excel ({len(export_df)} rows × {len(selected_columns)} columns)",
                data=csv_data,
                file_name=f"amc_media_mix_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel_button_media_mix",
                help="Note: Currently exports as CSV with .xlsx extension. For true Excel export, add 'openpyxl' to requirements."
            )

    generate_plots = st.button("📊 Generate Charts for Each Row", key="generate_plots_media_mix")

    # Color picker for channels
    st.subheader("Channel Colors")
    color_cols = st.columns(min(4, len(st.session_state.channel_colors)))

    for i, (channel, color) in enumerate(st.session_state.channel_colors.items()):
        with color_cols[i % 4]:
            new_color = st.color_picker(f"{channel}", color, key=f"media_mix_color_{channel}")
            if new_color != color:
                st.session_state.channel_colors[channel] = new_color

    # Display plots separately if generated
    if generate_plots:
        st.subheader("📊 Charts for Each Row")
        st.info(f"Displaying charts for the {len(display_df)} rows shown above")

        # Create and display plots for each row
        for idx, row in display_df.iterrows():
            # Get the path from the row
            path_value = str(row.get('path', f'Row {idx}'))
            path_clean = path_value.replace('[', '').replace(']', '').strip()

            with st.expander(f"📈 Chart for Path: {path_clean}", expanded=False):
                # Create diagrams based on analysis level
                if analysis_level_filter == 'Media Mix':
                    # Extract channels from path
                    try:
                        matches = re.findall(r'/([A-Z\s]+)', str(row['path']).upper())
                        unique_channels = list(dict.fromkeys(matches))  # Remove duplicates while preserving order

                        fig = create_venn_diagram(unique_channels, st.session_state.channel_colors)
                    except Exception as e:
                        # Fallback in case of error - show exact error
                        fig = go.Figure(data=[go.Bar(x=['DEBUG ERROR'], y=[1])])
                        fig.update_layout(title=f"Error: {str(e)[:100]}", height=300)

                st.plotly_chart(fig, config={'responsive': True}, key=f"media_mix_chart_{idx}")


def path_to_conversion_tab():
    st.header("Path to Conversion")

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

    # Fixed analysis level
    analysis_level_filter = 'Path to conversion'

    # Statistics (moved up)
    if len(df[df['analysis_level'] == analysis_level_filter]) > 0:
        filtered_df_for_stats = df[df['analysis_level'] == analysis_level_filter]
        st.subheader("Quick Statistics")

        numeric_cols = filtered_df_for_stats.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            selected_metric = st.selectbox(
                "Select metric for analysis",
                options=numeric_cols,
                index=min(10, len(numeric_cols)-1),  # Default to a metric column
                key="path_to_conversion_overview_metric"
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Average {selected_metric}",
                         f"{filtered_df_for_stats[selected_metric].mean():.2f}")
            with col2:
                st.metric(f"Max {selected_metric}",
                         f"{filtered_df_for_stats[selected_metric].max():.2f}")
            with col3:
                st.metric(f"Min {selected_metric}",
                         f"{filtered_df_for_stats[selected_metric].min():.2f}")

    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)

    with col1:
        granularity_options = df['granularity'].dropna().unique()
        if len(granularity_options) > 0:
            # Default to "format" if available, otherwise first option
            if "Format" in granularity_options:
                default_index = list(granularity_options).index("Format")
            else:
                default_index = 0
            granularity_filter = st.selectbox(
                "Granularity",
                options=granularity_options,
                index=default_index,
                key="path_to_conversion_granularity"
            )
        else:
            granularity_filter = None

    with col2:
        brand_options = df['brand'].dropna().unique()
        if len(brand_options) > 0:
            brand_filter = st.selectbox(
                "Brand",
                options=brand_options,
                index=0,
                key="path_to_conversion_brand"
            )
        else:
            brand_filter = None

    with col3:
        channel_filter = st.multiselect(
            "Path Channel",
            options=path_channels,
            default=[],
            key="path_to_conversion_channel"  # Start with no channels selected
        )

    # Apply filters
    filtered_df = df.copy()
    if analysis_level_filter:
        filtered_df = filtered_df[filtered_df['analysis_level'] == analysis_level_filter]
    if granularity_filter:
        filtered_df = filtered_df[filtered_df['granularity'] == granularity_filter]
    if brand_filter:
        filtered_df = filtered_df[filtered_df['brand'] == brand_filter]
    if channel_filter:
        def contains_selected_channels(path):
            try:
                matches = re.findall(r'/([A-Z\s]+)', str(path).upper())
                return all(channel in matches for channel in channel_filter)
            except:
                return False
        filtered_df = filtered_df[filtered_df['path'].apply(contains_selected_channels)]

    # Add sorting options and max rows
    col1, col2 = st.columns(2)

    with col1:
        sort_options = ['None'] + list(filtered_df.select_dtypes(include=[np.number]).columns)
        sort_by = st.selectbox("Sort by column (descending)", options=sort_options, key="path_to_conversion_sort_by")

    with col2:
        if sort_by != 'None':
            # Always sort in descending order when a column is selected
            sort_order = "Descending"
        else:
            sort_order = "Descending"  # default

        max_rows = st.slider("Maximum rows to display", min_value=5, max_value=max(10, len(filtered_df)), value=min(50, len(filtered_df)), step=5, key="path_to_conversion_max_rows")

    st.subheader(f"Filtered Data ({len(filtered_df)} rows)")

    # Apply sorting if requested
    if sort_by != 'None':
        ascending = (sort_order == "Ascending")
        filtered_df_sorted = filtered_df.sort_values(by=sort_by, ascending=ascending)
    else:
        filtered_df_sorted = filtered_df

    # Limit to max_rows for display and plots
    display_df = filtered_df_sorted.head(max_rows)

    # Display the table first (always, whether plots are generated or not)
    # Highlight columns logic
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

    # Check which KPI columns exist
    existing_conv_cols = [col for col in conversion_columns if col in filtered_df.columns]
    existing_cons_cols = [col for col in consideration_columns if col in filtered_df.columns]

    # Display table with styling
    # Define color function based on column category
    def get_colors(column):
        if column.name in existing_conv_cols:
            return ['background-color: #FFA500'] * len(column)  # Orange
        elif column.name in existing_cons_cols:
            return ['background-color: #C896C8'] * len(column)  # Light purple
        else:
            return [''] * len(column)  # No highlight

    # Apply pandas styling to highlight columns
    styled_df = display_df.style.apply(get_colors, axis=0)
    # Column configuration for constrained width (max ~20 characters)
    column_config = {
        col: st.column_config.Column(width=110) for col in display_df.columns
    }
    st.dataframe(styled_df, column_config=column_config)

    # Column selection for export/download
    st.subheader("🔽 Data Export")
    col1, col2 = st.columns(2)

    with col1:
        selected_columns = st.multiselect(
            "Select columns to export",
            options=list(display_df.columns),
            default=list(display_df.columns[:11]),  # All columns selected by default
            key="export_column_selection_path_to_conversion"
        )

    with col2:
        export_format = st.selectbox(
            "Export format",
            options=["CSV", "Excel"],
            index=0,
            key="export_format_path_to_conversion"
        )

    # Create export data with selected columns
    export_df = display_df[selected_columns] if selected_columns else pd.DataFrame()

    if not export_df.empty:
        if export_format == "CSV":
            csv_data = export_df.to_csv(index=False)
            st.download_button(
                label=f"📥 Download CSV ({len(export_df)} rows × {len(selected_columns)} columns)",
                data=csv_data,
                file_name=f"amc_path_to_conversion_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_csv_button_path_to_conversion"
            )
        else:  # Excel
            # For Excel export, we would need to add openpyxl to requirements
            csv_data = export_df.to_csv(index=False)  # Temporary, basic CSV for now
            st.download_button(
                label=f"📥 Download Excel ({len(export_df)} rows × {len(selected_columns)} columns)",
                data=csv_data,
                file_name=f"amc_path_to_conversion_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel_button_path_to_conversion",
                help="Note: Currently exports as CSV with .xlsx extension. For true Excel export, add 'openpyxl' to requirements."
            )

    generate_plots = st.button("📊 Generate Charts for Each Row", key="generate_plots_path_to_conversion")

    # Color picker for channels
    st.subheader("Channel Colors")
    color_cols = st.columns(min(4, len(st.session_state.channel_colors)))

    for i, (channel, color) in enumerate(st.session_state.channel_colors.items()):
        with color_cols[i % 4]:
            new_color = st.color_picker(f"{channel}", color, key=f"path_to_conversion_color_{channel}")
            if new_color != color:
                st.session_state.channel_colors[channel] = new_color

    # Display plots separately if generated
    if generate_plots:
        st.subheader("📊 Charts for Each Row")
        st.info(f"Displaying charts for the {len(display_df)} rows shown above")

        # Create and display plots for each row
        for idx, row in display_df.iterrows():
            # Get the path from the row
            path_value = str(row.get('path', f'Row {idx}'))
            path_clean = path_value.replace('[', '').replace(']', '').strip()

            with st.expander(f"📈 Chart for Path: {path_clean}", expanded=False):
                # Create diagrams based on analysis level
                if analysis_level_filter == 'Path to conversion':
                    # Extract steps from conversion path
                    try:
                        matches = re.findall(r'/([A-Z\s]+)', str(row['path']).upper())
                        if matches:
                            fig = create_simple_path_diagram(matches, st.session_state.channel_colors)
                        else:
                            fig = go.Figure(data=[go.Bar(x=['No path data'], y=[1])])
                            fig.update_layout(title="No conversion path found", height=300)
                    except:
                        # Fallback in case of error
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
    st.header("Documentation")

    tab1 = st.tabs(["📚 KPI Glossary"])[0]

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

if __name__ == "__main__":
    main()
