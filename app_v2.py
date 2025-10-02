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
    tab1, tab2, tab3 = st.tabs(["📁 Data Upload", "📋 Data Workspace", "📖 Documentation"])

    with tab1:
        data_upload_tab()

    with tab2:
        if st.session_state.data is not None:
            data_table_tab()
        else:
            st.info("Please upload a CSV file first")

    with tab3:
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
            video_cols = [col for col in df.columns if 'video' in col.lower()]
            if video_cols:
                df_filtered = df.drop(columns=video_cols)

        st.session_state.data = df_filtered

        st.subheader("Data Preview")
        st.dataframe(df_filtered.head(), width='stretch')

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
    col1, col2, col3 = st.columns(3)

    with col1:
        analysis_level_options = df['analysis_level'].unique()
        analysis_level_filter = st.selectbox(
            "Analysis Level",
            options=analysis_level_options,
            index=0 if len(analysis_level_options) > 0 else 0
        )

    with col2:
        granularity_options = df['granularity'].dropna().unique()
        granularity_filter = st.selectbox(
            "Granularity",
            options=granularity_options,
            index=0 if len(granularity_options) > 0 else None
        )

    with col3:
        channel_filter = st.multiselect(
            "Path Channel",
            options=path_channels,
            default=[]  # Start with no channels selected
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
            new_color = st.color_picker(f"{channel}", color)
            if new_color != color:
                st.session_state.channel_colors[channel] = new_color

    # Add sorting options
    col1, col2, col3 = st.columns(3)

    with col1:
        sort_options = ['None'] + list(filtered_df.select_dtypes(include=[np.number]).columns)
        sort_by = st.selectbox("Sort by column", options=sort_options)

    with col2:
        if sort_by != 'None':
            sort_order = st.selectbox("Sort order", options=["Descending", "Ascending"])
        else:
            sort_order = "Descending"  # default

    with col3:
        max_rows = st.slider("Maximum rows to display", min_value=5, max_value=max(10, len(filtered_df)), value=min(50, len(filtered_df)), step=5)

    st.subheader(f"Filtered Data ({len(filtered_df)} rows)")

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

    # Prepare warning messages for highlighted columns
    highlight_msgs = []
    if existing_conv_cols:
        highlight_msgs.append(f"🟠 **Conversion KPIs (orange):** {', '.join(existing_conv_cols)}")
    if existing_cons_cols:
        highlight_msgs.append(f"🟣 **Consideration KPIs (purple):** {', '.join(existing_cons_cols)}")

    # Normal display with styling
    if highlight_msgs:
        st.warning("  \n".join(highlight_msgs))

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

        st.dataframe(styled_df)  # Autosize columns by default
    else:
        st.dataframe(display_df)  # Autosize columns by default

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

                        if len(unique_channels) >= 2:
                            # Create Venn diagram visualization
                            fig = create_venn_diagram(unique_channels, st.session_state.channel_colors)
                        else:
                            # Fallback: simple channel display
                            fig = create_single_channel_display(unique_channels, st.session_state.channel_colors)
                    except:
                        # Fallback in case of error
                        fig = go.Figure(data=[go.Bar(x=['Error'], y=[1])])
                        fig.update_layout(title="Error creating Venn diagram", height=300)

                elif analysis_level_filter == 'Path to conversion':
                    # Extract steps from conversion path
                    try:
                        matches = re.findall(r'/([A-Z\s]+)', str(row['path']).upper())
                        if matches:
                            fig = create_chevron_diagram(matches, st.session_state.channel_colors)
                        else:
                            fig = go.Figure(data=[go.Bar(x=['No path data'], y=[1])])
                            fig.update_layout(title="No conversion path found", height=300)
                    except:
                        # Fallback in case of error
                        fig = go.Figure(data=[go.Bar(x=['Error'], y=[1])])
                        fig.update_layout(title="Error creating chevron diagram", height=300)

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

                st.plotly_chart(fig, config={'responsive': True})

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


def create_venn_diagram(channels, color_dict):
    """
    Create a Venn diagram visualization for channels using overlapping circles
    """
    fig = go.Figure()

    # Number of channels
    n_channels = len(channels)

    # Position circles in a pattern that keeps them circular
    if n_channels == 1:
        # Single large circle
        centers = [[0, 0]]
        radius = 1.5
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
        title="Media Mix Channels",
        height=900,
        width=900,  # Square aspect ratio to ensure circular shapes
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'  # Transparent background
    )

    # Force equal aspect ratio to keep circles perfectly round
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig

def create_single_channel_display(channels, color_dict):
    """
    Create a simple display for single channel or fallback
    """
    fig = go.Figure()

    if channels:
        channel = channels[0]
        color = color_dict.get(channel, '#1f77b4')

        # Create a simple colored rectangle
        fig.add_shape(type="rect",
                     x0=-0.5, y0=-0.5, x1=0.5, y1=0.5,
                     fillcolor=color, opacity=0.7, line=dict(width=0))

        # Add channel text
        fig.add_annotation(
            x=0, y=0,
            text=channel,
            showarrow=False,
            font=dict(size=14, color='white', weight='bold'),
            xanchor='center',
            yanchor='middle'
        )

        fig.update_layout(
            height=300,
            width=400,
            showlegend=False,
            plot_bgcolor='white'
        )
    else:
        fig.update_layout(title="No channels found", height=300)

    # Hide axes
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)

    return fig

def create_chevron_diagram(steps, color_dict):
    """
    Create a chevron process diagram for conversion paths
    Shows the flow: Step 1 → Step 2 → Step 3 → ...
    """
    fig = go.Figure()

    n_steps = len(steps)
    if n_steps == 0:
        fig.update_layout(title="No steps in path", height=300)
        return fig

    # Calculate positions for chevrons along the flow
    step_positions = []
    for i in range(n_steps):
        x_pos = i * 2.0  # Space chevrons horizontally
        y_pos = 0
        step_positions.append((x_pos, y_pos))

    # Create chevron shapes for each step
    for i, (step, (x_pos, y_pos)) in enumerate(zip(steps, step_positions)):
        color = color_dict.get(step, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])

        # Create chevron shape (arrow pointing right)
        chevron_width = 1.6
        chevron_height = 1.2

        # Chevron points: left triangle sloping to right
        chevron_points = [
            [x_pos - chevron_width, y_pos],                    # Left point
            [x_pos - chevron_width/4, y_pos + chevron_height/2],  # Top middle
            [x_pos + chevron_width, y_pos],                    # Right point (tip)
            [x_pos - chevron_width/4, y_pos - chevron_height/2],  # Bottom middle
        ]

        # Close the shape
        chevron_points.append(chevron_points[0])

        # Extract x and y coordinates for the shape
        chevron_x = [pt[0] for pt in chevron_points]
        chevron_y = [pt[1] for pt in chevron_points]

        # Add filled chevron shape
        fig.add_trace(go.Scatter(
            x=chevron_x,
            y=chevron_y,
            fill="toself",
            fillcolor=color,
            mode='lines',
            line=dict(color=color, width=2),
            hoverinfo='skip',
            showlegend=False
        ))

        # Add step text inside chevron
        fig.add_annotation(
            x=x_pos,
            y=y_pos,
            text=step,
            showarrow=False,
            font=dict(size=10, color='white', weight='bold'),
            xanchor='center',
            yanchor='middle'
        )

    # Add arrows between steps
    if n_steps > 1:
        for i in range(n_steps - 1):
            start_x = step_positions[i][0] + chevron_width - chevron_width/4
            end_x = step_positions[i+1][0] - chevron_width + chevron_width/4
            center_x = (start_x + end_x) / 2

            # Add arrow line
            fig.add_annotation(
                x=center_x,
                y=0,
                text="→",
                showarrow=False,
                font=dict(size=20, color='black'),
                xanchor='center',
                yanchor='middle'
            )

    # Set axis ranges to fit all chevrons and arrows
    x_min = -1.5
    x_max = (n_steps - 1) * 2.0 + 1.5
    y_min = -1.5
    y_max = 1.5

    fig.update_layout(
        height=400,
        width=max(600, n_steps * 150),  # Adjust width based on number of steps
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Configure axes
    fig.update_xaxes(
        range=[x_min, x_max],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False
    )
    fig.update_yaxes(
        range=[y_min, y_max],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        visible=False,
        scaleanchor="x",  # Ensure square aspect ratio
        scaleratio=1
    )

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

if __name__ == "__main__":
    main()
