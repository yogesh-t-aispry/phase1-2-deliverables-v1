import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Database Configuration
DB_CONFIG = {
    'host': 'ls-8a62cbf5a61df28a063d1a9329e3104b75eefab3.chvkci9uhgu1.ap-south-1.rds.amazonaws.com',
    'database': 'cognigen_new',
    'user': 'dbmasteruser',
    'password': '<lmFpK}PF4u$je+][w=:h4~_!Zz%It]J',
    'port': 5432,
    'connect_timeout': 10
}

# Color Palette - Clean and Professional
COLORS = {
    'primary': '#2E86C1',      # Soft Blue
    'secondary': '#28B463',    # Soft Green  
    'accent': '#F39C12',       # Soft Orange
    'warning': '#E74C3C',      # Soft Red
    'neutral': '#85929E',      # Grey
    'light_blue': '#AED6F1',
    'light_green': '#ABEBC6',
    'light_orange': '#F8C471',
    'light_red': '#F1948A'
}

COLOR_SEQUENCE = [
    COLORS['primary'], COLORS['secondary'], COLORS['accent'],
    COLORS['warning'], COLORS['neutral'], COLORS['light_blue'],
    COLORS['light_green'], COLORS['light_orange']
]

def execute_query(query):
    """Execute SQL query and return DataFrame with proper connection handling"""
    conn = None
    cursor = None
    try:
        # Create a fresh connection for each query
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Check if connection is alive
        if conn.closed:
            st.error("Database connection is closed")
            return pd.DataFrame()
        
        # Execute query and return DataFrame
        df = pd.read_sql_query(query, conn)
        return df
        
    except psycopg2.OperationalError as e:
        st.error(f"Database connection error: {str(e)}")
        st.error("Please check your database credentials and network connectivity")
        return pd.DataFrame()
    except psycopg2.Error as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Query execution failed: {str(e)}")
        return pd.DataFrame()
    finally:
        # Always close the connection
        if conn is not None and not conn.closed:
            conn.close()

@st.cache_data(ttl=300)  # Cache data for 5 minutes instead of caching connection
def execute_query_cached(query):
    """Cached version of execute_query for better performance"""
    return execute_query(query)

def test_connection():
    """Test database connection"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return True, "Connection successful"
    except Exception as e:
        return False, str(e)

def create_metric_card(title, value, delta=None, delta_color="normal", icon=""):
    """Create an enhanced metric card with better styling"""
    delta_html = ""
    if delta:
        delta_class = "metric-delta-positive" if delta_color == "normal" else "metric-delta-negative"
        delta_symbol = "↗" if delta_color == "normal" else "↘"
        delta_html = f'<div class="metric-delta {delta_class}">{delta_symbol} {delta}</div>'
    
    icon_html = f'<span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>' if icon else ""
    
    card_html = f"""
    <div class="metric-card">
        <div class="metric-title">{icon_html}{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def create_chart_container(fig, title, subtitle=""):
    """Create enhanced container for charts"""
    subtitle_html = f'<div style="font-size: 0.9rem; color: #6c757d; margin-top: 0.25rem;">{subtitle}</div>' if subtitle else ""
    
    st.markdown(f"""
    <div class="chart-container">
        <div class="chart-title">{title}{subtitle_html}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.plotly_chart(
        fig, 
        use_container_width=True,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'autoScale2d']
        }
    )

def create_table_container(df, title, format_dict=None, subtitle=""):
    """Create enhanced container for tables"""
    subtitle_html = f'<div style="font-size: 0.9rem; color: #6c757d; margin-top: 0.25rem;">{subtitle}</div>' if subtitle else ""
    
    st.markdown(f"""
    <div class="table-container">
        <div class="table-title">{title}{subtitle_html}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if format_dict:
        styled_df = df.style.format(format_dict)
    else:
        styled_df = df
    
    st.dataframe(styled_df, use_container_width=True, height=400)

def group_revenue_data(df, revenue_col, name_col, top_n=10):
    """Group revenue data into High/Medium/Low performers"""
    if df.empty:
        return df, pd.DataFrame()
    
    df_sorted = df.sort_values(revenue_col, ascending=False).reset_index(drop=True)
    total_items = len(df_sorted)
    
    # Define thresholds
    high_threshold = min(top_n, max(1, int(total_items * 0.1)))  # Top 10% or top_n
    medium_threshold = min(top_n * 3, max(high_threshold + 1, int(total_items * 0.3)))  # Next 20%
    
    # Create groups
    df_sorted['performance_group'] = 'Low Performers'
    df_sorted.iloc[:high_threshold, df_sorted.columns.get_loc('performance_group')] = 'High Performers'
    df_sorted.iloc[high_threshold:medium_threshold, df_sorted.columns.get_loc('performance_group')] = 'Medium Performers'
    
    # Group summary
    grouped_summary = df_sorted.groupby('performance_group').agg({
        revenue_col: 'sum',
        name_col: 'count'
    }).reset_index()
    grouped_summary = grouped_summary.rename(columns={name_col: 'count'})
    
    return df_sorted, grouped_summary

def create_bar_chart(df, x_col, y_col, title, color_col=None, chart_type="bar"):
    """Create a clean bar chart with enhanced styling"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
    
    if chart_type == "horizontal":
        fig = px.bar(
            df,
            x=y_col,
            y=x_col,
            color=color_col if color_col and color_col in df.columns else None,
            color_discrete_sequence=COLOR_SEQUENCE,
            orientation='h'
        )
    else:
        fig = px.bar(
            df,
            x=x_col,
            y=y_col,
            color=color_col if color_col and color_col in df.columns else None,
            color_discrete_sequence=COLOR_SEQUENCE
        )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color="#2C3E50"),
        showlegend=True if color_col else False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.1)',
            showline=True,
            linecolor='rgba(128,128,128,0.2)',
            title_font=dict(size=12, color="#34495E")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(128,128,128,0.1)',
            showline=True,
            linecolor='rgba(128,128,128,0.2)',
            title_font=dict(size=12, color="#34495E")
        ),
        margin=dict(l=50, r=50, t=20, b=50),
        hovermode='x unified'
    )
    
    # Enhanced hover template
    if chart_type == "horizontal":
        hovertemplate = "<b>%{y}</b><br>" + f"{y_col}: %{{x:,.0f}}<br><extra></extra>"
    else:
        hovertemplate = "<b>%{x}</b><br>" + f"{y_col}: %{{y:,.0f}}<br><extra></extra>"
    
    fig.update_traces(
        hovertemplate=hovertemplate,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
    )
    
    return fig

def create_pie_chart(df, values_col, names_col, title):
    """Create a clean pie chart with enhanced styling"""
    if df.empty:
        return go.Figure().add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
    
    fig = px.pie(
        df,
        values=values_col,
        names=names_col,
        color_discrete_sequence=COLOR_SEQUENCE
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=12, color="#2C3E50"),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        ),
        margin=dict(l=20, r=120, t=20, b=20)
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>" +
                     f"{values_col}: %{{value:,.0f}}<br>" +
                     "Percentage: %{percent}<br>" +
                     "<extra></extra>",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial")
    )
    
    return fig

# Streamlit Configuration
st.set_page_config(
    page_title="Cliira Descriptive Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        color: #1A5490;
        text-align: center;
        margin-bottom: 3rem;
        margin-top: 2rem;
        font-family: 'Segoe UI', 'Arial Black', sans-serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        background: linear-gradient(135deg, #1A5490, #2E86C1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #34495E;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2E86C1;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border-left: 4px solid #2E86C1;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    .metric-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #7B8794;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 0.3rem;
    }
    .metric-delta {
        font-size: 0.8rem;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        background-color: #F8F9FA;
        border-radius: 10px;
    }
    .connection-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .connection-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .connection-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    /* Enhanced Chart Container */
    .chart-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .chart-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(0,0,0,0.12);
    }
    
    .chart-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #2E86C1, #28B463, #F39C12);
    }
    
    .chart-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 3px solid #2E86C1;
    }
    
    /* Enhanced Table Container */
    .table-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
        position: relative;
        overflow: hidden;
    }
    
    .table-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #E74C3C, #8E44AD, #3498DB);
    }
    
    .table-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #2C3E50;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 3px solid #E74C3C;
    }
    
    /* Enhanced DataFrame Styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    .stDataFrame > div {
        border-radius: 10px;
    }
    
    /* Section Divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #2E86C1, transparent);
        margin: 2rem 0;
        border-radius: 1px;
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        background-color: transparent;
        border-radius: 10px;
        color: #6c757d;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #2E86C1;
        color: white;
        box-shadow: 0 4px 12px rgba(46, 134, 193, 0.3);
</style>
""", unsafe_allow_html=True)

# Main Dashboard
def main():
    
    # Header
    st.markdown('<h1 class="main-header">Cliira Descriptive Analytics</h1>', unsafe_allow_html=True)
        
    # Sidebar Filters
    with st.sidebar:
        st.header("🔍 Global Filters")
        
        # Get department list
        dept_query = "SELECT DISTINCT dept_name FROM departments ORDER BY dept_name"
        dept_df = execute_query_cached(dept_query)
        
        if not dept_df.empty:
            dept_options = ["All"] + dept_df['dept_name'].tolist()
            selected_dept = st.selectbox(
                "Select Department",
                options=dept_options,
                index=0  # Default to "All"
            )
            
            # Convert selection to list for backend compatibility
            if selected_dept == "All":
                selected_depts = dept_df['dept_name'].tolist()
            else:
                selected_depts = [selected_dept]
        else:
            selected_depts = []
            st.warning("No departments found")
        
        # Category filter
        category_query = "SELECT DISTINCT category FROM skus WHERE category IS NOT NULL ORDER BY category"
        category_df = execute_query_cached(category_query)
        
        if not category_df.empty:
            category_options = ["All"] + category_df['category'].tolist()
            selected_category = st.selectbox(
                "Select Category",
                options=category_options,
                index=0  # Default to "All"
            )
            
            # Convert selection to list for backend compatibility
            if selected_category == "All":
                selected_categories = category_df['category'].tolist()
            else:
                selected_categories = [selected_category]
        else:
            selected_categories = []
    
    # Global Metrics
    st.markdown('<h2 class="section-header">Key Performance Overview</h2>', unsafe_allow_html=True)

    if selected_depts:
        dept_list = "'" + "','".join(selected_depts) + "'"
        dept_filter = f"AND d.dept_name IN ({dept_list})"
    else:
        dept_filter = ""

    if selected_categories:
        category_list = "'" + "','".join(selected_categories) + "'"
        category_filter = f"AND s.category IN ({category_list})"
    else:
        category_filter = ""

    global_metrics_query = f"""
    WITH transactions_all AS (
        SELECT * FROM transactions_2023
        UNION ALL
        SELECT * FROM transactions_2024
    )
    SELECT 
        SUM(t.total_cost) as total_revenue,
        COUNT(DISTINCT t.patient_id) as total_patients,
        COUNT(DISTINCT t.sku_id) as total_skus,
        COUNT(DISTINCT t.transaction_id) as total_transactions
    FROM transactions_all t
    JOIN departments d ON t.dept_id = d.dept_id AND t.hospital_id = d.hospital_id
    JOIN skus s ON t.sku_id = s.sku_id
    WHERE t.total_cost IS NOT NULL AND t.quantity_consumed > 0
        {dept_filter}
        {category_filter}
    """

    global_metrics = execute_query_cached(global_metrics_query)

    if not global_metrics.empty:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            revenue_val = f"₹{global_metrics.iloc[0]['total_revenue']:,.0f}" if global_metrics.iloc[0]['total_revenue'] else "₹0"
            create_metric_card("Total Revenue", revenue_val, icon="💰")
        
        with col2:
            patients_val = f"{global_metrics.iloc[0]['total_patients']:,}" if global_metrics.iloc[0]['total_patients'] else "0"
            create_metric_card("Total Patients", patients_val, icon="👥")
        
        with col3:
            total_skus_val = int(global_metrics.iloc[0]['total_skus'] or 0)
            skus_val = f"{total_skus_val:,}"
            create_metric_card("Total SKUs", skus_val, icon="📦")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


    # Revenue Analysis
    st.markdown('<h2 class="section-header">Revenue Analysis</h2>', unsafe_allow_html=True)
    
    revenue_dept_tab, revenue_sku_tab = st.tabs(["Department Level", "SKU Level"])
    
    with revenue_dept_tab:
        if selected_depts:
            dept_list = "'" + "','".join(selected_depts) + "'"
            dept_filter = f"AND d.dept_name IN ({dept_list})"
        else:
            dept_filter = ""
        
        revenue_dept_query = f"""
            SELECT 
                d.dept_name,
                d.dept_id,
                COUNT(DISTINCT t.sku_id) as unique_skus,
                SUM(t.quantity_consumed) as total_quantity,
                SUM(t.total_cost) as total_revenue
            FROM (
                SELECT * FROM transactions_2023
                UNION ALL
                SELECT * FROM transactions_2024
            ) t
            JOIN departments d ON t.dept_id = d.dept_id
            JOIN skus s ON t.sku_id = s.sku_id
            WHERE t.total_cost IS NOT NULL 
                AND t.quantity_consumed > 0
                {dept_filter}
                {category_filter}
            GROUP BY d.dept_name, d.dept_id
            ORDER BY total_revenue DESC;
            """
        
        dept_revenue_df = execute_query_cached(revenue_dept_query)
        
        if not dept_revenue_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dept_revenue = create_bar_chart(
                    dept_revenue_df.head(10), 
                    'dept_name', 
                    'total_revenue', 
                    'Top 10 Departments by Revenue'
                )
                create_chart_container(fig_dept_revenue, "📊 Top 10 Departments by Revenue", 
                                    f"Total departments analyzed: {len(dept_revenue_df)}")
            
            with col2:
                fig_dept_skus = create_bar_chart(
                    dept_revenue_df.head(10), 
                    'dept_name', 
                    'unique_skus', 
                    'Top 10 Departments by SKU Count'
                )
                create_chart_container(fig_dept_skus, "📦 Department SKU Distribution",
                                    "Showing unique SKU count per department")
            
            # Department details table
            with st.expander("📋 Department Revenue Details"):
                create_table_container(
                    dept_revenue_df,
                    "Comprehensive Department Revenue Analysis",
                    {
                        'total_revenue': '₹{:,.0f}',
                        'total_quantity': '{:,.0f}',
                        'unique_skus': '{:,.0f}'
                    },
                    f"Showing data for {len(dept_revenue_df)} departments"
                )
    
    with revenue_sku_tab:
        if selected_categories:
            category_list = "'" + "','".join(selected_categories) + "'"
            category_filter = f"AND s.category IN ({category_list})"
        else:
            category_filter = ""
        
        revenue_sku_query = f"""
        SELECT 
            s.sku_name,
            s.sku_id,
            s.category,
            s.sub_category,
            s.therapeutic_area,
            s.brand_generic_flag,
            SUM(t.quantity_consumed) AS sku_total_quantity,
            AVG(t.unit_cost) AS sku_avg_unit_cost,
            SUM(t.total_cost) AS sku_total_revenue,
            COUNT(DISTINCT t.transaction_id) AS total_transactions,
            COUNT(DISTINCT t.patient_id) AS unique_patients_served,
            COUNT(DISTINCT t.physician_id) AS unique_physicians_prescribing,
            ROUND((
                SUM(t.total_cost) / NULLIF(COUNT(DISTINCT t.patient_id), 0)
            )::numeric, 2) AS revenue_per_patient,
            ROUND((
                SUM(t.total_cost) / NULLIF(COUNT(DISTINCT t.transaction_id), 0)
            )::numeric, 2) AS revenue_per_transaction
        FROM (
            SELECT * FROM transactions_2023
            UNION ALL
            SELECT * FROM transactions_2024
        ) t
        JOIN skus s ON t.sku_id = s.sku_id
        WHERE t.total_cost IS NOT NULL 
            AND t.quantity_consumed > 0
            {category_filter}
        GROUP BY s.sku_name, s.sku_id, 
                 s.category, s.sub_category, s.therapeutic_area, s.brand_generic_flag
        ORDER BY sku_total_revenue DESC;
        """
        
        sku_revenue_df = execute_query_cached(revenue_sku_query)
        
        if not sku_revenue_df.empty:
            # Group SKUs by performance
            grouped_skus, grouped_summary = group_revenue_data(
                sku_revenue_df, 'sku_total_revenue', 'sku_name'
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if not grouped_summary.empty:
                    fig_performance = create_bar_chart(
                        grouped_summary, 
                        'performance_group', 
                        'sku_total_revenue', 
                        'Revenue by Performance Groups'
                    )
                    create_chart_container(fig_performance, "🎯 Revenue by Performance Groups",
                                        "High/Medium/Low performing SKU analysis")
            
            with col2:
                # Brand vs Generic analysis
                if 'brand_generic_flag' in sku_revenue_df.columns:
                    brand_generic = sku_revenue_df.groupby('brand_generic_flag')['sku_total_revenue'].sum().reset_index()
                    fig_brand = create_pie_chart(
                        brand_generic, 
                        'sku_total_revenue', 
                        'brand_generic_flag', 
                        'Revenue: Brand vs Generic'
                    )
                    create_chart_container(fig_brand, "💊 Revenue: Brand vs Generic Split",
                                        "Revenue distribution across brand types")
            
            # Performance group drill-down
            selected_group = st.selectbox(
                "🔍 Select Performance Group to View Details:", 
                options=['All'] + grouped_summary['performance_group'].tolist() if not grouped_summary.empty else ['All']
            )
            
            if selected_group != 'All' and not grouped_skus.empty:
                filtered_skus = grouped_skus[grouped_skus['performance_group'] == selected_group].head(20)
                
                fig_drill_down = create_bar_chart(
                    filtered_skus, 
                    'sku_name', 
                    'sku_total_revenue', 
                    f'{selected_group} - Individual SKU Revenue'
                )
                fig_drill_down.update_layout(xaxis=dict(tickangle=45))
                create_chart_container(fig_drill_down, f"🔬 {selected_group} - Individual SKU Revenue",
                                    f"Top 20 SKUs in {selected_group.lower()} category")
                
                with st.expander(f"📊 {selected_group} - Detailed Data"):
                    create_table_container(
                        filtered_skus,
                        f"{selected_group} - Detailed Performance Metrics",
                        {
                            'sku_total_revenue': '₹{:,.0f}',
                            'sku_avg_unit_cost': '₹{:,.2f}',
                            'revenue_per_patient': '₹{:,.2f}'
                        },
                        f"Detailed breakdown of {len(filtered_skus)} SKUs"
                    )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    # Margin Analysis
    st.markdown('<h2 class="section-header">Margin Analysis</h2>', unsafe_allow_html=True)

    margin_dept_tab, margin_sku_tab = st.tabs(["Department Level", "SKU Level"])

    with margin_dept_tab:
        # Define dept_list and dept_filter
        if selected_depts:
            dept_list = "'" + "','".join(selected_depts) + "'"
            dept_filter = f"AND dep.dept_name IN ({dept_list})"
        else:
            dept_list = ""
            dept_filter = ""
            
        margin_dept_query = f"""
            WITH delivery_ranked AS (
                SELECT
                    sku_id,
                    hospital_id,
                    delivery_date,
                    actual_unit_price,
                    ROW_NUMBER() OVER (PARTITION BY sku_id, hospital_id ORDER BY delivery_date DESC) as rn
                FROM deliveries
            ),
            transactions_all AS (
                SELECT * FROM transactions_2023
                UNION ALL
                SELECT * FROM transactions_2024
            )
            SELECT
                dep.dept_id,
                dep.dept_name,
                SUM(t.total_cost) as total_revenue,
                SUM(t.quantity_consumed * d.actual_unit_price) as total_cost,
                SUM(t.total_cost) - SUM(t.quantity_consumed * d.actual_unit_price) as margin_amount,
                CASE
                    WHEN SUM(t.quantity_consumed * d.actual_unit_price) > 0
                    THEN ((SUM(t.total_cost) - SUM(t.quantity_consumed * d.actual_unit_price)) / SUM(t.quantity_consumed * d.actual_unit_price) * 100)
                    ELSE 0
                END as margin_percentage
            FROM transactions_all t
            JOIN departments dep ON t.dept_id = dep.dept_id AND t.hospital_id = dep.hospital_id
            JOIN skus s ON t.sku_id = s.sku_id
            JOIN delivery_ranked d ON t.sku_id = d.sku_id
                AND t.hospital_id = d.hospital_id
                AND d.delivery_date <= t.transaction_date
                AND d.rn = 1
            WHERE 1=1 {dept_filter} {category_filter}
            GROUP BY dep.dept_id, dep.dept_name
            ORDER BY margin_amount DESC;
            """
        
        margin_dept_df = execute_query_cached(margin_dept_query)
    
        if not margin_dept_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_margin_amt = create_bar_chart(
                    margin_dept_df.head(10), 
                    'dept_name', 
                    'margin_amount', 
                    'Top 10 Departments by Margin Amount'
                )
                create_chart_container(fig_margin_amt, "💰 Top 10 Departments by Margin Amount",
                                    "Absolute margin performance analysis")
            
            with col2:
                fig_margin_pct = create_bar_chart(
                    margin_dept_df.head(10), 
                    'dept_name', 
                    'margin_percentage', 
                    'Top 10 Departments by Margin %'
                )
                create_chart_container(fig_margin_pct, "📈 Top 10 Departments by Margin %",
                                    "Percentage-based margin efficiency")

    with margin_sku_tab:
        if selected_categories:
            category_list = "'" + "','".join(selected_categories) + "'"
            category_filter = f"AND s.category IN ({category_list})"
        else:
            category_filter = ""
            
        margin_sku_query = f"""
        WITH delivery_ranked AS (
            SELECT
                sku_id,
                hospital_id,
                delivery_date,
                actual_unit_price,
                ROW_NUMBER() OVER (PARTITION BY sku_id, hospital_id ORDER BY delivery_date DESC) as rn
            FROM deliveries
        ),
        transactions_all AS (
            SELECT * FROM transactions_2023
            UNION ALL
            SELECT * FROM transactions_2024
        )
        SELECT
            s.sku_id,
            s.sku_name,
            s.category,
            SUM(t.total_cost) as total_revenue,
            SUM(t.quantity_consumed * d.actual_unit_price) as total_cost,
            SUM(t.total_cost) - SUM(t.quantity_consumed * d.actual_unit_price) as margin_amount,
            CASE
                WHEN SUM(t.quantity_consumed * d.actual_unit_price) > 0
                THEN ((SUM(t.total_cost) - SUM(t.quantity_consumed * d.actual_unit_price)) / SUM(t.quantity_consumed * d.actual_unit_price) * 100)
                ELSE 0
            END as margin_percentage
        FROM transactions_all t
        JOIN skus s ON t.sku_id = s.sku_id
        JOIN delivery_ranked d ON t.sku_id = d.sku_id
            AND t.hospital_id = d.hospital_id
            AND d.delivery_date <= t.transaction_date
            AND d.rn = 1
        WHERE 1=1 {category_filter}
        GROUP BY s.sku_id, s.sku_name, s.category
        ORDER BY margin_amount DESC;
        """
        
        margin_sku_df = execute_query_cached(margin_sku_query)
        
        if not margin_sku_df.empty:
            # Group by margin performance
            grouped_margins, margin_summary = group_revenue_data(
                margin_sku_df, 'margin_amount', 'sku_name', top_n=15
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if not margin_summary.empty:
                    fig_margin_groups = create_bar_chart(
                        margin_summary, 
                        'performance_group', 
                        'margin_amount', 
                        'Margin by Performance Groups'
                    )
                    st.plotly_chart(fig_margin_groups, use_container_width=True)
            
            with col2:
                # Scatter plot: Margin vs Revenue
                fig_scatter = px.scatter(
                    margin_sku_df.head(100), 
                    x='total_revenue', 
                    y='margin_amount',
                    color='category',
                    title='Margin vs Revenue Analysis',
                    color_discrete_sequence=COLOR_SEQUENCE
                )
                fig_scatter.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Arial, sans-serif", size=12)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

    # Prescription Analytics
    st.markdown('<h2 class="section-header">Prescription Analytics</h2>', unsafe_allow_html=True)
    
    rx_dept_tab, rx_sku_tab = st.tabs(["Department Level", "SKU Level"])
    
    with rx_dept_tab:
        if selected_depts:
            dept_list = "'" + "','".join(selected_depts) + "'"
            dept_filter = f"AND d.dept_name IN ({dept_list})"
        else:
            dept_filter = ""
            
        rx_dept_query = f"""
        WITH all_transactions AS (
            SELECT DISTINCT t.transaction_id, t.dept_id, t.transaction_date, t.sku_id
            FROM transactions_2023 t
            JOIN skus s ON t.sku_id = s.sku_id
            WHERE t.transaction_type = 'Prescription' {category_filter}
            UNION
            SELECT DISTINCT t.transaction_id, t.dept_id, t.transaction_date, t.sku_id
            FROM transactions_2024 t
            JOIN skus s ON t.sku_id = s.sku_id
            WHERE t.transaction_type = 'Prescription' {category_filter}
        )
        SELECT
            d.dept_name,
            d.dept_type,
            COUNT(DISTINCT t.transaction_id) as total_prescriptions
        FROM all_transactions t
        JOIN departments d ON t.dept_id = d.dept_id
        WHERE 1=1 {dept_filter}
        GROUP BY d.dept_name, d.dept_type
        ORDER BY total_prescriptions DESC;
        """
        
        rx_dept_df = execute_query_cached(rx_dept_query)
    
        if not rx_dept_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_rx_dept = create_bar_chart(
                    rx_dept_df.head(10), 
                    'dept_name', 
                    'total_prescriptions', 
                    'Prescriptions by Department'
                )
                create_chart_container(fig_rx_dept, "💊 Prescriptions by Department",
                                    "Total prescription volume analysis")
            
            with col2:
                # Department type distribution
                dept_type_dist = rx_dept_df.groupby('dept_type')['total_prescriptions'].sum().reset_index()
                fig_dept_type = create_pie_chart(
                    dept_type_dist, 
                    'total_prescriptions', 
                    'dept_type', 
                    'Prescriptions by Department Type'
                )
                create_chart_container(fig_dept_type, "🏥 Prescriptions by Department Type",
                                    "Distribution across department categories")


    with rx_sku_tab:
        if selected_categories:
            category_list = "'" + "','".join(selected_categories) + "'"
            category_filter = f"AND s.category IN ({category_list})"
        else:
            category_filter = ""
            
        rx_sku_query = f"""
        WITH all_transactions AS (
            SELECT transaction_id, dept_id, sku_id, transaction_date
            FROM transactions_2023
            WHERE transaction_type = 'Prescription'
            UNION ALL
            SELECT transaction_id, dept_id, sku_id, transaction_date
            FROM transactions_2024
            WHERE transaction_type = 'Prescription'
        )
        SELECT
            s.sku_name,
            s.category,
            s.sub_category,
            s.therapeutic_area,
            s.brand_generic_flag,
            COUNT(DISTINCT t.transaction_id) as prescriptions_containing_this_sku,
            COUNT(DISTINCT t.dept_id) as prescribed_by_departments
        FROM all_transactions t
        JOIN skus s ON t.sku_id = s.sku_id
        WHERE 1=1 {category_filter}
        GROUP BY s.sku_name, s.category, s.sub_category, s.therapeutic_area, s.brand_generic_flag
        ORDER BY prescriptions_containing_this_sku DESC;
        """
        
        rx_sku_df = execute_query_cached(rx_sku_query)
        
        if not rx_sku_df.empty:
            # Group by therapeutic area
            therapeutic_area_dist = rx_sku_df.groupby('therapeutic_area')['prescriptions_containing_this_sku'].sum().reset_index()
            therapeutic_area_dist = therapeutic_area_dist.sort_values('prescriptions_containing_this_sku', ascending=False).head(10)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_therapeutic = create_bar_chart(
                    therapeutic_area_dist, 
                    'therapeutic_area', 
                    'prescriptions_containing_this_sku', 
                    'Prescriptions by Therapeutic Area'
                )
                fig_therapeutic.update_layout(xaxis=dict(tickangle=45))
                st.plotly_chart(fig_therapeutic, use_container_width=True)
            
            with col2:
                # Top prescribed SKUs
                fig_top_rx = create_bar_chart(
                    rx_sku_df.head(15), 
                    'sku_name', 
                    'prescriptions_containing_this_sku', 
                    'Top 15 Prescribed SKUs'
                )
                fig_top_rx.update_layout(xaxis=dict(tickangle=45))
                st.plotly_chart(fig_top_rx, use_container_width=True)
    
    # Margin per Patient Analysis
    st.markdown('<h2 class="section-header">Margin per Patient Analysis</h2>', unsafe_allow_html=True)

    margin_patient_dept_tab, margin_patient_sku_tab = st.tabs(["Department Level", "SKU Level"])

    with margin_patient_dept_tab:
        if selected_depts:
            dept_list = "'" + "','".join(selected_depts) + "'"
            dept_filter = f"AND dep.dept_name IN ({dept_list})"
        else:
            dept_filter = ""
        
        margin_per_patient_dept_query = f"""
        WITH delivery_ranked AS (
            SELECT
                sku_id,
                hospital_id,
                delivery_date,
                actual_unit_price,
                ROW_NUMBER() OVER (PARTITION BY sku_id, hospital_id ORDER BY delivery_date DESC) as rn
            FROM deliveries
        ),
        transactions_all AS (
            SELECT * FROM transactions_2023
            UNION ALL
            SELECT * FROM transactions_2024
        )
        SELECT
            dep.dept_id,
            dep.dept_name,
            COUNT(DISTINCT t.patient_id) as total_patients,
            SUM(t.total_cost) - SUM(t.quantity_consumed * d.actual_unit_price) as total_margin,
            CASE
                WHEN COUNT(DISTINCT t.patient_id) > 0
                THEN (SUM(t.total_cost) - SUM(t.quantity_consumed * d.actual_unit_price)) / COUNT(DISTINCT t.patient_id)
                ELSE 0
            END as margin_per_patient
        FROM transactions_all t
        JOIN departments dep ON t.dept_id = dep.dept_id AND t.hospital_id = dep.hospital_id
        JOIN skus s ON t.sku_id = s.sku_id
        JOIN delivery_ranked d ON t.sku_id = d.sku_id
            AND t.hospital_id = d.hospital_id
            AND d.delivery_date <= t.transaction_date
            AND d.rn = 1
        WHERE t.patient_id IS NOT NULL {dept_filter} {category_filter}
        GROUP BY dep.dept_id, dep.dept_name
        ORDER BY margin_per_patient DESC;
        """
        
        margin_patient_dept_df = execute_query_cached(margin_per_patient_dept_query)
        
        if not margin_patient_dept_df.empty:
            col1, col2 = st.columns(2)
        
            with col1:
                fig_margin_patient = create_bar_chart(
                    margin_patient_dept_df.head(10), 
                    'dept_name', 
                    'margin_per_patient', 
                    'Top 10 Departments by Margin per Patient'
                )
                create_chart_container(fig_margin_patient, "💰 Top 10 Departments by Margin per Patient",
                                    "Margin efficiency per patient by department")
        
            with col2:
                fig_patients_count = create_bar_chart(
                    margin_patient_dept_df.head(10), 
                    'dept_name', 
                    'total_patients', 
                    'Patient Count by Department'
                )
                create_chart_container(fig_patients_count, "👥 Patient Count by Department",
                                    "Total patients served by department")

    with margin_patient_sku_tab:
        if selected_categories:
            category_list = "'" + "','".join(selected_categories) + "'"
            category_filter = f"AND s.category IN ({category_list})"
        else:
            category_filter = ""
        
        margin_per_patient_sku_query = f"""
        WITH delivery_ranked AS (
            SELECT
                sku_id,
                hospital_id,
                delivery_date,
                actual_unit_price,
                ROW_NUMBER() OVER (PARTITION BY sku_id, hospital_id ORDER BY delivery_date DESC) as rn
            FROM deliveries
        ),
        transactions_all AS (
            SELECT * FROM transactions_2023
            UNION ALL
            SELECT * FROM transactions_2024
        )
        SELECT
            s.sku_id,
            s.sku_name,
            s.category,
            COUNT(DISTINCT t.patient_id) as total_patients,
            SUM(t.total_cost) - SUM(t.quantity_consumed * d.actual_unit_price) as total_margin,
            CASE
                WHEN COUNT(DISTINCT t.patient_id) > 0
                THEN (SUM(t.total_cost) - SUM(t.quantity_consumed * d.actual_unit_price)) / COUNT(DISTINCT t.patient_id)
                ELSE 0
            END as margin_per_patient
        FROM transactions_all t
        JOIN skus s ON t.sku_id = s.sku_id
        JOIN delivery_ranked d ON t.sku_id = d.sku_id
            AND t.hospital_id = d.hospital_id
            AND d.delivery_date <= t.transaction_date
            AND d.rn = 1
        WHERE t.patient_id IS NOT NULL {category_filter}
        GROUP BY s.sku_id, s.sku_name, s.category
        ORDER BY margin_per_patient DESC;
        """
        
        margin_patient_sku_df = execute_query_cached(margin_per_patient_sku_query)
        
        if not margin_patient_sku_df.empty:
            # Show top performing SKUs by margin per patient
            fig_sku_margin_patient = create_bar_chart(
                margin_patient_sku_df.head(20), 
                'sku_name', 
                'margin_per_patient', 
                'Top 20 SKUs by Margin per Patient'
            )
            fig_sku_margin_patient.update_layout(xaxis=dict(tickangle=45))
            create_chart_container(fig_sku_margin_patient, "💊 Top 20 SKUs by Margin per Patient",
                                "Margin efficiency per patient by SKU")

   # Bounce Rate Analysis
    st.markdown('<h2 class="section-header">Bounce Rate Analysis</h2>', unsafe_allow_html=True)

    bounce_dept_tab, bounce_sku_tab = st.tabs(["Department Level", "SKU Level"])

    with bounce_dept_tab:
        bounce_dept_query = f"""
        WITH transactions_all AS (
            SELECT * FROM transactions_2023
            UNION ALL
            SELECT * FROM transactions_2024
        ),
        dept_avg_cost AS (
            SELECT
                t.dept_id,
                t.sku_id,
                AVG(t.unit_cost) as avg_unit_cost
            FROM transactions_all t
            JOIN skus s ON t.sku_id = s.sku_id
            WHERE 1=1 {category_filter}
            GROUP BY t.dept_id, t.sku_id
        )
        SELECT
            dep.dept_id,
            dep.dept_name,
            COUNT(*) as total_sku_records,
            COUNT(CASE WHEN i.bounce_status = 'Bounced' THEN 1 END) as bounced_records,
            SUM(CASE WHEN i.bounce_status = 'Bounced' THEN COALESCE(i.bounce_quantity, 0) ELSE 0 END) as total_bounced_quantity,
            SUM(CASE WHEN i.bounce_status = 'Bounced' THEN COALESCE(i.bounce_quantity, 0) * COALESCE(dac.avg_unit_cost, 0) ELSE 0 END) as financial_impact,
            ROUND(
                (COUNT(CASE WHEN i.bounce_status = 'Bounced' THEN 1 END) * 100.0) / NULLIF(COUNT(*), 0), 
                2
            ) as bounce_rate_percentage
        FROM inventory i
        JOIN departments dep ON i.hospital_id = dep.hospital_id
        JOIN skus s ON i.sku_id = s.sku_id
        LEFT JOIN dept_avg_cost dac ON dep.dept_id = dac.dept_id AND i.sku_id = dac.sku_id
        WHERE i.bounce_status IS NOT NULL {dept_filter} {category_filter}
        GROUP BY dep.dept_id, dep.dept_name
        ORDER BY total_bounced_quantity DESC;
        """

        bounce_dept_df = execute_query_cached(bounce_dept_query)

        if not bounce_dept_df.empty:
            col1, col2 = st.columns(2)
        
            with col1:
                fig_bounce_quantity = create_bar_chart(
                    bounce_dept_df.head(10), 
                    'dept_name', 
                    'total_bounced_quantity', 
                    'Bounced Quantity by Department'
                )
                create_chart_container(fig_bounce_quantity, "📦 Bounced Quantity by Department",
                                    "Total items bounced per department")
        
            with col2:
                fig_bounce_impact = create_bar_chart(
                    bounce_dept_df.head(10), 
                    'dept_name', 
                    'financial_impact', 
                    'Financial Impact of Bounces'
                )
                create_chart_container(fig_bounce_impact, "💸 Financial Impact of Bounces",
                                    "Monetary loss due to bounced items")
                
            # Additional bounce rate charts
            col3, col4 = st.columns(2)
            
            with col3:
                fig_bounce_rate_dept = create_bar_chart(
                    bounce_dept_df.head(10), 
                    'dept_name', 
                    'bounce_rate_percentage', 
                    'Bounce Rate Percentage by Department'
                )
                create_chart_container(fig_bounce_rate_dept, "📊 Bounce Rate % by Department",
                                    "Percentage of items bounced")
                
            with col4:
                # Get SKU-wise bounce data for the chart
                sku_bounce_query = """
                WITH transactions_all AS (
                    SELECT * FROM transactions_2023
                    UNION ALL
                    SELECT * FROM transactions_2024
                ),
                sku_totals AS (
                    SELECT
                        t.sku_id,
                        COUNT(*) AS total_records
                    FROM transactions_all t
                    GROUP BY t.sku_id
                )
                SELECT
                    s.sku_id,
                    s.sku_name,
                    COUNT(CASE WHEN i.bounce_status = 'Bounced' THEN 1 END) AS bounced_records,
                    st.total_records,
                    ROUND(
                        (COUNT(CASE WHEN i.bounce_status = 'Bounced' THEN 1 END) * 100.0) / NULLIF(st.total_records, 0),
                        2
                    ) AS sku_bounce_rate_percentage
                FROM skus s
                JOIN sku_totals st ON s.sku_id = st.sku_id
                LEFT JOIN inventory i ON s.sku_id = i.sku_id
                GROUP BY s.sku_id, s.sku_name, st.total_records
                HAVING COUNT(CASE WHEN i.bounce_status = 'Bounced' THEN 1 END) > 0
                ORDER BY sku_bounce_rate_percentage DESC;
                """

                sku_bounce_df = execute_query_cached(sku_bounce_query)

                if not sku_bounce_df.empty:
                    fig_sku_bounce_rate = create_bar_chart(
                        sku_bounce_df,
                        'sku_name',
                        'sku_bounce_rate_percentage',
                        'Bounce Rate Percentage by SKU'
                    )
                    create_chart_container(fig_sku_bounce_rate, "🎯 Bounce Rate % by SKU",
                                        "Individual SKU bounce performance")

        else:
            st.warning("No bounce data found or bounce_status column may not exist in the inventory table")

    with bounce_sku_tab:
        bounce_sku_query = """
        -- Combine transactions from 2023 and 2024
        WITH transactions_all AS (
            SELECT * FROM transactions_2023
            UNION ALL
            SELECT * FROM transactions_2024
        ),
        -- Calculate average unit cost per SKU
        sku_avg_cost AS (
            SELECT
                t.sku_id,
                AVG(t.unit_cost) as avg_unit_cost
            FROM transactions_all t
            GROUP BY t.sku_id
        )
        -- Aggregate bounce metrics by SKU
        SELECT
            s.sku_id,
            s.sku_name,
            s.category,
            COUNT(*) as total_records,
            COUNT(CASE WHEN i.bounce_status = 'Bounced' THEN 1 END) as bounced_records,
            SUM(CASE WHEN i.bounce_status = 'Bounced' THEN COALESCE(i.bounce_quantity, 0) ELSE 0 END) as total_bounced_quantity,
            SUM(CASE WHEN i.bounce_status = 'Bounced' THEN COALESCE(i.bounce_quantity, 0) * COALESCE(sac.avg_unit_cost, 0) ELSE 0 END) as financial_impact,
            ROUND(
                (COUNT(CASE WHEN i.bounce_status = 'Bounced' THEN 1 END) * 100.0) / NULLIF(COUNT(*), 0), 
                2
            ) as bounce_rate_percentage
        FROM inventory i
        JOIN skus s ON i.sku_id = s.sku_id
        LEFT JOIN sku_avg_cost sac ON i.sku_id = sac.sku_id
        WHERE i.bounce_status IS NOT NULL
        GROUP BY s.sku_id, s.sku_name, s.category
        HAVING COUNT(CASE WHEN i.bounce_status = 'Bounced' THEN 1 END) > 0
        ORDER BY total_bounced_quantity DESC;
        """

        bounce_sku_df = execute_query_cached(bounce_sku_query)

        if not bounce_sku_df.empty and len(bounce_sku_df) > 0:
            # Filter out rows where total_bounced_quantity is 0 or null
            bounce_sku_df_filtered = bounce_sku_df[
                (bounce_sku_df['total_bounced_quantity'] > 0) & 
                (bounce_sku_df['total_bounced_quantity'].notna())
            ]
            
            if not bounce_sku_df_filtered.empty:
                col1, col2 = st.columns(2)
                
                # Get the actual number of SKUs to display (max 15)
                num_skus_to_show = min(len(bounce_sku_df_filtered), 15)
                top_skus_df = bounce_sku_df_filtered.head(num_skus_to_show)
            
                with col1:
                    fig_sku_bounce = create_bar_chart(
                        top_skus_df, 
                        'sku_name', 
                        'total_bounced_quantity', 
                        f'Top {num_skus_to_show} SKUs by Bounce Quantity'
                    )
                    fig_sku_bounce.update_layout(xaxis=dict(tickangle=45))
                    create_chart_container(fig_sku_bounce, f"📦 Top {num_skus_to_show} SKUs by Bounce Quantity",
                                        f"Showing SKUs with actual bounce quantities")
            
                with col2:
                    fig_sku_financial_impact = create_bar_chart(
                        top_skus_df, 
                        'sku_name', 
                        'financial_impact', 
                        f'Financial Impact of Top {num_skus_to_show} Bounced SKUs'
                    )
                    fig_sku_financial_impact.update_layout(xaxis=dict(tickangle=45))
                    create_chart_container(fig_sku_financial_impact, f"💸 Financial Impact of Top {num_skus_to_show} Bounced SKUs",
                                        f"Monetary impact of bounced items")
                                                
                # Detailed bounce data table
                with st.expander("📊 Detailed Bounce Analysis"):
                    create_table_container(
                        bounce_sku_df_filtered,
                        "Comprehensive SKU Bounce Analysis",
                        {
                            'total_bounced_quantity': '{:,.0f}',
                            'financial_impact': '₹{:,.2f}',
                            'bounce_rate_percentage': '{:.2f}%'
                        },
                        f"Detailed breakdown of {len(bounce_sku_df_filtered)} SKUs with bounce data"
                    )
            else:
                st.info("No SKUs found with actual bounce quantities greater than 0")
        else:
            st.warning("No bounce data found for SKUs or bounce_status column may not exist in the inventory table")

    # with bounce_sku_tab:
    #     bounce_sku_query = """
    #     -- Combine transactions from 2023 and 2024
    #     WITH transactions_all AS (
    #         SELECT * FROM transactions_2023
    #         UNION ALL
    #         SELECT * FROM transactions_2024
    #     ),
    #     -- Calculate average unit cost per SKU
    #     sku_avg_cost AS (
    #         SELECT
    #             t.sku_id,
    #             AVG(t.unit_cost) as avg_unit_cost
    #         FROM transactions_all t
    #         GROUP BY t.sku_id
    #     )
    #     -- Aggregate bounce metrics by SKU
    #     SELECT
    #         s.sku_id,
    #         s.sku_name,
    #         s.category,
    #         COUNT(*) as total_records,
    #         COUNT(CASE WHEN i.bounce_status = 'Bounced' THEN 1 END) as bounced_records,
    #         SUM(CASE WHEN i.bounce_status = 'Bounced' THEN COALESCE(i.bounce_quantity, 0) ELSE 0 END) as total_bounced_quantity,
    #         SUM(CASE WHEN i.bounce_status = 'Bounced' THEN COALESCE(i.bounce_quantity, 0) * COALESCE(sac.avg_unit_cost, 0) ELSE 0 END) as financial_impact,
    #         ROUND(
    #             (COUNT(CASE WHEN i.bounce_status = 'Bounced' THEN 1 END) * 100.0) / NULLIF(COUNT(*), 0), 
    #             2
    #         ) as bounce_rate_percentage
    #     FROM inventory i
    #     JOIN skus s ON i.sku_id = s.sku_id
    #     LEFT JOIN sku_avg_cost sac ON i.sku_id = sac.sku_id
    #     WHERE i.bounce_status IS NOT NULL
    #     GROUP BY s.sku_id, s.sku_name, s.category
    #     ORDER BY total_bounced_quantity DESC;
    #     """

    #     bounce_sku_df = execute_query_cached(bounce_sku_query)

    #     if not bounce_sku_df.empty:
    #         col1, col2 = st.columns(2)
        
    #         with col1:
    #             fig_sku_bounce = create_bar_chart(
    #                 bounce_sku_df.head(15), 
    #                 'sku_name', 
    #                 'total_bounced_quantity', 
    #                 'Top 15 SKUs by Bounce Quantity'
    #             )
    #             fig_sku_bounce.update_layout(xaxis=dict(tickangle=45))
    #             st.plotly_chart(fig_sku_bounce, use_container_width=True)
        
    #         with col2:
    #             fig_sku_financial_impact = create_bar_chart(
    #                 bounce_sku_df.head(15), 
    #                 'sku_name', 
    #                 'financial_impact', 
    #                 'Financial Impact of Bounced SKUs'
    #             )
    #             fig_sku_financial_impact.update_layout(xaxis=dict(tickangle=45))
    #             st.plotly_chart(fig_sku_financial_impact, use_container_width=True)
                                            
    #         # Detailed bounce data table
    #         with st.expander("Detailed Bounce Analysis"):
    #             st.dataframe(
    #                 bounce_sku_df.style.format({
    #                     'total_bounced_quantity': '{:,.0f}',
    #                     'financial_impact': '₹{:,.2f}',
    #                     'bounce_rate_percentage': '{:.2f}%'
    #                 }),
    #                 use_container_width=True
    #             )
    #     else:
    #         st.warning("No bounce data found for SKUs or bounce_status column may not exist in the inventory table")

    # Inventory Turnover Analysis
    st.markdown('<h2 class="section-header">Inventory Turnover Analysis</h2>', unsafe_allow_html=True)

    turnover_dept_tab, turnover_sku_tab = st.tabs(["Department Level", "SKU Level"])

    with turnover_dept_tab:
        turnover_dept_query = f"""
            WITH transactions_all AS (
                SELECT * FROM transactions_2023
                UNION ALL
                SELECT * FROM transactions_2024
            ),
            dept_inventory AS (
                SELECT
                    t.dept_id,
                    i.sku_id,
                    i.hospital_id,
                    i.opening_stock_h1_2023,
                    (i.stock_in_h1_2023 + i.stock_in_h2_2023 + i.stock_in_h1_2024 + i.stock_in_h2_2024) as total_stock_in,
                    (i.stock_out_h1_2023 + i.stock_out_h2_2023 + i.stock_out_h1_2024 + i.stock_out_h2_2024) as total_stock_out,
                    i.current_stock
                FROM inventory i
                JOIN transactions_all t ON i.sku_id = t.sku_id AND i.hospital_id = t.hospital_id
                JOIN skus s ON i.sku_id = s.sku_id
                WHERE 1=1 {category_filter}
                GROUP BY t.dept_id, i.sku_id, i.hospital_id, i.opening_stock_h1_2023,
                        i.stock_in_h1_2023, i.stock_in_h2_2023, i.stock_in_h1_2024, i.stock_in_h2_2024,
                        i.stock_out_h1_2023, i.stock_out_h2_2023, i.stock_out_h1_2024, i.stock_out_h2_2024,
                        i.current_stock
            )
            SELECT
                dep.dept_id,
                dep.dept_name,
                SUM(di.opening_stock_h1_2023) as total_opening_stock,
                SUM(di.total_stock_in) as total_ordered_quantity,
                SUM(di.total_stock_out) as total_consumed_quantity,
                SUM(di.current_stock) as total_current_stock,
                SUM((di.opening_stock_h1_2023 + di.total_stock_in) / 2.0) as average_inventory,
                CASE
                    WHEN SUM((di.opening_stock_h1_2023 + di.total_stock_in) / 2.0) > 0
                    THEN SUM(di.total_stock_out) / SUM((di.opening_stock_h1_2023 + di.total_stock_in) / 2.0)
                    ELSE 0
                END as inventory_turnover_ratio
            FROM dept_inventory di
            JOIN departments dep ON di.dept_id = dep.dept_id AND di.hospital_id = dep.hospital_id
            WHERE 1=1 {dept_filter}
            GROUP BY dep.dept_id, dep.dept_name
            ORDER BY inventory_turnover_ratio DESC;
            """
        
        turnover_dept_df = execute_query_cached(turnover_dept_query)
        
        if not turnover_dept_df.empty:
            col1, col2 = st.columns(2)
        
            with col1:
                fig_turnover_ratio = create_bar_chart(
                    turnover_dept_df.head(10), 
                    'dept_name', 
                    'inventory_turnover_ratio', 
                    'Inventory Turnover Ratio by Department'
                )
                create_chart_container(fig_turnover_ratio, "🔄 Inventory Turnover Ratio by Department",
                                    "Efficiency of inventory usage by department")
        
            with col2:
                fig_current_stock = create_bar_chart(
                    turnover_dept_df.head(10), 
                    'dept_name', 
                    'total_current_stock', 
                    'Current Stock by Department'
                )
                create_chart_container(fig_current_stock, "📦 Current Stock by Department",
                                    "Current inventory levels by department")

    with turnover_sku_tab:
        turnover_sku_query = """
        SELECT
            s.sku_id,
            s.sku_name,
            s.category,
            SUM(i.opening_stock_h1_2023) as total_opening_stock,
            SUM(i.stock_in_h1_2023 + i.stock_in_h2_2023 + i.stock_in_h1_2024 + i.stock_in_h2_2024) as total_ordered_quantity,
            SUM(i.stock_out_h1_2023 + i.stock_out_h2_2023 + i.stock_out_h1_2024 + i.stock_out_h2_2024) as total_consumed_quantity,
            SUM(i.current_stock) as total_current_stock,
            SUM((i.opening_stock_h1_2023 + i.stock_in_h1_2023 + i.stock_in_h2_2023 + i.stock_in_h1_2024 + i.stock_in_h2_2024) / 2.0) as average_inventory,
            CASE
                WHEN SUM((i.opening_stock_h1_2023 + i.stock_in_h1_2023 + i.stock_in_h2_2023 + i.stock_in_h1_2024 + i.stock_in_h2_2024) / 2.0) > 0
                THEN SUM(i.stock_out_h1_2023 + i.stock_out_h2_2023 + i.stock_out_h1_2024 + i.stock_out_h2_2024) / 
                    SUM((i.opening_stock_h1_2023 + i.stock_in_h1_2023 + i.stock_in_h2_2023 + i.stock_in_h1_2024 + i.stock_in_h2_2024) / 2.0)
                ELSE 0
            END as inventory_turnover_ratio
        FROM inventory i
        JOIN skus s ON i.sku_id = s.sku_id
        GROUP BY s.sku_id, s.sku_name, s.category
        ORDER BY inventory_turnover_ratio DESC;
        """
        
        turnover_sku_df = execute_query_cached(turnover_sku_query)
        
        if not turnover_sku_df.empty:
            fig_sku_turnover = create_bar_chart(
                turnover_sku_df.head(20), 
                'sku_name', 
                'inventory_turnover_ratio', 
                'Top 20 SKUs by Inventory Turnover'
            )
            fig_sku_turnover.update_layout(xaxis=dict(tickangle=45))
            create_chart_container(fig_sku_turnover, "🔄 Top 20 SKUs by Inventory Turnover",
                                "Efficiency of inventory usage by SKU")

    # Formulary Adherence Analysis
    st.markdown('<h2 class="section-header">Formulary Adherence Analysis</h2>', unsafe_allow_html=True)

    formulary_dept_tab, formulary_physician_tab = st.tabs(["Department Level", "Physician Level"])

    with formulary_dept_tab:
        formulary_dept_query = f"""
            WITH transactions_all AS (
                SELECT * FROM transactions_2023
                UNION ALL
                SELECT * FROM transactions_2024
            )
            SELECT
                dep.dept_id,
                dep.dept_name,
                COUNT(*) as total_transactions,
                COUNT(CASE WHEN s.formulary_status = 'Formulary' THEN 1 END) as formulary_transactions,
                COUNT(CASE WHEN s.formulary_status = 'Non-Formulary' THEN 1 END) as non_formulary_transactions,
                CASE
                    WHEN COUNT(*) > 0
                    THEN (COUNT(CASE WHEN s.formulary_status = 'Formulary' THEN 1 END) * 100.0) / COUNT(*)
                    ELSE 0
                END as formulary_adherence_rate,
                SUM(CASE WHEN s.formulary_status = 'Formulary' THEN t.total_cost ELSE 0 END) as formulary_revenue,
                SUM(CASE WHEN s.formulary_status = 'Non-Formulary' THEN t.total_cost ELSE 0 END) as non_formulary_revenue
            FROM transactions_all t
            JOIN departments dep ON t.dept_id = dep.dept_id AND t.hospital_id = dep.hospital_id
            JOIN skus s ON t.sku_id = s.sku_id
            WHERE s.formulary_status IS NOT NULL {dept_filter} {category_filter}
            GROUP BY dep.dept_id, dep.dept_name
            ORDER BY non_formulary_revenue DESC;
            """
        
        formulary_dept_df = execute_query_cached(formulary_dept_query)
        
        if not formulary_dept_df.empty:
            col1, col2 = st.columns(2)
        
            with col1:
                fig_adherence_rate = create_bar_chart(
                    formulary_dept_df.head(10), 
                    'dept_name', 
                    'formulary_adherence_rate', 
                    'Formulary Adherence Rate by Department (%)'
                )
                create_chart_container(fig_adherence_rate, "📋 Formulary Adherence Rate by Department (%)",
                                    "Compliance with formulary guidelines by department")
        
            with col2:
                fig_non_formulary_impact = create_bar_chart(
                    formulary_dept_df.head(10), 
                    'dept_name', 
                    'non_formulary_revenue', 
                    'Non-Formulary Revenue Impact'
                )
                create_chart_container(fig_non_formulary_impact, "💸 Non-Formulary Revenue Impact",
                                    "Revenue from non-formulary items by department")

    with formulary_physician_tab:
        formulary_physician_query = f"""
            WITH transactions_all AS (
                SELECT * FROM transactions_2023
                UNION ALL
                SELECT * FROM transactions_2024
            )
            SELECT
                p.physician_id,
                p.physician_name,
                p.specialty,
                dep.dept_name,
                COUNT(*) as total_transactions,
                COUNT(CASE WHEN s.formulary_status = 'Formulary' THEN 1 END) as formulary_transactions,
                COUNT(CASE WHEN s.formulary_status = 'Non-Formulary' THEN 1 END) as non_formulary_transactions,
                CASE
                    WHEN COUNT(*) > 0
                    THEN (COUNT(CASE WHEN s.formulary_status = 'Formulary' THEN 1 END) * 100.0) / COUNT(*)
                    ELSE 0
                END as formulary_adherence_rate,
                SUM(CASE WHEN s.formulary_status = 'Non-Formulary' THEN t.total_cost ELSE 0 END) as non_formulary_revenue_impact
            FROM transactions_all t
            JOIN physicians p ON t.physician_id = p.physician_id AND t.hospital_id = p.hospital_id
            JOIN departments dep ON p.primary_dept_id = dep.dept_id AND p.hospital_id = dep.hospital_id
            JOIN skus s ON t.sku_id = s.sku_id
            WHERE s.formulary_status IS NOT NULL {dept_filter} {category_filter}
            GROUP BY p.physician_id, p.physician_name, p.specialty, dep.dept_name
            ORDER BY non_formulary_revenue_impact DESC;
            """
        
        formulary_physician_df = execute_query_cached(formulary_physician_query)
        
        if not formulary_physician_df.empty:
            # Show top physicians with highest non-formulary impact
            fig_physician_impact = create_bar_chart(
                formulary_physician_df.head(15), 
                'physician_name', 
                'non_formulary_revenue_impact', 
                'Top 15 Physicians by Non-Formulary Impact'
            )
            fig_physician_impact.update_layout(xaxis=dict(tickangle=45))
            create_chart_container(fig_physician_impact, "👩‍⚕️ Top 15 Physicians by Non-Formulary Impact",
                                "Non-formulary revenue impact by physician")

    # Outpatient to Prescription Conversion
    st.markdown('<h2 class="section-header">Outpatient to Prescription Conversion</h2>', unsafe_allow_html=True)

    conversion_query = f"""
        WITH transactions_all AS (
    SELECT * FROM transactions_2023
    UNION ALL
    SELECT * FROM transactions_2024
),
filtered_transactions AS (
    SELECT t.*
    FROM transactions_all t
    JOIN departments d ON t.dept_id = d.dept_id AND t.hospital_id = d.hospital_id
    JOIN skus s ON t.sku_id = s.sku_id
    WHERE 1=1 
        AND d.dept_name IN ('Cardiology','Emergency Medicine','General Surgery','ICU','Internal Medicine','Nephrology','Obstetrics & Gynecology','Oncology','Orthopedics','Pediatrics')
        AND s.category IN ('Consumables','Pharmacy')
),
outpatient_analysis AS (
    SELECT
        COUNT(DISTINCT CASE WHEN EXISTS (
            SELECT 1 FROM filtered_transactions ft WHERE ft.patient_id = p.patient_id
        ) THEN p.patient_id END) as total_outpatients,
        COUNT(DISTINCT CASE WHEN EXISTS (
            SELECT 1 FROM filtered_transactions ft 
            WHERE ft.patient_id = p.patient_id AND ft.transaction_type = 'Prescription'
        ) THEN p.patient_id END) as outpatients_with_prescriptions
    FROM patients p
    WHERE p.patient_type = 'Outpatient'
)
SELECT
    total_outpatients,
    outpatients_with_prescriptions,
    CASE
        WHEN total_outpatients > 0
        THEN (outpatients_with_prescriptions * 100.0) / total_outpatients
        ELSE 0
    END as op_to_prescription_conversion_rate
FROM outpatient_analysis;
        """

    conversion_df = execute_query_cached(conversion_query)

    if not conversion_df.empty:
        col1, col2, col3 = st.columns(3)
    
        with col1:
            total_op = f"{conversion_df.iloc[0]['total_outpatients']:,}" if conversion_df.iloc[0]['total_outpatients'] else "0"
            create_metric_card("Total Outpatients", total_op)
    
        with col2:
            op_with_rx = f"{conversion_df.iloc[0]['outpatients_with_prescriptions']:,}" if conversion_df.iloc[0]['outpatients_with_prescriptions'] else "0"
            create_metric_card("Outpatients with Prescriptions", op_with_rx)
    
        with col3:
            conversion_rate = f"{conversion_df.iloc[0]['op_to_prescription_conversion_rate']:.1f}%" if conversion_df.iloc[0]['op_to_prescription_conversion_rate'] else "0%"
            create_metric_card("Conversion Rate", conversion_rate)

    # Expiry Items Analysis
    st.markdown('<h2 class="section-header">Expiry Items Analysis</h2>', unsafe_allow_html=True)

    expiry_query = f"""
            SELECT
                d.dept_name,
                s.sku_name,
                s.category,
                i.current_stock,
                i.expiry_date,
                s.standard_cost,
                (i.current_stock * s.standard_cost) as inventory_value,
                EXTRACT(days FROM (i.expiry_date - CURRENT_DATE)) as days_to_expiry,
                CASE
                    WHEN i.expiry_date <= CURRENT_DATE + INTERVAL '365 days' THEN '0-12 months'
                    WHEN i.expiry_date <= CURRENT_DATE + INTERVAL '730 days' THEN '1-2 years'
                    WHEN i.expiry_date <= CURRENT_DATE + INTERVAL '1095 days' THEN '2-3 years'
                    ELSE '3+ years'
                END as expiry_timeframe
            FROM inventory i
            JOIN departments d ON i.hospital_id = d.hospital_id
            JOIN skus s ON i.sku_id = s.sku_id
            WHERE i.expiry_date IS NOT NULL
                AND i.current_stock > 0
                {dept_filter.replace('dep.dept_name', 'd.dept_name') if dept_filter else ''}
                {category_filter}
            ORDER BY i.expiry_date;
                """

    expiry_df = execute_query_cached(expiry_query)

    if not expiry_df.empty:
        col1, col2 = st.columns(2)
    
        with col1:
            # Group by expiry timeframe
            expiry_summary = expiry_df.groupby('expiry_timeframe')['inventory_value'].sum().reset_index()
            fig_expiry_value = create_pie_chart(
                expiry_summary, 
                'inventory_value', 
                'expiry_timeframe', 
                'Inventory Value by Expiry Timeframe'
            )
            st.plotly_chart(fig_expiry_value, use_container_width=True)
    
        with col2:
            # Items expiring in next 12 months
            critical_expiry = expiry_df[expiry_df['expiry_timeframe'] == '0-12 months'].nlargest(10, 'inventory_value')
            if not critical_expiry.empty:
                fig_critical = create_bar_chart(
                    critical_expiry, 
                    'sku_name', 
                    'inventory_value', 
                    'Top 10 Items by Value (Expiring in 12 months)'
                )
                fig_critical.update_layout(xaxis=dict(tickangle=45))
                st.plotly_chart(fig_critical, use_container_width=True)

    # Consumables Analytics
    st.markdown('<h2 class="section-header">Consumables Analytics</h2>', unsafe_allow_html=True)

    consumables_value_tab, consumables_quantity_tab, consumables_patient_tab, consumables_turnover_tab = st.tabs([
        "Value Analysis", "Quantity Analysis", "Per Patient Usage", "Turnover Analysis"
    ])

    with consumables_value_tab:
        consumables_value_query = f"""
            WITH transactions_all AS (
                SELECT * FROM transactions_2023
                UNION ALL
                SELECT * FROM transactions_2024
            )
            SELECT
                dep.dept_id,
                dep.dept_name,
                SUM(t.total_cost) as total_consumables_value
            FROM transactions_all t
            JOIN departments dep ON t.dept_id = dep.dept_id AND t.hospital_id = dep.hospital_id
            JOIN skus s ON t.sku_id = s.sku_id
            WHERE s.category = 'Consumables' {dept_filter} {category_filter}
            GROUP BY dep.dept_id, dep.dept_name
            ORDER BY total_consumables_value DESC;
            """
        
        consumables_value_df = execute_query_cached(consumables_value_query)
        
        if not consumables_value_df.empty:
            fig_consumables_value = create_bar_chart(
                consumables_value_df.head(10), 
                'dept_name', 
                'total_consumables_value', 
                'Consumables Value by Department'
            )
            create_chart_container(fig_consumables_value, "💰 Consumables Value by Department",
                                "Total value of consumables used by department")

    with consumables_quantity_tab:
        consumables_quantity_query = f"""
        WITH transactions_all AS (
            SELECT * FROM transactions_2023
            UNION ALL
            SELECT * FROM transactions_2024
        )
        SELECT
            s.sku_id,
            s.sku_name,
            s.sub_category,
            s.unit_of_measure,
            SUM(t.quantity_consumed) as total_consumables_quantity
        FROM transactions_all t
        JOIN skus s ON t.sku_id = s.sku_id
        JOIN departments dep ON t.dept_id = dep.dept_id AND t.hospital_id = dep.hospital_id
        WHERE s.category = 'Consumables' {dept_filter} {category_filter}
        GROUP BY s.sku_id, s.sku_name, s.sub_category, s.unit_of_measure
        ORDER BY total_consumables_quantity DESC;
        """
        
        consumables_quantity_df = execute_query_cached(consumables_quantity_query)
        
        if not consumables_quantity_df.empty:
            fig_consumables_qty = create_bar_chart(
                consumables_quantity_df.head(15), 
                'sku_name', 
                'total_consumables_quantity', 
                'Top 15 Consumables by Quantity'
            )
            fig_consumables_qty.update_layout(xaxis=dict(tickangle=45))
            create_chart_container(fig_consumables_qty, "📦 Top 15 Consumables by Quantity",
                                "Most used consumables by quantity")

    with consumables_patient_tab:
        consumables_patient_query = f"""
        WITH transactions_all AS (
            SELECT * FROM transactions_2023
            UNION ALL
            SELECT * FROM transactions_2024
        )
        SELECT
            dep.dept_id,
            dep.dept_name,
            COUNT(DISTINCT t.patient_id) as total_patients,
            SUM(t.quantity_consumed) as total_consumables_quantity,
            SUM(t.total_cost) as total_consumables_value,
            CASE
                WHEN COUNT(DISTINCT t.patient_id) > 0
                THEN SUM(t.quantity_consumed) / COUNT(DISTINCT t.patient_id)
                ELSE 0
            END as consumables_quantity_per_patient,
            CASE
                WHEN COUNT(DISTINCT t.patient_id) > 0
                THEN SUM(t.total_cost) / COUNT(DISTINCT t.patient_id)
                ELSE 0
            END as consumables_value_per_patient
        FROM transactions_all t
        JOIN departments dep ON t.dept_id = dep.dept_id AND t.hospital_id = dep.hospital_id
        JOIN skus s ON t.sku_id = s.sku_id
        WHERE s.category = 'Consumables' AND t.patient_id IS NOT NULL {dept_filter} {category_filter}
        GROUP BY dep.dept_id, dep.dept_name
        ORDER BY consumables_value_per_patient DESC;
        """
        
        consumables_patient_df = execute_query_cached(consumables_patient_query)
        
        if not consumables_patient_df.empty:
            col1, col2 = st.columns(2)
        
            with col1:
                fig_consumables_per_patient = create_bar_chart(
                    consumables_patient_df.head(10), 
                    'dept_name', 
                    'consumables_value_per_patient', 
                    'Consumables Value per Patient'
                )
                create_chart_container(fig_consumables_per_patient, "💸 Consumables Value per Patient",
                                    "Average consumables cost per patient by department")
        
            with col2:
                fig_consumables_qty_per_patient = create_bar_chart(
                    consumables_patient_df.head(10), 
                    'dept_name', 
                    'consumables_quantity_per_patient', 
                    'Consumables Quantity per Patient'
                )
                create_chart_container(fig_consumables_qty_per_patient, "📏 Consumables Quantity per Patient",
                                    "Average consumables quantity per patient by department")

    with consumables_turnover_tab:
        consumables_turnover_query = f"""
        WITH transactions_all AS (
            SELECT * FROM transactions_2023
            UNION ALL
            SELECT * FROM transactions_2024
        ),
        dept_consumables_inventory AS (
            SELECT
                t.dept_id,
                i.sku_id,
                i.hospital_id,
                i.opening_stock_h1_2023,
                (i.stock_in_h1_2023 + i.stock_in_h2_2023 + i.stock_in_h1_2024 + i.stock_in_h2_2024) as total_stock_in,
                (i.stock_out_h1_2023 + i.stock_out_h2_2023 + i.stock_out_h1_2024 + i.stock_out_h2_2024) as total_stock_out,
                i.current_stock
            FROM inventory i
            JOIN skus s ON i.sku_id = s.sku_id
            JOIN transactions_all t ON i.sku_id = t.sku_id AND i.hospital_id = t.hospital_id
            WHERE s.category = 'Consumables' {category_filter}
            GROUP BY t.dept_id, i.sku_id, i.hospital_id, i.opening_stock_h1_2023,
                    i.stock_in_h1_2023, i.stock_in_h2_2023, i.stock_in_h1_2024, i.stock_in_h2_2024,
                    i.stock_out_h1_2023, i.stock_out_h2_2023, i.stock_out_h1_2024, i.stock_out_h2_2024,
                    i.current_stock
        )
        SELECT
            dep.dept_id,
            dep.dept_name,
            SUM(dci.total_stock_out) as total_consumption,
            SUM((dci.opening_stock_h1_2023 + dci.total_stock_in) / 2.0) as average_inventory,
            SUM(dci.current_stock) as current_stock,
            CASE
                WHEN SUM((dci.opening_stock_h1_2023 + dci.total_stock_in) / 2.0) > 0
                THEN SUM(dci.total_stock_out) / SUM((dci.opening_stock_h1_2023 + dci.total_stock_in) / 2.0)
                ELSE 0
            END as consumables_turnover_ratio
        FROM dept_consumables_inventory dci
        JOIN departments dep ON dci.dept_id = dep.dept_id AND dci.hospital_id = dep.hospital_id
        WHERE 1=1 {dept_filter}
        GROUP BY dep.dept_id, dep.dept_name
        ORDER BY consumables_turnover_ratio DESC;
        """
        
        consumables_turnover_df = execute_query_cached(consumables_turnover_query)
        
        if not consumables_turnover_df.empty:
            col1, col2 = st.columns(2)
        
            with col1:
                fig_consumables_turnover = create_bar_chart(
                    consumables_turnover_df.head(10), 
                    'dept_name', 
                    'consumables_turnover_ratio', 
                    'Consumables Turnover Ratio by Department'
                )
                create_chart_container(fig_consumables_turnover, "🔄 Consumables Turnover Ratio by Department",
                                    "Efficiency of consumables inventory usage")
        
            with col2:
                fig_consumables_consumption = create_bar_chart(
                    consumables_turnover_df.head(10), 
                    'dept_name', 
                    'total_consumption', 
                    'Total Consumables Consumption'
                )
                create_chart_container(fig_consumables_consumption, "📦 Total Consumables Consumption",
                                    "Total consumables used by department")

    # Enhanced Generic vs Brand Analysis
    st.markdown('<h2 class="section-header">Enhanced Generic vs Brand Mix Analysis</h2>', unsafe_allow_html=True)

    enhanced_brand_dept_tab, enhanced_brand_sku_tab = st.tabs(["Department Level", "SKU Level"])

    with enhanced_brand_dept_tab:
        if selected_depts:
            dept_list = "'" + "','".join(selected_depts) + "'"
            dept_filter = f"AND d.dept_name IN ({dept_list})"
        else:
            dept_filter = ""
        
        # Enhanced Brand Department Level
        enhanced_brand_dept_query = f"""
        WITH transactions AS (
            SELECT * FROM transactions_2023
            UNION ALL
            SELECT * FROM transactions_2024
        )
        SELECT
            d.dept_name,
            s.brand_generic_flag,
            COUNT(DISTINCT s.sku_id) AS sku_count,
            SUM(t.quantity_consumed) AS total_volume,
            SUM(t.total_cost) AS total_revenue,
            AVG(t.unit_cost) AS avg_unit_cost,
            ROUND(
                (SUM(t.total_cost) / SUM(SUM(t.total_cost)) OVER (PARTITION BY d.dept_id) * 100)::numeric,
                2
            ) AS revenue_percentage_in_dept
        FROM transactions t
        JOIN departments d ON t.dept_id = d.dept_id
        JOIN skus s ON t.sku_id = s.sku_id
        WHERE t.total_cost IS NOT NULL
            AND t.quantity_consumed > 0
            AND s.brand_generic_flag IS NOT NULL
            {dept_filter}
            {category_filter}
        GROUP BY d.dept_name, d.dept_id, s.brand_generic_flag
        ORDER BY d.dept_name, s.brand_generic_flag;
        """
    
        enhanced_brand_dept_df = execute_query_cached(enhanced_brand_dept_query)
    
        if not enhanced_brand_dept_df.empty:
            # Create pivot table for better visualization
            pivot_df = enhanced_brand_dept_df.pivot(
                index='dept_name',
                columns='brand_generic_flag',
                values='revenue_percentage_in_dept'
            ).fillna(0)
        
            if not pivot_df.empty:
                # Stacked bar chart showing brand vs generic mix
                fig_brand_mix = go.Figure()
            
                if 'Brand' in pivot_df.columns:
                    fig_brand_mix.add_trace(go.Bar(
                        name='Brand',
                        x=pivot_df.index,
                        y=pivot_df['Brand'],
                        marker_color=COLORS['primary']
                    ))
            
                if 'Generic' in pivot_df.columns:
                    fig_brand_mix.add_trace(go.Bar(
                        name='Generic',
                        x=pivot_df.index,
                        y=pivot_df['Generic'],
                        marker_color=COLORS['secondary']
                    ))
            
                fig_brand_mix.update_layout(
                    title='Brand vs Generic Mix by Department (%)',
                    barmode='stack',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Arial, sans-serif", size=12),
                    xaxis=dict(tickangle=45)
                )
                st.plotly_chart(fig_brand_mix, use_container_width=True)
        
            # Summary metrics
            brand_summary = enhanced_brand_dept_df.groupby('brand_generic_flag').agg({
                'total_revenue': 'sum',
                'sku_count': 'sum',
                'total_volume': 'sum'
            }).reset_index()
        
            if not brand_summary.empty:
                col1, col2 = st.columns(2)
            
                with col1:
                    fig_brand_revenue = create_pie_chart(
                        brand_summary, 
                        'total_revenue', 
                        'brand_generic_flag', 
                        'Overall Revenue: Brand vs Generic'
                    )
                    st.plotly_chart(fig_brand_revenue, use_container_width=True)
            
                with col2:
                    fig_brand_volume = create_pie_chart(
                        brand_summary, 
                        'total_volume', 
                        'brand_generic_flag', 
                        'Overall Volume: Brand vs Generic'
                    )
                    st.plotly_chart(fig_brand_volume, use_container_width=True)

    with enhanced_brand_sku_tab:
        if selected_categories:
            category_list = "'" + "','".join(selected_categories) + "'"
            category_filter = f"AND s.category IN ({category_list})"
        else:
            category_filter = ""
        
        enhanced_brand_sku_query = f"""
        WITH transactions AS (
            SELECT * FROM transactions_2023
            UNION ALL
            SELECT * FROM transactions_2024
        )
        SELECT
            s.brand_generic_flag,
            s.sku_name,
            s.sku_id,
            s.category,
            SUM(t.quantity_consumed) AS total_volume,
            SUM(t.total_cost) AS total_revenue,
            AVG(t.unit_cost) AS avg_unit_cost
        FROM transactions t
        JOIN skus s ON t.sku_id = s.sku_id
        WHERE t.total_cost IS NOT NULL
            AND t.quantity_consumed > 0
            AND s.brand_generic_flag IS NOT NULL
            {category_filter}
        GROUP BY s.brand_generic_flag, s.sku_name, s.sku_id, s.category
        ORDER BY s.brand_generic_flag, total_revenue DESC;
        """
    
        enhanced_brand_sku_df = execute_query_cached(enhanced_brand_sku_query)
    
        if not enhanced_brand_sku_df.empty:
            # Show top SKUs by brand/generic
            brand_selection = st.selectbox(
                "Select Brand/Generic:", 
                options=['All'] + enhanced_brand_sku_df['brand_generic_flag'].unique().tolist()
            )
        
            if brand_selection != 'All':
                filtered_sku_df = enhanced_brand_sku_df[enhanced_brand_sku_df['brand_generic_flag'] == brand_selection].head(15)
            else:
                filtered_sku_df = enhanced_brand_sku_df.head(20)
        
            fig_top_skus = create_bar_chart(
                filtered_sku_df, 
                'sku_name', 
                'total_revenue', 
                f'Top SKUs by Revenue - {brand_selection}',
                'brand_generic_flag'
            )
            fig_top_skus.update_layout(xaxis=dict(tickangle=45))
            st.plotly_chart(fig_top_skus, use_container_width=True)

    # Average Consumable Usage Analysis
    # Average Consumable Usage Analysis
    st.markdown('<h2 class="section-header">Average Consumable Usage Patterns</h2>', unsafe_allow_html=True)

    avg_usage_query = f"""
        WITH transactions_all AS (
            SELECT * FROM transactions_2023
            UNION ALL
            SELECT * FROM transactions_2024
        )
        SELECT
            dep.dept_id,
            dep.dept_name,
            COUNT(DISTINCT t.transaction_id) as total_transactions,
            COUNT(DISTINCT t.patient_id) as total_patients,
            COUNT(DISTINCT t.sku_id) as unique_consumables_used,
            SUM(t.quantity_consumed) as total_quantity_used,
            SUM(t.total_cost) as total_value_used,
            CASE
                WHEN COUNT(DISTINCT t.transaction_id) > 0
                THEN SUM(t.quantity_consumed) / COUNT(DISTINCT t.transaction_id)
                ELSE 0
            END as avg_quantity_per_transaction,
            CASE
                WHEN COUNT(DISTINCT t.patient_id) > 0
                THEN SUM(t.quantity_consumed) / COUNT(DISTINCT t.patient_id)
                ELSE 0
            END as avg_quantity_per_patient,
            CASE
                WHEN COUNT(DISTINCT t.patient_id) > 0
                THEN SUM(t.total_cost) / COUNT(DISTINCT t.patient_id)
                ELSE 0
            END as avg_value_per_patient
        FROM transactions_all t
        JOIN departments dep ON t.dept_id = dep.dept_id AND t.hospital_id = dep.hospital_id
        JOIN skus s ON t.sku_id = s.sku_id
        WHERE s.category = 'Consumables' {dept_filter.replace('d.dept_name', 'dep.dept_name') if dept_filter else ''} {category_filter}
        GROUP BY dep.dept_id, dep.dept_name
        ORDER BY total_quantity_used DESC;
"""

    avg_usage_df = execute_query_cached(avg_usage_query)

    if not avg_usage_df.empty:
        col1, col2 = st.columns(2)
    
        with col1:
            fig_avg_per_transaction = create_bar_chart(
                avg_usage_df.head(10), 
                'dept_name', 
                'avg_quantity_per_transaction', 
                'Average Consumables per Transaction'
            )
            st.plotly_chart(fig_avg_per_transaction, use_container_width=True)
    
        with col2:
            fig_unique_consumables = create_bar_chart(
                avg_usage_df.head(10), 
                'dept_name', 
                'unique_consumables_used', 
                'Unique Consumables Used by Department'
            )
            st.plotly_chart(fig_unique_consumables, use_container_width=True)
    
        # Summary table with detailed metrics
        with st.expander("Detailed Usage Metrics"):
            st.dataframe(
                avg_usage_df.style.format({
                    'total_quantity_used': '{:,.0f}',
                    'total_value_used': '₹{:,.0f}',
                    'avg_quantity_per_transaction': '{:.2f}',
                    'avg_quantity_per_patient': '{:.2f}',
                    'avg_value_per_patient': '₹{:.2f}'
                }),
                use_container_width=True
            )

# Run the dashboard
if __name__ == "__main__":
    main()
