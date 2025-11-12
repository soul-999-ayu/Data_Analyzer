import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# --- NEW IMPORTS ---
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- Page Setup ---
st.set_page_config(
    page_title="Data Analysis Hub",
    page_icon="📊",
    layout="wide"
)

# --- Analysis Logic Functions ---

def analyze_cricket(df):
    """
    Performs analysis on a DataFrame with expected cricket columns.
    """
    try:
        # 1. Batting Avg
        df['Batting_Avg'] = 0.0
        avg_mask = df['Dismissals'] > 0
        df.loc[avg_mask, 'Batting_Avg'] = df.loc[avg_mask, 'Runs_Scored'] / df.loc[avg_mask, 'Dismissals']
        
        # 2. Batting Strike Rate
        df['Batting_Strike_Rate'] = 0.0
        batting_mask = df['Balls_Faced'] > 0
        df.loc[batting_mask, 'Batting_Strike_Rate'] = \
            (df.loc[batting_mask, 'Runs_Scored'] / df.loc[batting_mask, 'Balls_Faced']) * 100

        # 3. Bowling Avg
        df['Bowling_Avg'] = 0.0
        b_avg_mask = df['Wickets_Taken'] > 0
        df.loc[b_avg_mask, 'Bowling_Avg'] = df.loc[b_avg_mask, 'Runs_Conceded'] / df.loc[b_avg_mask, 'Wickets_Taken']

        # 4. Bowling Economy
        df['Bowling_Economy'] = 0.0
        bowling_mask = df['Balls_Bowled'] > 0
        df.loc[bowling_mask, 'Bowling_Economy'] = \
            df.loc[bowling_mask, 'Runs_Conceded'] / (df.loc[bowling_mask, 'Balls_Bowled'] / 6)
        
        # 5. Boundary %
        df['Boundary_%'] = 0.0
        # Add a mask to prevent division by zero if Runs_Scored is 0
        runs_mask = (df['Runs_Scored'] > 0) & batting_mask
        df.loc[runs_mask, 'Boundary_%'] = \
            ((df.loc[runs_mask, 'Fours'] * 4 + df.loc[runs_mask, 'Sixes'] * 6) / 
             df.loc[runs_mask, 'Runs_Scored']) * 100

        # 6. Final Cleaning
        df = df.round(2)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.fillna(0)
        return df
    except KeyError as e:
        raise ValueError(f"Missing expected cricket column: {e}. Please check your file.")
    except Exception as e:
        raise Exception(f"An error occurred during cricket analysis: {e}")

def analyze_movies(df):
    """
    Performs analysis on a DataFrame with expected movie columns.
    """
    try:
        # 1. Calculate Profit
        df['Profit'] = df['Revenue'] - df['Budget']
        # 2. Calculate Profit Margin
        df['Profit_Margin_%'] = 0.0
        budget_mask = df['Budget'] > 0
        df.loc[budget_mask, 'Profit_Margin_%'] = (df.loc[budget_mask, 'Profit'] / df.loc[budget_mask, 'Budget']) * 100
        # 3. Final Cleaning
        df = df.round(2)
        df.fillna(0, inplace=True)
        return df
    except KeyError as e:
        raise ValueError(f"Missing expected movie column: {e}. Please check your file.")
    except Exception as e:
        raise Exception(f"An error occurred during movie analysis: {e}")

def analyze_video_games(df):
    """
    Performs analysis on a DataFrame with expected video game columns.
    """
    try:
        # 1. Create Sales Categories
        bins = [0, 1, 5, 10, 20, 100] # e.g., <1M, 1-5M, 5-10M, 10-20M, >20M
        labels = ['<1M (Niche)', '1-5M (Standard)', '5-10M (Hit)', '10-20M (Blockbuster)', '>20M (Mega-Blockbuster)']
        df['Sales_Category'] = pd.cut(df['Global_Sales_M'], bins=bins, labels=labels, right=False)
        # 2. Final Cleaning
        df['Sales_Category'] = df['Sales_Category'].cat.add_categories('Unknown').fillna('Unknown')
        return df
    except KeyError as e:
        raise ValueError(f"Missing expected video game column: {e}. Please check your file.")
    except Exception as e:
        raise Exception(f"An error occurred during video game analysis: {e}")

# --- Plotly Template ---
# Define a dark theme for all charts
plotly_dark_template = "plotly_dark"

# -------------------------------------------------------------------
# --- MODAL DISPLAY FUNCTIONS ---
# -------------------------------------------------------------------
# --- REMOVED ALL MODAL FUNCTIONS ---

# -------------------------------------------------------------------
# --- REGRESSION ANALYSIS FUNCTIONS ---
# -------------------------------------------------------------------

def display_movie_regression(df_to_display):
    """
    Trains a linear regression model for Movies and displays the results.
    """
    st.header("🎬 Regression Analysis: Budget vs. Revenue")
    st.markdown("""
    This analysis attempts to predict a movie's **Revenue** based on its **Budget**.
    We use a simple Linear Regression model trained on 80% of your data and 
    tested on the remaining 20%.
    """)

    # Check if we have enough data to work with
    if 'Budget' not in df_to_display.columns or 'Revenue' not in df_to_display.columns:
        st.error("Error: The data must contain 'Budget' and 'Revenue' columns for this analysis.")
        return

    if df_to_display.shape[0] < 10:
        st.warning("Warning: Not enough data to perform a reliable regression (requires at least 10 data points).")
        return

    # 1. Define Features (X) and Target (y)
    model_df = df_to_display[(df_to_display['Budget'] > 0) & (df_to_display['Revenue'] > 0)]
    X = model_df[['Budget']]
    y = model_df['Revenue']

    if model_df.shape[0] < 10:
        st.warning("Warning: Not enough *valid* (non-zero) data to perform a reliable regression.")
        return

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Create and Train the Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Make Predictions on the test set
    y_pred = model.predict(X_test)

    # 5. Calculate Performance Metrics (The "Test Cases")
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.subheader("Model Performance (Test Cases)")
    st.markdown("We evaluate the model on the 20% of data it has *never* seen before.")
    
    kpi_col1, kpi_col2 = st.columns(2)
    kpi_col1.metric(
        label="R-squared (R²)", 
        value=f"{r2:.2f}",
        help="How much of the revenue is 'explained' by the budget? (1.0 is a perfect fit, 0.0 means no fit)"
    )
    kpi_col2.metric(
        label="Mean Absolute Error (MAE)", 
        value=f"${mae/1_000_000:.1f}M",
        help="On average, how 'wrong' was the model's prediction on the test set? (Lower is better)"
    )

    st.subheader("Regression Model Visualization")
    
    coef = model.coef_[0]
    intercept = model.intercept_
    st.markdown(f"**Model's Formula:** `Revenue = {coef:.2f} * Budget + ${intercept:,.0f}`")
    st.markdown(f"This means for every **$1** spent on budget, the model predicts **${coef:.2f}** in revenue.")

    fig = px.scatter(
        model_df, x='Budget', y='Revenue', 
        hover_name='Title', color='Genre',
        title="Budget vs. Revenue with Regression Line",
        template=plotly_dark_template
    )
    
    line_x = np.array([X['Budget'].min(), X['Budget'].max()])
    line_y = model.predict(line_x.reshape(-1, 1))
    
    fig.add_traces(go.Scatter(
        x=line_x, 
        y=line_y, 
        name='Regression Line', 
        line=dict(color='red', width=3, dash='dash')
    ))
    st.plotly_chart(fig, use_container_width=True)

def display_cricket_regression(df_to_display):
    """
    Trains a linear regression model for Cricket and displays the results.
    """
    st.header("🏏 Regression Analysis: Balls Faced vs. Runs Scored")
    st.markdown("""
    This analysis attempts to predict a player's **Runs Scored** based on **Balls Faced**.
    This model helps identify the average scoring rate (strike rate).
    """)

    if 'Balls_Faced' not in df_to_display.columns or 'Runs_Scored' not in df_to_display.columns:
        st.error("Error: The data must contain 'Balls_Faced' and 'Runs_Scored' columns.")
        return

    if df_to_display.shape[0] < 10:
        st.warning("Warning: Not enough data to perform a reliable regression (requires at least 10 data points).")
        return

    model_df = df_to_display[df_to_display['Balls_Faced'] > 0]
    X = model_df[['Balls_Faced']]
    y = model_df['Runs_Scored']

    if model_df.shape[0] < 10:
        st.warning("Warning: Not enough *valid* (non-zero) data to perform a reliable regression.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.subheader("Model Performance (Test Cases)")
    kpi_col1, kpi_col2 = st.columns(2)
    kpi_col1.metric(label="R-squared (R²)", value=f"{r2:.2f}",
        help="How much of the runs are 'explained' by the balls faced? (1.0 is a perfect fit)")
    kpi_col2.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.1f} Runs",
        help="On average, how 'wrong' was the model's run prediction? (Lower is better)")

    st.subheader("Regression Model Visualization")
    coef = model.coef_[0]
    intercept = model.intercept_
    st.markdown(f"**Model's Formula:** `Runs Scored = {coef:.2f} * Balls Faced + {intercept:.2f}`")
    st.markdown(f"This means for every **1 ball faced**, the model predicts **{coef:.2f} runs** (a {coef*100:.1f} strike rate).")

    fig = px.scatter(
        model_df, x='Balls_Faced', y='Runs_Scored', 
        hover_name='Player_Name', color='Role',
        title="Balls Faced vs. Runs Scored with Regression Line",
        template=plotly_dark_template
    )
    
    line_x = np.array([X['Balls_Faced'].min(), X['Balls_Faced'].max()])
    line_y = model.predict(line_x.reshape(-1, 1))
    
    fig.add_traces(go.Scatter(
        x=line_x, y=line_y, name='Regression Line', 
        line=dict(color='red', width=3, dash='dash')
    ))
    st.plotly_chart(fig, use_container_width=True)

def display_game_regression(df_to_display):
    """
    Trains a linear regression model for Video Games and displays the results.
    """
    st.header("🎮 Regression Analysis: Release Year vs. Global Sales")
    st.markdown("""
    This analysis attempts to find a trend in **Global Sales** based on the **Release Year**.
    A simple linear regression is not the best tool for this (as sales are not necessarily linear),
    but the trend line can still show if the market is generally growing or shrinking.
    """)

    if 'Release_Year' not in df_to_display.columns or 'Global_Sales_M' not in df_to_display.columns:
        st.error("Error: The data must contain 'Release_Year' and 'Global_Sales_M' columns.")
        return

    if df_to_display.shape[0] < 10:
        st.warning("Warning: Not enough data to perform a reliable regression (requires at least 10 data points).")
        return

    model_df = df_to_display[(df_to_display['Release_Year'] > 0) & (df_to_display['Global_Sales_M'] > 0)]
    X = model_df[['Release_Year']]
    y = model_df['Global_Sales_M']

    if model_df.shape[0] < 10:
        st.warning("Warning: Not enough *valid* (non-zero) data to perform a reliable regression.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.subheader("Model Performance (Test Cases)")
    kpi_col1, kpi_col2 = st.columns(2)
    kpi_col1.metric(label="R-squared (R²)", value=f"{r2:.2f}",
        help="How much of the sales are 'explained' by the year? (Note: A low R² is expected here)")
    kpi_col2.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.1f}M Sales",
        help="On average, how 'wrong' was the model's sales prediction?")

    st.subheader("Regression Model Visualization")
    coef = model.coef_[0]
    intercept = model.intercept_
    st.markdown(f"**Model's Formula:** `Global Sales = {coef:.2f} * Release Year + {intercept:.2f}`")
    st.markdown(f"A **positive** coefficient (`{coef:.2f}`) suggests that, on average, games released later tend to sell more (or less, if negative).")

    fig = px.scatter(
        model_df, x='Release_Year', y='Global_Sales_M', 
        hover_name='Title', color='Genre',
        title="Release Year vs. Global Sales with Regression Line",
        template=plotly_dark_template
    )
    
    line_x = np.array([X['Release_Year'].min(), X['Release_Year'].max()])
    line_y = model.predict(line_x.reshape(-1, 1))
    
    fig.add_traces(go.Scatter(
        x=line_x, y=line_y, name='Regression Line', 
        line=dict(color='red', width=3, dash='dash')
    ))
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------------------
# --- DASHBOARD DISPLAY FUNCTIONS ---
# -------------------------------------------------------------------

def display_cricket_dashboard(df_to_display, tab_key):
    """
    Renders the entire Cricket dashboard (KPIs, Charts, Table)
    for a given DataFrame.
    """
    # --- NEW SAFETY CHECK ---
    if df_to_display.empty:
        st.warning(f"No players found for this filter. Please adjust your data or filters.")
        return
    # --- END SAFETY CHECK ---
    
    st.header("🏏 Cricket Player Analysis")

    # --- KPIs ---
    st.subheader("Tournament Metrics")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Total Runs Scored", df_to_display['Runs_Scored'].sum())
    kpi_col2.metric("Total Wickets Taken", df_to_display['Wickets_Taken'].sum())
    # Safe KPI calculation
    best_avg_df = df_to_display.loc[df_to_display['Dismissals'] > 0, 'Batting_Avg']
    kpi_col3.metric("Best Batting Avg", best_avg_df.max() if not best_avg_df.empty else 0)
    best_econ_df = df_to_display.loc[df_to_display['Balls_Bowled'] > 60, 'Bowling_Economy']
    kpi_col4.metric("Best Bowling Economy", best_econ_df.min() if not best_econ_df.empty else 0)
    
    st.subheader("Detailed Charts")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Batting Performance Matrix
        plot_df_bat = df_to_display[df_to_display['Balls_Faced'] > 20] # Filter for qualified
        if not plot_df_bat.empty:
            fig_bat = px.scatter(plot_df_bat, x="Batting_Avg", y="Batting_Strike_Rate",
                                  hover_name="Player_Name", color="Role",
                                  title="Batting Performance (Avg vs. Strike Rate)",
                                  template=plotly_dark_template)
            st.plotly_chart(fig_bat, use_container_width=True)
        else:
            st.info("Not enough batting data to display this chart.")
        
        # Player Role Distribution
        if not df_to_display.empty:
            role_counts = df_to_display['Role'].value_counts().reset_index()
            fig_pie = px.pie(role_counts, values='count', names='Role', title="Player Role Distribution", template=plotly_dark_template)
            st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        # Bowling Performance Matrix
        plot_df_bowl = df_to_display[df_to_display['Balls_Bowled'] > 20] # Filter for qualified
        if not plot_df_bowl.empty:
            fig_bowl = px.scatter(plot_df_bowl, x="Bowling_Avg", y="Bowling_Economy",
                                  hover_name="Player_Name", color="Role",
                                  title="Bowling Performance (Avg vs. Economy)",
                                  template=plotly_dark_template)
            fig_bowl.update_yaxes(autorange="reversed") # Lower economy is better
            fig_bowl.update_xaxes(autorange="reversed") # Lower avg is better
            st.plotly_chart(fig_bowl, use_container_width=True)
        else:
            st.info("Not enough bowling data to display this chart.")

        # Top 10 Wicket Takers
        if not df_to_display.empty:
            fig_wickets = px.bar(df_to_display.nlargest(10, 'Wickets_Taken'), 
                                 x='Wickets_Taken', y='Player_Name', 
                                 title="Top 10 Wicket Takers", orientation='h',
                                 template=plotly_dark_template)
            st.plotly_chart(fig_wickets, use_container_width=True)
    
    # --- Display Raw Data ---
    st.subheader("Analyzed Player Data")
    
    # Define column order and image column
    column_order = [
        "Image_URL", "Player_Name", "Team", "Role", 
        "Batting_Avg", "Batting_Strike_Rate", "Runs_Scored", 
        "Bowling_Avg", "Bowling_Economy", "Wickets_Taken",
        "Matches", "Balls_Faced", "Dismissals", "Fours", "Sixes",
        "Runs_Conceded", "Balls_Bowled"
    ]
    
    # --- REMOVED "Select" COLUMN AND LOGIC ---

    st.data_editor(
        df_to_display, # Use the original dataframe
        column_config={
            # "Select" column config removed
            "Image_URL": st.column_config.ImageColumn("Image", width="small"),
            "Player_Name": "Player",
            "Batting_Avg": "Bat Avg",
            "Batting_Strike_Rate": "Bat SR",
            "Bowling_Avg": "Bowl Avg",
            "Bowling_Economy": "Bowl Econ"
        },
        column_order=[col for col in column_order if col in df_to_display.columns],
        hide_index=True,
        disabled=True, # Make all columns read-only
        key=f"editor_{tab_key}" # Unique key for the editor in this tab
    )
    
    # --- REMOVED MODAL TRIGGER ---


def display_movie_dashboard(df_to_display, tab_key):
    """
    Renders the entire Movie dashboard (KPIs, Charts, Table)
    for a given DataFrame.
    """
    # --- NEW SAFETY CHECK ---
    if df_to_display.empty:
        st.warning(f"No movies found for this filter. Please adjust your data or filters.")
        return
    # --- END SAFETY CHECK ---

    st.header("🎬 Movie Analysis")
    
    # KPIs
    st.subheader("Dataset Metrics")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Total Movies", df_to_display['Title'].count())
    kpi_col2.metric("Total Revenue", f"${df_to_display['Revenue'].sum()/1_000_000_000:.2f}B")
    kpi_col3.metric("Total Profit", f"${df_to_display['Profit'].sum()/1_000_000_000:.2f}B")
    kpi_col4.metric("Average Rating", f"{df_to_display['Rating'].mean():.1f}/10")
    
    st.subheader("Detailed Charts")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        if not df_to_display.empty:
            fig_scatter = px.scatter(df_to_display, x='Budget', y='Profit_Margin_%', 
                                     hover_name='Title', color='Genre',
                                     title="Budget vs. Profit Margin (%)",
                                     template=plotly_dark_template)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            genre_counts = df_to_display['Genre'].value_counts().reset_index()
            fig_pie = px.pie(genre_counts, values='count', names='Genre', 
                             title="Movie Genre Distribution",
                             template=plotly_dark_template)
            st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        if not df_to_display.empty:
            fig_bar = px.bar(df_to_display.nlargest(10, 'Profit'), x='Profit', y='Title', 
                             title="Top 10 Movies by Profit", orientation='h',
                             template=plotly_dark_template)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            profit_over_time = df_to_display.groupby('Release_Year')['Profit'].sum().reset_index()
            fig_line = px.line(profit_over_time, x='Release_Year', y='Profit', 
                               title="Total Profit Over Time",
                               template=plotly_dark_template)
            st.plotly_chart(fig_line, use_container_width=True)

    # --- Display Raw Data ---
    st.subheader("Analyzed Movie Data")
    column_order = [
        "Image_URL", "Title", "Genre", "Release_Year", 
        "Rating", "Budget", "Revenue", "Profit", "Profit_Margin_%"
    ]
    
    # --- REMOVED "Select" COLUMN AND LOGIC ---
    
    st.data_editor(
        df_to_display, # Use the original dataframe
        column_config={
            # "Select" column config removed
            "Image_URL": st.column_config.ImageColumn("Poster", width="small"),
            "Profit_Margin_%": st.column_config.ProgressColumn(
                "Profit Margin",
                format="%.0f%%",
                min_value=df_to_display['Profit_Margin_%'].min(),
                max_value=df_to_display['Profit_Margin_%'].max(),
            ),
            "Budget": st.column_config.NumberColumn(format="$%d"),
            "Revenue": st.column_config.NumberColumn(format="$%d"),
            "Profit": st.column_config.NumberColumn(format="$%d"),
        },
        column_order=[col for col in column_order if col in df_to_display.columns],
        hide_index=True,
        disabled=True, # Make all columns read-only
        key=f"editor_{tab_key}"
    )
    
    # --- REMOVED MODAL TRIGGER ---


def display_game_dashboard(df_to_display, tab_key):
    """
    Renders the entire Video Game dashboard (KPIs, Charts, Table)
    for a given DataFrame.
    """
    # --- NEW SAFETY CHECK ---
    if df_to_display.empty:
        st.warning(f"No games found for this filter. Please adjust your data or filters.")
        return
    # --- END SAFETY CHECK ---

    st.header("🎮 Video Game Analysis")
    
    # KPIs
    st.subheader("Dataset Metrics")
    kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
    kpi_col1.metric("Total Games", df_to_display['Title'].count())
    kpi_col2.metric("Total Global Sales", f"{df_to_display['Global_Sales_M'].sum():.0f}M")
    if not df_to_display.empty:
        # --- FIX: Changed 1j to 1 ---
        top_publisher = df_to_display.groupby('Publisher')['Global_Sales_M'].sum().nlargest(1).index[0]
        kpi_col3.metric("Top Publisher", top_publisher)
    
    st.subheader("Detailed Charts")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        if not df_to_display.empty:
            sales_by_genre = df_to_display.groupby('Genre')['Global_Sales_M'].sum().nlargest(10).reset_index()
            fig1 = px.bar(sales_by_genre, x='Global_Sales_M', y='Genre', 
                          title="Top 10 Genres by Global Sales", orientation='h',
                          template=plotly_dark_template)
            st.plotly_chart(fig1, use_container_width=True)
            
            category_counts = df_to_display['Sales_Category'].value_counts().reset_index()
            fig2 = px.pie(category_counts, values='count', names='Sales_Category', 
                          title="Game Sales Category Distribution",
                          template=plotly_dark_template)
            st.plotly_chart(fig2, use_container_width=True)

    with chart_col2:
        if not df_to_display.empty:
            sales_by_platform = df_to_display.groupby('Platform')['Global_Sales_M'].sum().nlargest(10).reset_index()
            fig3 = px.bar(sales_by_platform, x='Global_Sales_M', y='Platform', 
                          title="Top 10 Platforms by Global Sales", orientation='h',
                          template=plotly_dark_template)
            st.plotly_chart(fig3, use_container_width=True)
            
            sales_over_time = df_to_display.groupby('Release_Year')['Global_Sales_M'].sum().reset_index()
            fig4 = px.line(sales_over_time, x='Release_Year', y='Global_Sales_M', 
                           title="Total Global Sales Over Time",
                           template=plotly_dark_template)
            st.plotly_chart(fig4, use_container_width=True)

    # --- Display Raw Data ---
    st.subheader("Analyzed Game Data")
    column_order = [
        "Image_URL", "Title", "Platform", "Genre", "Publisher", 
        "Release_Year", "Global_Sales_M", "Sales_Category"
    ]
    
    # --- REMOVED "Select" COLUMN AND LOGIC ---

    st.data_editor(
        df_to_display, # Use the original dataframe
        column_config={
            # "Select" column config removed
            "Image_URL": st.column_config.ImageColumn("Cover Art", width="small"),
            "Global_Sales_M": st.column_config.NumberColumn("Global Sales (M)", format="%.1fM")
        },
        column_order=[col for col in column_order if col in df_to_display.columns],
        hide_index=True,
        disabled=True, # Make all columns read-only
        key=f"editor_{tab_key}"
    )
    
    # --- REMOVED MODAL TRIGGER ---


# -------------------------------------------------------------------
# --- MAIN LOGIC ---
# -------------------------------------------------------------------

# --- Sidebar ---
st.title("📊 Data Analysis Hub")
st.markdown("Upload your Excel file, choose an analysis type, and see the results instantly.")

st.sidebar.header("Controls")
data_type = st.sidebar.radio(
    "1. Select Data Type",
    ("Cricket", "Movies", "Video Games"),
    key='data_type'
)
uploaded_file = st.sidebar.file_uploader(
    "2. Upload your .xlsx or .csv file",
    type=["xlsx", "csv"],
    key='uploader'
)

# --- Main App Execution ---
if uploaded_file is not None:
    try:
        if st.sidebar.button("Run Analysis"):
            with st.spinner(f"Running {data_type} analysis..."):
                
                # --- NEW FILE READING LOGIC ---
                file_name = uploaded_file.name
                if file_name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif file_name.endswith('.xlsx'):
                    df = pd.read_excel(uploaded_file)
                else:
                    st.error("Invalid file type. Please upload a .csv or .xlsx file.")
                    st.stop() # Stop execution
                # --- END NEW LOGIC ---

                cleaned_df = None
                
                if data_type == "Cricket":
                    cleaned_df = analyze_cricket(df.copy())
                elif data_type == "Movies":
                    cleaned_df = analyze_movies(df.copy())
                elif data_type == "Video Games":
                    cleaned_df = analyze_video_games(df.copy())
                
                st.session_state.cleaned_df = cleaned_df
                st.session_state.analysis_run = True
                st.success("Analysis Complete!")

    except Exception as e:
        st.error(f"Error reading or analyzing file: {e}")

# --- Dashboard Display Logic ---
try:
    if 'analysis_run' in st.session_state and st.session_state.analysis_run:
        
        cleaned_df = st.session_state.cleaned_df
        data_type = st.session_state.data_type # Get data_type from state
        
        if data_type == "Cricket":
            # --- TABS for filtering ---
            tab_all, tab_hitters, tab_anchors, tab_bowlers, tab_regression = st.tabs([
                "All Players", "Power Hitters (SR > 150)", 
                "Anchors (Avg > 40)", "Power Bowlers (Wkts > 15)",
                "Regression Analysis"
            ])
            
            with tab_all:
                display_cricket_dashboard(cleaned_df, tab_key="all")
            
            with tab_hitters:
                filtered_df = cleaned_df[
                    (cleaned_df['Batting_Strike_Rate'] > 150) & 
                    (cleaned_df['Runs_Scored'] > 100)
                ]
                display_cricket_dashboard(filtered_df, tab_key="hitters")
                
            with tab_anchors:
                filtered_df = cleaned_df[
                    (cleaned_df['Batting_Avg'] > 40) & 
                    (cleaned_df['Dismissals'] > 5) # Qualified
                ]
                display_cricket_dashboard(filtered_df, tab_key="anchors")
                
            with tab_bowlers:
                filtered_df = cleaned_df[
                    (cleaned_df['Wickets_Taken'] > 15) & 
                    (cleaned_df['Balls_Bowled'] > 60)
                ]
                display_cricket_dashboard(filtered_df, tab_key="bowlers")
            
            with tab_regression:
                display_cricket_regression(cleaned_df)
        
        elif data_type == "Movies":
            # --- TABS for filtering ---
            tab_all, tab_blockbusters, tab_darlings, tab_efficient, tab_regression = st.tabs([
                "All Movies", "Blockbusters (Revenue > $500M)",
                "Critical Darlings (Rating > 8.5)", "Efficient Hits (Margin > 500%)",
                "Regression Analysis"
            ])
            
            with tab_all:
                display_movie_dashboard(cleaned_df, tab_key="all")
            
            with tab_blockbusters:
                filtered_df = cleaned_df[cleaned_df['Revenue'] > 500_000_000]
                display_movie_dashboard(filtered_df, tab_key="blockbusters")
                
            with tab_darlings:
                filtered_df = cleaned_df[cleaned_df['Rating'] > 8.5]
                display_movie_dashboard(filtered_df, tab_key="darlings")
                
            with tab_efficient:
                filtered_df = cleaned_df[cleaned_df['Profit_Margin_%'] > 500]
                display_movie_dashboard(filtered_df, tab_key="efficient")
            
            with tab_regression:
                display_movie_regression(cleaned_df)
        
        elif data_type == "Video Games":
            # --- TABS for filtering ---
            tab_all, tab_mega, tab_block, tab_hit, tab_niche, tab_regression = st.tabs([
                "All Games", "Mega-Blockbusters (>20M)", 
                "Blockbusters (10-20M)", "Hits (5-10M)", "Niche (<1M)",
                "Regression Analysis"
            ])
            
            with tab_all:
                display_game_dashboard(cleaned_df, tab_key="all")
            
            with tab_mega:
                filtered_df = cleaned_df[cleaned_df['Sales_Category'] == '>20M (Mega-Blockbuster)']
                display_game_dashboard(filtered_df, tab_key="mega")
            
            with tab_block:
                filtered_df = cleaned_df[cleaned_df['Sales_Category'] == '10-20M (Blockbuster)']
                display_game_dashboard(filtered_df, tab_key="block")
                
            with tab_hit:
                filtered_df = cleaned_df[cleaned_df['Sales_Category'] == '5-10M (Hit)']
                display_game_dashboard(filtered_df, tab_key="hit")

            with tab_niche:
                filtered_df = cleaned_df[cleaned_df['Sales_Category'] == '<1M (Niche)']
                display_game_dashboard(filtered_df, tab_key="niche")

            with tab_regression:
                display_game_regression(cleaned_df)

except Exception as e:
    # Catch any rendering errors
    st.error(f"An error occurred while trying to display the dashboard: {e}")

if uploaded_file is None:
    # This is the "home page" before anything is uploaded
    st.info("Please upload an .xlsx file using the sidebar to begin.")