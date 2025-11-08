import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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

# --- Dashboard UI ---
st.title("📊 Data Analysis Hub")
st.markdown("Upload your Excel file, choose an analysis type, and see the results instantly.")

# --- Sidebar for Controls ---
st.sidebar.header("Controls")

# 1. Data Type Selection
data_type = st.sidebar.radio(
    "1. Select Data Type",
    ("Cricket", "Movies", "Video Games"),
    key='data_type'
)

# 2. File Uploader
uploaded_file = st.sidebar.file_uploader(
    "2. Upload your .xlsx file",
    type="xlsx",
    key='uploader'
)

# --- Main Logic ---

# --- Main Logic ---

if uploaded_file is not None:
    try:
        # 3. Run Analysis Button
        if st.sidebar.button("Run Analysis"):
            with st.spinner(f"Running {data_type} analysis..."):
                df = pd.read_excel(uploaded_file)
                cleaned_df = None
                
                # The 'data_type' variable here is already correct
                # because the radio widget set it in session_state
                if data_type == "Cricket":
                    cleaned_df = analyze_cricket(df.copy())
                elif data_type == "Movies":
                    cleaned_df = analyze_movies(df.copy())
                elif data_type == "Video Games":
                    cleaned_df = analyze_video_games(df.copy())
                
                # Save the results to session state
                st.session_state.cleaned_df = cleaned_df
                st.session_state.analysis_run = True
                
                # DELETE THIS LINE:
                # st.session_state.data_type = data_type 
                
                st.success("Analysis Complete!")

    except Exception as e:
        st.error(f"Error reading or analyzing file: {e}")

# Check session state to re-display dashboard on re-runs
try:
    if 'analysis_run' in st.session_state and st.session_state.analysis_run:
        
        # Retrieve the saved data
        cleaned_df = st.session_state.cleaned_df
        data_type = st.session_state.data_type
        
        # --- CRICKET DASHBOARD ---
        if data_type == "Cricket":
            st.header("🏏 Cricket Player Analysis")
            
            # --- KPIs ---
            st.subheader("Tournament Metrics")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            kpi_col1.metric("Total Runs Scored", cleaned_df['Runs_Scored'].sum())
            kpi_col2.metric("Total Wickets Taken", cleaned_df['Wickets_Taken'].sum())
            kpi_col3.metric("Best Batting Avg", cleaned_df.loc[cleaned_df['Dismissals'] > 0, 'Batting_Avg'].max())
            kpi_col4.metric("Best Bowling Economy", cleaned_df.loc[cleaned_df['Balls_Bowled'] > 60, 'Bowling_Economy'].min())
            
            st.subheader("Detailed Charts")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Batting Performance Matrix
                plot_df_bat = cleaned_df[cleaned_df['Balls_Faced'] > 20] # Filter for qualified
                fig_bat = px.scatter(plot_df_bat, x="Batting_Avg", y="Batting_Strike_Rate",
                                      hover_name="Player_Name", color="Role",
                                      title="Batting Performance (Avg vs. Strike Rate)",
                                      template=plotly_dark_template)
                st.plotly_chart(fig_bat, use_container_width=True)
                
                # Player Role Distribution
                role_counts = cleaned_df['Role'].value_counts().reset_index()
                fig_pie = px.pie(role_counts, values='count', names='Role', title="Player Role Distribution", template=plotly_dark_template)
                st.plotly_chart(fig_pie, use_container_width=True)

            with chart_col2:
                # Bowling Performance Matrix
                plot_df_bowl = cleaned_df[cleaned_df['Balls_Bowled'] > 20] # Filter for qualified
                fig_bowl = px.scatter(plot_df_bowl, x="Bowling_Avg", y="Bowling_Economy",
                                      hover_name="Player_Name", color="Role",
                                      title="Bowling Performance (Avg vs. Economy)",
                                      template=plotly_dark_template)
                fig_bowl.update_yaxes(autorange="reversed") # Lower economy is better
                fig_bowl.update_xaxes(autorange="reversed") # Lower avg is better
                st.plotly_chart(fig_bowl, use_container_width=True)

                # Top 10 Wicket Takers
                fig_wickets = px.bar(cleaned_df.nlargest(10, 'Wickets_Taken'), 
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
            
            # Create columns for the static list and the main table
            list_col, table_col = st.columns([1, 4]) # 1 part for list, 4 for table

            with list_col:
                # Title for the list
                st.markdown("##### All Players") 
                # Create a simple DataFrame with just the player names
                player_list_df = cleaned_df[['Player_Name']]
                # Display this as a read-only, scrollable list
                st.data_editor(
                    player_list_df,
                    hide_index=True,
                    disabled=True,
                    use_container_width=True,
                    key="player_list_display" # Added a unique key
                )

            with table_col:
                # Display the FULL, UNFILTERED DataFrame
                st.data_editor(
                    cleaned_df,
                    column_config={
                        "Image_URL": st.column_config.ImageColumn("Image", width="small"),
                        "Player_Name": "Player",
                        "Batting_Avg": "Bat Avg",
                        "Batting_Strike_Rate": "Bat SR",
                        "Bowling_Avg": "Bowl Avg",
                        "Bowling_Economy": "Bowl Econ"
                    },
                    column_order=column_order,
                    hide_index=True,
                    disabled=True # Make read-only
                )


        # --- MOVIES DASHBOARD ---
        elif data_type == "Movies":
            st.header("🎬 Movie Analysis")
            
            # KPIs
            st.subheader("Dataset Metrics")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            kpi_col1.metric("Total Movies", cleaned_df['Title'].count())
            kpi_col2.metric("Total Revenue", f"${cleaned_df['Revenue'].sum()/1_000_000_000:.2f}B")
            kpi_col3.metric("Total Profit", f"${cleaned_df['Profit'].sum()/1_000_000_000:.2f}B")
            kpi_col4.metric("Average Rating", f"{cleaned_df['Rating'].mean():.1f}/10")
            
            st.subheader("Detailed Charts")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Budget vs. Profit Margin
                fig_scatter = px.scatter(cleaned_df, x='Budget', y='Profit_Margin_%', 
                                         hover_name='Title', color='Genre',
                                         title="Budget vs. Profit Margin (%)",
                                         template=plotly_dark_template)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Genre Distribution
                genre_counts = cleaned_df['Genre'].value_counts().reset_index()
                fig_pie = px.pie(genre_counts, values='count', names='Genre', 
                                 title="Movie Genre Distribution",
                                 template=plotly_dark_template)
                st.plotly_chart(fig_pie, use_container_width=True)

            with chart_col2:
                # Top 10 by Profit
                fig_bar = px.bar(cleaned_df.nlargest(10, 'Profit'), x='Profit', y='Title', 
                                 title="Top 10 Movies by Profit", orientation='h',
                                 template=plotly_dark_template)
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Profit Over Time
                profit_over_time = cleaned_df.groupby('Release_Year')['Profit'].sum().reset_index()
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
            st.data_editor(
                cleaned_df,
                column_config={
                    "Image_URL": st.column_config.ImageColumn("Poster", width="small"),
                    "Profit_Margin_%": st.column_config.ProgressColumn(
                        "Profit Margin",
                        format="%.0f%%",
                        # --- FIX IS HERE ---
                        min_value=cleaned_df['Profit_Margin_%'].min(),
                        max_value=cleaned_df['Profit_Margin_%'].max(),
                    ),
                    "Budget": st.column_config.NumberColumn(format="$%d"),
                    "Revenue": st.column_config.NumberColumn(format="$%d"),
                    "Profit": st.column_config.NumberColumn(format="$%d"),
                },
                column_order=column_order,
                hide_index=True,
                disabled=True
            )


        # --- VIDEO GAMES DASHBOARD ---
        elif data_type == "Video Games":
            st.header("🎮 Video Game Analysis")
            
            # KPIs
            st.subheader("Dataset Metrics")
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            kpi_col1.metric("Total Games", cleaned_df['Title'].count())
            kpi_col2.metric("Total Global Sales", f"{cleaned_df['Global_Sales_M'].sum():.0f}M")
            top_publisher = cleaned_df.groupby('Publisher')['Global_Sales_M'].sum().nlargest(1).index[0]
            kpi_col3.metric("Top Publisher", top_publisher)
            
            st.subheader("Detailed Charts")
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Sales by Genre
                sales_by_genre = cleaned_df.groupby('Genre')['Global_Sales_M'].sum().nlargest(10).reset_index()
                fig1 = px.bar(sales_by_genre, x='Global_Sales_M', y='Genre', 
                              title="Top 10 Genres by Global Sales", orientation='h',
                              template=plotly_dark_template)
                st.plotly_chart(fig1, use_container_width=True)
                
                # Sales Category Distribution
                category_counts = cleaned_df['Sales_Category'].value_counts().reset_index()
                fig2 = px.pie(category_counts, values='count', names='Sales_Category', 
                              title="Game Sales Category Distribution",
                              template=plotly_dark_template)
                st.plotly_chart(fig2, use_container_width=True)

            with chart_col2:
                # Sales by Platform
                sales_by_platform = cleaned_df.groupby('Platform')['Global_Sales_M'].sum().nlargest(10).reset_index()
                fig3 = px.bar(sales_by_platform, x='Global_Sales_M', y='Platform', 
                              title="Top 10 Platforms by Global Sales", orientation='h',
                              template=plotly_dark_template)
                st.plotly_chart(fig3, use_container_width=True)
                
                # Sales Over Time
                sales_over_time = cleaned_df.groupby('Release_Year')['Global_Sales_M'].sum().reset_index()
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
            st.data_editor(
                cleaned_df,
                column_config={
                    "Image_URL": st.column_config.ImageColumn("Cover Art", width="small"),
                    "Global_Sales_M": st.column_config.NumberColumn("Global Sales (M)", format="%.1fM")
                },
                column_order=column_order,
                hide_index=True,
                disabled=True
            )

except Exception as e:
    # Catch any rendering errors
    st.error(f"An error occurred while trying to display the dashboard: {e}")

# THIS IS THE FIX: Changed 'elif' to 'if'
if uploaded_file is None:
    st.info("Please upload an .xlsx file using the sidebar to begin.")