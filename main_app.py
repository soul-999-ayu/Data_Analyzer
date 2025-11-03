import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io # Used to handle the file buffer

# --- Page Setup ---
st.set_page_config(
    page_title="Data Analysis Hub",
    page_icon="📊",
    layout="wide"
)

# --- Analysis Logic Functions ---
# (These are identical to your original functions)

def analyze_cricket(df):
    """
    Performs analysis on a DataFrame with expected cricket columns.
    Calculates Batting Strike Rate and Bowling Economy.
    """
    try:
        # 1. Calculate Batting Strike Rate
        df['Batting_Strike_Rate'] = 0.0
        batting_mask = df['Balls_Faced'] > 0
        df.loc[batting_mask, 'Batting_Strike_Rate'] = \
            (df.loc[batting_mask, 'Runs_Scored'] / df.loc[batting_mask, 'Balls_Faced']) * 100

        # 2. Calculate Bowling Economy
        df['Bowling_Economy'] = 0.0
        bowling_mask = df['Balls_Bowled'] > 0
        df.loc[bowling_mask, 'Bowling_Economy'] = \
            df.loc[bowling_mask, 'Runs_Conceded'] / (df.loc[bowling_mask, 'Balls_Bowled'] / 6)

        # 3. Final Cleaning
        df['Batting_Strike_Rate'] = df['Batting_Strike_Rate'].round(2)
        df['Bowling_Economy'] = df['Bowling_Economy'].round(2)
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        return df
    except KeyError as e:
        raise ValueError(f"Missing expected cricket column: {e}. Please check your file.")
    except Exception as e:
        raise Exception(f"An error occurred during cricket analysis: {e}")

def analyze_movies(df):
    """
    Performs analysis on a DataFrame with expected movie columns.
    Calculates Profit and Profit Margin.
    """
    try:
        # 1. Calculate Profit
        df['Profit'] = df['Revenue'] - df['Budget']
        # 2. Calculate Profit Margin
        df['Profit_Margin_%'] = 0.0
        budget_mask = df['Budget'] > 0
        df.loc[budget_mask, 'Profit_Margin_%'] = (df.loc[budget_mask, 'Profit'] / df.loc[budget_mask, 'Budget']) * 100
        # 3. Final Cleaning
        df['Profit_Margin_%'] = df['Profit_Margin_%'].round(2)
        df.fillna(0, inplace=True)
        return df
    except KeyError as e:
        raise ValueError(f"Missing expected movie column: {e}. Please check your file.")
    except Exception as e:
        raise Exception(f"An error occurred during movie analysis: {e}")

def analyze_video_games(df):
    """
    Performs analysis on a DataFrame with expected video game columns.
    Categorizes games by sales figures.
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

# --- Dashboard UI ---
st.title("📊 Data Analysis Hub")
st.markdown("Upload your Excel file, choose an analysis type, and see the results instantly.")

# --- Sidebar for Controls ---
st.sidebar.header("Controls")

# 1. Data Type Selection
data_type = st.sidebar.radio(
    "1. Select Data Type",
    ("Cricket", "Movies", "Video Games")
)

# 2. File Uploader
uploaded_file = st.sidebar.file_uploader(
    "2. Upload your .xlsx file",
    type="xlsx"
)

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_excel(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
        
        # 3. Run Analysis Button
        if st.sidebar.button("Run Analysis"):
            cleaned_df = None
            
            with st.spinner(f"Running {data_type} analysis..."):
                try:
                    # 4. Route to the correct analysis function
                    if data_type == "Cricket":
                        cleaned_df = analyze_cricket(df)
                    elif data_type == "Movies":
                        cleaned_df = analyze_movies(df)
                    elif data_type == "Video Games":
                        cleaned_df = analyze_video_games(df)
                    
                    st.success("Analysis Complete!")
                    
                    # 5. Display the results!
                    
                    # --- CRICKET DASHBOARD ---
                    if data_type == "Cricket":
                        st.header("🏆 Cricket Analysis")
                        
                        # KPIs
                        st.subheader("Key Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Runs Scored", cleaned_df['Runs_Scored'].sum())
                        col2.metric("Total Wickets Taken", cleaned_df['Wickets_Taken'].sum())
                        col3.metric("Best Strike Rate", cleaned_df.loc[cleaned_df['Balls_Faced'] > 20, 'Batting_Strike_Rate'].max())
                        col4.metric("Best Economy", cleaned_df.loc[cleaned_df['Balls_Bowled'] > 20, 'Bowling_Economy'].min())
                        
                        st.subheader("Detailed Charts")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Top 10 Run Scorers
                            if 'Runs_Scored' in cleaned_df.columns and 'Player_Name' in cleaned_df.columns:
                                fig1 = px.bar(cleaned_df.nlargest(10, 'Runs_Scored'), x='Runs_Scored', y='Player_Name', title="Top 10 Run Scorers", orientation='h')
                                st.plotly_chart(fig1, use_container_width=True)
                            
                            # Player Role Distribution
                            if 'Role' in cleaned_df.columns:
                                fig2 = px.pie(cleaned_df['Role'].value_counts().reset_index(), values='count', names='Role', title="Player Role Distribution")
                                st.plotly_chart(fig2, use_container_width=True)

                        with col2:
                            # Top 10 Wicket Takers
                            if 'Wickets_Taken' in cleaned_df.columns and 'Player_Name' in cleaned_df.columns:
                                fig3 = px.bar(cleaned_df.nlargest(10, 'Wickets_Taken'), x='Wickets_Taken', y='Player_Name', title="Top 10 Wicket Takers", orientation='h')
                                st.plotly_chart(fig3, use_container_width=True)
                            
                            # Performance Matrix (All-Rounders)
                            if 'Batting_Strike_Rate' in cleaned_df.columns and 'Bowling_Economy' in cleaned_df.columns:
                                plot_df = cleaned_df[ (cleaned_df['Balls_Faced'] > 0) & (cleaned_df['Balls_Bowled'] > 0) ]
                                fig4 = px.scatter(plot_df, x="Batting_Strike_Rate", y="Bowling_Economy",
                                                  hover_name="Player_Name", color="Role",
                                                  title="Performance Matrix: Strike Rate vs. Economy (Min. 1 ball faced & bowled)")
                                fig4.update_yaxes(autorange="reversed") # Lower economy is better
                                st.plotly_chart(fig4, use_container_width=True)

                    # --- MOVIES DASHBOARD ---
                    elif data_type == "Movies":
                        st.header("🎬 Movie Analysis")
                        
                        # KPIs
                        st.subheader("Key Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Movies", cleaned_df['Title'].count())
                        col2.metric("Total Revenue", f"${cleaned_df['Revenue'].sum():,.0f}")
                        col3.metric("Total Profit", f"${cleaned_df['Profit'].sum():,.0f}")
                        if 'Rating' in cleaned_df.columns:
                            col4.metric("Average Rating", f"{cleaned_df['Rating'].mean():.1f}/10")
                        
                        st.subheader("Detailed Charts")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Budget vs. Profit
                            if 'Budget' in cleaned_df.columns and 'Profit' in cleaned_df.columns:
                                fig1 = px.scatter(cleaned_df, x='Budget', y='Profit', hover_name='Title', 
                                                  color='Profit', color_continuous_scale='RdYlGn',
                                                  title="Budget vs. Profit")
                                st.plotly_chart(fig1, use_container_width=True)
                            
                            # Genre Distribution
                            if 'Genre' in cleaned_df.columns:
                                fig2 = px.pie(cleaned_df['Genre'].value_counts().reset_index(), values='count', names='Genre', title="Movie Genre Distribution")
                                st.plotly_chart(fig2, use_container_width=True)

                        with col2:
                            # Top 10 by Profit Margin
                            if 'Profit_Margin_%' in cleaned_df.columns and 'Title' in cleaned_df.columns:
                                fig3 = px.bar(cleaned_df.nlargest(10, 'Profit_Margin_%'), x='Profit_Margin_%', y='Title', title="Top 10 Movies by Profit Margin (%)", orientation='h')
                                st.plotly_chart(fig3, use_container_width=True)
                            
                            # Profit Over Time
                            if 'Release_Year' in cleaned_df.columns and 'Profit' in cleaned_df.columns:
                                profit_over_time = cleaned_df.groupby('Release_Year')['Profit'].sum().reset_index()
                                fig4 = px.line(profit_over_time, x='Release_Year', y='Profit', title="Total Profit Over Time")
                                st.plotly_chart(fig4, use_container_width=True)

                    # --- VIDEO GAMES DASHBOARD ---
                    elif data_type == "Video Games":
                        st.header("🎮 Video Game Analysis")
                        
                        # KPIs
                        st.subheader("Key Metrics")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Games", cleaned_df['Title'].count())
                        col2.metric("Total Global Sales", f"{cleaned_df['Global_Sales_M'].sum():.0f}M")
                        if 'Publisher' in cleaned_df.columns:
                            top_publisher = cleaned_df['Publisher'].value_counts().idxmax()
                            col3.metric("Top Publisher", top_publisher)
                        
                        st.subheader("Detailed Charts")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Sales by Genre
                            if 'Genre' in cleaned_df.columns and 'Global_Sales_M' in cleaned_df.columns:
                                sales_by_genre = cleaned_df.groupby('Genre')['Global_Sales_M'].sum().nlargest(10).reset_index()
                                fig1 = px.bar(sales_by_genre, x='Global_Sales_M', y='Genre', title="Top 10 Genres by Global Sales", orientation='h')
                                st.plotly_chart(fig1, use_container_width=True)
                            
                            # Sales Category Distribution
                            if 'Sales_Category' in cleaned_df.columns:
                                category_counts = cleaned_df['Sales_Category'].value_counts().reset_index()
                                fig2 = px.pie(category_counts, values='count', names='Sales_Category', title="Game Sales Category Distribution")
                                st.plotly_chart(fig2, use_container_width=True)

                        with col2:
                            # Sales by Platform
                            if 'Platform' in cleaned_df.columns and 'Global_Sales_M' in cleaned_df.columns:
                                sales_by_platform = cleaned_df.groupby('Platform')['Global_Sales_M'].sum().nlargest(10).reset_index()
                                fig3 = px.bar(sales_by_platform, x='Global_Sales_M', y='Platform', title="Top 10 Platforms by Global Sales", orientation='h')
                                st.plotly_chart(fig3, use_container_width=True)
                            
                            # Sales Over Time
                            if 'Release_Year' in cleaned_df.columns and 'Global_Sales_M' in cleaned_df.columns:
                                sales_over_time = cleaned_df.groupby('Release_Year')['Global_Sales_M'].sum().reset_index()
                                fig4 = px.line(sales_over_time, x='Release_Year', y='Global_Sales_M', title="Total Global Sales Over Time")
                                st.plotly_chart(fig4, use_container_width=True)

                    # --- Display Raw Data ---
                    st.subheader("Analyzed Data Table")
                    st.dataframe(cleaned_df)

                except Exception as e:
                    st.error(f"AN ERROR OCCURRED: {e}")

    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
else:
    st.info("Please upload an .xlsx file using the sidebar to begin.")

