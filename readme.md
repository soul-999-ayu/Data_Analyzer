# **📊 Data Analysis Hub**

This project is a web-based application built with Streamlit that allows you to upload Excel data for Cricket, Movies, or Video Games, and instantly performs an analysis and displays the results with interactive graphs.

# **Guide 1: How to Set Up and Run the App**

This guide will walk you through setting up the necessary Python environment, installing all the required packages, and running your new Streamlit dashboard.

### **Step 1: Create a Python Virtual Environment (venv)**

This is a best practice that keeps your project's packages separate from your system's.

1. Open your terminal in your project folder (e.g., ```.../Projects/College/Mohit```).  
2. Run the following command to create a virtual environment folder named venv:  
   ```python3 -m venv venv```

   *(Note: If ```python3``` doesn't work, you can try just ```python```)*  
3. Activate the environment. This changes your terminal to "use" this new empty environment:  
   ```source venv/bin/activate```

   Your terminal prompt should now look something like (venv)   ...

### **Step 2: Install All Required Modules (Packages)**

Now that your environment is active, you can safely install the packages.

Run this single command to install streamlit (the GUI), pandas (data logic), numpy (math for pandas), openpyxl (to read Excel files), and plotly (for graphs):

```pip install streamlit pandas numpy openpyxl plotly```

### **Step 3: Save Your Code**

Save the Python code from our last conversation into a file named ```analyzer_dashboard.py``` in your project folder.

### **Step 4: Run the Analyzer App**

This is the most important part. You **do not** run this script with python .... You must use the streamlit command:

1. Make sure your terminal is in the correct folder and your (venv) is active.  
2. Run this command:  
   ```streamlit run analyzer_dashboard.py```

### **What Happens Next**

Your terminal will show you a message like this:

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501  
  Network URL: \[http://192.168.1.10:8501\](http://192.168.1.10:8501)

Streamlit will automatically open your default web browser to the "Local URL". You will see your app live, ready to use\!

To stop the app, go back to your terminal and press Ctrl+C.

# **Guide 2: How the Code Actually Works**

This guide breaks down the streamlit\_dashboard.py script to explain what each part does.

The app's magic comes from **Streamlit**. Streamlit's core idea is simple: **the script re-runs from top to bottom every time you interact with a widget (like a button or radio button).**

### **1\. Imports (The "Cast of Characters")**

* import streamlit as st: This is the main library for creating the web app. We use st as a nickname.  
* import pandas as pd: This is for loading and manipulating the Excel data.  
* import numpy as np: Used by Pandas for numerical operations.  
* import plotly.express as px: This is a powerful library for making interactive charts (bar, scatter, pie, etc.).

### **2\. Analysis Logic Functions**

These are your analyze\_cricket(), analyze\_movies(), and analyze\_video\_games() functions.

* **Purpose:** Their job is to be the "data engine." They are pure Python and Pandas.  
* **Input:** They each take one argument: a raw Pandas DataFrame (df) loaded from your Excel file.  
* **Process:** They use Pandas to create new columns (like Batting\_Strike\_Rate or Profit) and clean up the data.  
* **Output:** They each return a new, *cleaned* DataFrame with the new columns added.

### **3\. The Streamlit UI (The "Main" App)**

This is everything after the analysis functions.

* st.set\_page\_config(...): This is just setup. It sets the browser tab's title ("Data Analysis Hub"), icon (📊), and makes the layout use the full screen width.  
* st.title(...) / st.markdown(...): These are the simplest commands. They just write text onto the web page.  
* **st.sidebar**: Any command starting with st.sidebar. (like st.sidebar.radio) places that widget in the left-hand sidebar.  
* data\_type \= st.sidebar.radio(...): This command does two things:  
  1. It draws the radio button selector in the sidebar.  
  2. It takes the user's *choice* (e.g., "Cricket") and stores it in the data\_type variable.  
* uploaded\_file \= st.sidebar.file\_uploader(...): This draws the file upload box.  
  * If no file is uploaded, uploaded\_file is None.  
  * When you upload a file, the script re-runs, and uploaded\_file now contains the file's data.

### **4\. The Main Logic Flow (Putting It All Together)**

This is the core of the app, which starts with if uploaded\_file is not None:.

1. **The "Gatekeeper":** The code first checks if a file has been uploaded. If uploaded\_file is None, it skips everything and just runs st.info("Please upload an .xlsx file...") at the very end.  
2. **The "Trigger":** Inside the if block, if st.sidebar.button("Run Analysis"): is the main trigger. The code inside *this* block only runs when the "Run Analysis" button is clicked.  
3. **The "Router":** Inside the button's code block, the app looks at the data\_type variable (which we got from the radio button) and decides which analysis function to run:  
   if data\_type \== "Cricket":  
       cleaned\_df \= analyze\_cricket(df)  
   elif data\_type \== "Movies":  
       cleaned\_df \= analyze\_movies(df)  
   \# ...and so on

4. **Displaying Results:** This is the most "Streamlit-y" part.  
   * st.success(...): Shows the green "Analysis Complete\!" box.  
   * st.header(...) / st.subheader(...): Writes the section titles.  
   * st.metric(...): This draws one of the "Key Metric" boxes with a title and a large number.  
   * col1, col2 \= st.columns(2): This creates a 2-column layout. Any code inside with col1: goes in the left column, and code inside with col2: goes in the right.  
   * fig \= px.bar(...): This line **does not** draw a chart. It uses Plotly Express (px) to *create* a chart object (a "figure") and store it in a variable (e.g., fig1).  
   * st.plotly\_chart(fig): This is the command that tells Streamlit: "Take this chart object you have in memory and draw it on the web page."  
   * st.dataframe(cleaned\_df): This is a powerful command that takes your final, cleaned Pandas DataFrame and draws it as an interactive table that you can sort and scroll.