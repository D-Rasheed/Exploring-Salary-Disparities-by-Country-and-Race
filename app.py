import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
# Importing data
df = pd.read_csv('D:/Prog_project_summer/assets/Salary_Data_Based_country_and_race.csv')
eda = pd.read_csv('D:/Prog_project_summer/clean.csv')
	
st.sidebar.header('SALARY DATASET BASED ON COUNTRY AND RACE')
menu = st.sidebar.radio(
    "Menu:",
    ("Intro", "Data", "Analysis", "Models"),
)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.sidebar.markdown('---')
st.sidebar.write('Project Submitted By: Danish Rasheed')
st.sidebar.write('Matricola No.: VR497604')
st.sidebar.write('Github Repositories:')
st.sidebar.write('https://github.com/D-Rasheed/Exploring-Salary-Disparities-by-Country-and-Race')
if menu == 'Intro':
   st.write('Dataset Description The dataset comprises the following attributes:')
   st.write('Age: The age of the individual.')
   st.write('Gender: The gender of the individual.')     
   st.write('Education: The highest level of education attained by the individual.')
   st.write('Country: The country of residence of the individual.')
   st.write('Race: The racial background of the individual.')
   st.write('Years of Experience: The number of years of professional experience.')
   st.write('Salary: The income of the individual.')
   st.write('The dataset comprehensive nature enables researchers to explore patterns and trends in income distribution across different demographic categories. By conducting thorough exploratory data analysis (EDA), we can identify potential disparities or variations in earning potential. Moreover, the integration of years of experience as a variable allows for the investigation of how income varies based on both demographic characteristics and accumulated work experience.')
elif menu == 'Data':

   st.title("DataFrame:")
   st.write(">***6704 entries | 09 columns***")
   st.dataframe(df)
elif menu == 'Analysis':
# Filter out the 'Other' gender
   eda_without_other_gender = eda[eda['Gender'] != 'Other']

# Set the title of your Streamlit app
   st.title("EDA Visualizations")

# Boxplot by gender and salary
   st.subheader("Boxplot by Gender and Salary")
   fig1, ax1 = plt.subplots(figsize=(8, 5))
   sns.boxplot(data=eda_without_other_gender, x='Gender', y='Salary', ax=ax1)
   plt.title('Stats by Gender')
   st.pyplot(fig1)

# Barplot of salary by gender
   st.subheader("Barplot of Salary by Gender")
   fig2, ax2 = plt.subplots(figsize=(8, 5))
   sns.barplot(data=eda_without_other_gender, x="Gender", y="Salary", ax=ax2)
   ax2.set_title("Salary by Gender")
   st.pyplot(fig2)

# Barplot of salary by education level
   st.subheader("Barplot of Salary by Education Level")
   fig3, ax3 = plt.subplots(figsize=(8, 5))
   sns.barplot(data=eda, x="Education Level", y="Salary", ax=ax3)
   ax3.set_title("Salary and Education Level")
   plt.xticks(rotation=0)
   st.pyplot(fig3)

# Barplot of age by education level
   st.subheader("Barplot of Age by Education Level")
   fig4, ax4 = plt.subplots(figsize=(8, 5))
   sns.barplot(data=eda, x="Education Level", y="Age", ax=ax4)
   ax4.set_title("Age by Education Level")
   plt.xticks(rotation=90)
   st.pyplot(fig4)

# Barplots for race and country
   st.subheader("Barplots for Race and Country")
   fig5, (ax5, ax6) = plt.subplots(1, 2, figsize=(15, 5))
   sns.barplot(data=eda, x="Race", y="Salary", ax=ax5)
   ax5.set_title("Salary by Race")
   sns.barplot(data=eda, x="Country", y="Salary", ax=ax6)
   ax6.set_title("Salary by Country")
   plt.xticks(rotation=45, ha='right')
   plt.tight_layout()
   st.pyplot(fig5)

# Barplot of top 20 jobs
   st.subheader("Top 20 Jobs")
   top_jobs = eda['Job Title'].value_counts().head(20)
   fig7, ax7 = plt.subplots(figsize=(10, 5))
   top_jobs.plot(kind='bar', ax=ax7)
   ax7.set_title("Top 20 Jobs")
   ax7.set_xlabel("Job Title")
   ax7.set_ylabel("Count")
   plt.xticks(rotation=45, ha='right')
   plt.tight_layout()
   st.pyplot(fig7)

# Scatterplot of salary by years of experience
   st.subheader("Scatterplot of Salary by Years of Experience")
   fig8, ax8 = plt.subplots(figsize=(8, 5))
   sns.scatterplot(data=eda_without_other_gender, x='Years of Experience', y='Salary', hue='Gender', ax=ax8)
   ax8.set_title("Correlation between Salary and Experience")
   st.pyplot(fig8)


elif menu == 'Models':
   # Data preprocessing
   label_encoder = preprocessing.LabelEncoder()
   categorical_columns = ['Gender', 'Country', 'Education Level', 'Job Title', 'Race']
   le = LabelEncoder()
   
   for col in categorical_columns:
       eda[col] = le.fit_transform(eda[col])
   unique_values = eda[col].unique()
    

   x = eda.drop(['Salary'], axis=1)
   y = eda['Salary']
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=2)

   # Model training and evaluation
   #LinearRegression
   lr_model = LinearRegression()
   lr_model.fit(x_train, y_train)
   lr_predictions = lr_model.predict(x_test)
   lr_r2 = r2_score(y_test, lr_predictions)
   st.write("Linear Regression R-squared:", lr_r2)
   #RandomForestRegressor
   rf_model = RandomForestRegressor(random_state=42)
   rf_model.fit(x_train, y_train)
   rf_predictions = rf_model.predict(x_test)
   rf_r2 = r2_score(y_test, rf_predictions)
   st.write("Random Forest R-squared:", rf_r2)
