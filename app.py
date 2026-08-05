import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("Employee Attrition Prediction")
st.write("Fill employee details and click Predict.")

c1,c2,c3 = st.columns(3)

with c1:
    age=st.number_input("Age",18,60,30)
    business=st.selectbox("Business Travel",["Travel_Rarely","Travel_Frequently","Non-Travel"])
    daily=st.number_input("Daily Rate",100,1500,800)
    dept=st.selectbox("Department",["Sales","Research & Development","Human Resources"])
    dist=st.number_input("Distance From Home",1,29,5)
    edu=st.selectbox("Education",[1,2,3,4,5])
    edufield=st.selectbox("Education Field",["Life Sciences","Medical","Marketing","Technical Degree","Other","Human Resources"])
    env=st.selectbox("Environment Satisfaction",[1,2,3,4])
    gender=st.selectbox("Gender",["Male","Female"])
    hourly=st.number_input("Hourly Rate",30,100,60)

with c2:
    involvement=st.selectbox("Job Involvement",[1,2,3,4])
    level=st.selectbox("Job Level",[1,2,3,4,5])
    role=st.selectbox("Job Role",[
        "Sales Executive","Research Scientist","Laboratory Technician",
        "Manufacturing Director","Healthcare Representative","Manager",
        "Sales Representative","Research Director","Human Resources"
    ])
    satisfaction=st.selectbox("Job Satisfaction",[1,2,3,4])
    marital=st.selectbox("Marital Status",["Single","Married","Divorced"])
    income=st.number_input("Monthly Income",1000,30000,5000)
    monthly_rate=st.number_input("Monthly Rate",1000,30000,10000)
    companies=st.number_input("Companies Worked",0,10,1)
    overtime=st.selectbox("OverTime",["Yes","No"])
    hike=st.number_input("Percent Salary Hike",11,25,15)

with c3:
    performance=st.selectbox("Performance Rating",[3,4])
    relation=st.selectbox("Relationship Satisfaction",[1,2,3,4])
    stock=st.selectbox("Stock Option Level",[0,1,2,3])
    total=st.number_input("Total Working Years",0,40,5)
    training=st.number_input("Training Times Last Year",0,6,2)
    balance=st.selectbox("Work Life Balance",[1,2,3,4])
    years_company=st.number_input("Years At Company",0,40,5)
    years_role=st.number_input("Years In Current Role",0,18,3)
    promotion=st.number_input("Years Since Last Promotion",0,15,1)
    manager=st.number_input("Years With Current Manager",0,17,3)

if st.button("Predict Attrition", type="primary"):
    df = pd.DataFrame([{
        "Age":age,
        "BusinessTravel":business,
        "DailyRate":daily,
        "Department":dept,
        "DistanceFromHome":dist,
        "Education":edu,
        "EducationField":edufield,
        "EnvironmentSatisfaction":env,
        "Gender":gender,
        "HourlyRate":hourly,
        "JobInvolvement":involvement,
        "JobLevel":level,
        "JobRole":role,
        "JobSatisfaction":satisfaction,
        "MaritalStatus":marital,
        "MonthlyIncome":income,
        "MonthlyRate":monthly_rate,
        "NumCompaniesWorked":companies,
        "OverTime":overtime,
        "PercentSalaryHike":hike,
        "PerformanceRating":performance,
        "RelationshipSatisfaction":relation,
        "StockOptionLevel":stock,
        "TotalWorkingYears":total,
        "TrainingTimesLastYear":training,
        "WorkLifeBalance":balance,
        "YearsAtCompany":years_company,
        "YearsInCurrentRole":years_role,
        "YearsSinceLastPromotion":promotion,
        "YearsWithCurrManager":manager
    }])
    pred=model.predict(df)[0]
    prob=model.predict_proba(df)[0][1]
    if pred==1:
        st.error("Prediction: Employee is likely to leave.")
    else:
        st.success("Prediction: Employee is likely to stay.")
    st.metric("Attrition Probability", f"{prob:.2%}")
