
from flask import Flask, request, render_template
import pickle
import pandas as pd

ld=pd.read_csv('Loan_Defaulters - Copy.csv')

app = Flask(__name__)

with open('model1.pkl', 'rb') as file:
    model1 = pickle.load(file)

@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    loan_amount=request.form['Loan Amount']

    funded_amount=request.form['Funded Amount']

    funded_amount_investor=request.form['Funded Amount Investor']

    term=request.form['Term']

    interest_rate =request.form['Interest Rate']

    Grade =request.form['Grade']

    employment_duration=request.form['Employment Duration']

    home_ownership=request.form['Home Ownership']

    verification_status=request.form['Verification Status']

    loan_title=request.form['Loan Title']

    debit_to_income=request.form['Debit to Income']

    delinquency_two_years=request.form['Delinquency - two years']

    revolving_balance=request.form['Revolving Balance']

    revolving_utilities=request.form['Revolving Utilities']

    total_accounts=request.form['Total Accounts']

    initial_list_status=request.form['Initial List Status']

    total_received_interest=request.form['Total Received Interest']

    total_received_late_fee=request.form['Total Received Late Fee']

    recoveries=request.form['Recoveries']

    collection_recovery_fee=request.form['Collection Recovery Fee']

    collection_12months_medical=request.form['Collection 12 months Medical']

    application_type=request.form['Application Type']

    last_week_pay=request.form['Last week Pay']

    total_collection_amount=request.form['Total Collection Amount']

    total_current_balance=request.form['Total Current Balance']

    total_revolving_credit=request.form['Total Revolving Credit Limit']

    # Create DataFrame from form data
    df = pd.DataFrame({
        'Loan Amount': [float(loan_amount)],
        'Funded Amount': [float(funded_amount)],
        'Funded Amount Investor': [float(funded_amount_investor)],
        'Term': [float(term)],
        'Interest Rate': [float(interest_rate)],
        'Grade': [float(Grade)],
        'Employment Duration': [float(employment_duration)],
        'Home Ownership': [float(home_ownership)],
        'Verification Status': [float(verification_status)],
        'Loan Title': [float(loan_title)],
        'Debit to Income': [float(debit_to_income)],
        'Delinquency - two years': [float(delinquency_two_years)],
        'Revolving Balance': [float(revolving_balance)],
        'Revolving Utilities': [float(revolving_utilities)],
        'Total Accounts': [float(total_accounts)],
        'Initial List Status': [float(initial_list_status)],
        'Total Received Interest': [float(total_received_interest)],
        'Total Received Late Fee': [float(total_received_late_fee)],
        'Recoveries': [float(recoveries)],
        'Collection Recovery Fee': [float(collection_recovery_fee)],
        'Collection 12 months Medical': [float(collection_12months_medical)],
        'Application Type': [float(application_type)],
        'Last week Pay': [float(last_week_pay)],
        'Total Collection Amount': [float(total_collection_amount)],
        'Total Current Balance': [float(total_current_balance)],
        'Total Revolving Credit Limit': [float(total_revolving_credit)]
    })



    prediction = model1.predict(df)
    print(prediction)
    prediction_result='Defaulter' if prediction[0]== 1 else 'Non-Defaulter'

    return render_template('result.html',pred=prediction_result)


if __name__ == '__main__':
    app.run(debug=True)