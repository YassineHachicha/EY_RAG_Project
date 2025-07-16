from ml_pipeline import predict_from_input

exemple_input = {
    "Age": 42,
    "AnnualIncome": 60000,
    "CreditScore": 700,
    "Experience": 10,
    "LoanAmount": 20000,
    "LoanDuration": 36,
    "NumberOfDependents": 2,
    "MonthlyDebtPayments": 500,
    "CreditCardUtilizationRate": 0.3,
    "NumberOfOpenCreditLines": 3,
    "MonthlyIncome": 5000,
    "UtilityBillsPaymentHistory": 0.9,
    "JobTenure": 5,
    "NetWorth": 100000,
    "BaseInterestRate": 0.1,
    "InterestRate": 0.12,
    "MonthlyLoanPayment": 600,
    "TotalDebtToIncomeRatio": 0.2,

    # Catégorielles encodées (met 1 si présent)
    "EmploymentStatus_Self-Employed": 0,
    "EmploymentStatus_Unemployed": 0,
    "MaritalStatus_Married": 1,
    "MaritalStatus_Single": 0,
    "MaritalStatus_Widowed": 0,
    "HomeOwnershipStatus_Other": 0,
    "HomeOwnershipStatus_Own": 1,
    "HomeOwnershipStatus_Rent": 0,
    "LoanPurpose_Debt Consolidation": 1,
    "LoanPurpose_Education": 0,
    "LoanPurpose_Home": 0,
    "LoanPurpose_Other": 0
}

pred = predict_from_input(exemple_input, target="LoanApproved", model_type="xgboost")
print("🔮 Prediction LoanApproved:", pred)
