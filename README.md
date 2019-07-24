# Bank-loan-prediction

Problem statement

Goal
Understand the loan business of the bank.
Description
A Bank want to understand its loan business. It wants to identify who would be the right target for the loans. The data contains both customers who have been given loans and those who haven’t yet received loans. The aim is to identify of the customers who haven’t been granted a loan which are more likely to repay the loan and provide loans to them.
Data
    1. Borrower_table - info about borrowers
    2. Loan_table – loan details
Questions
    1. Perform an EDA of the features in the data. Describe any interesting insight you might get when you go through the data.
    2. Implement a model to make maximum profit for the bank when providing loans. Please implement atleast 2 algorithms of your choice. Explain the reason for choosing the algorithms.
    
    
Files:

1. Pred_final_data(1).csv has the entire table along with the predicted class, loan ids and various training parameters
2. final_loan_id.csv contains the loan_id s of people with predicted clas=1

#Notes:
1. Two algorithms were used here as per the problem statement- GBM and Random Forest
2. Due to better results model created in  GBM was used to predict the final data set
The file ‘pred_final_data.csv’ contains the final data of the test set along with an added column of pred_loan_repaid which denotes if the person would have paid paid the loan or not had he/she been granted one
Since we are interested in finding out the customers who would have paid the loan back if they were granted a loan, the file ‘final loan id.csv’ contains the loan ids with the predicted class=1.0, denoting the people who would have paid the loan back- making them our target customers

#Choosing the target class
Here we have chosen the column ‘ loan_repaid’ to be our target class
In our sample data I have taken the data of the customers who have been granted a loan and modelled if they have repaid it back or not
Test data consists of the customers who have not been granted a loan. Here I predict the customers who if granted, would’ve repaid the loan back because these are the customers who have been missed out and can be used to increase the revenue of the bank

#Data Analysis and preprocessing
First I matched the two datasets given on their loanid and prepared the master table
Removed irrelevant columns such as date
Plotted the distribution of age and categorised it in three major categories
Removed the continuous distribution of age and replaced it with categories


#Preparing the dataset
Classified age into 3 different categories based on the histogram plot
Converted the categorical value of ‘loan purpose’ to dummy columns with binary entries
Test data contains the value where loan repaid = NaN
Rest values where loan has been granted is considered as training
Dropped the column ‘loan granted’ from both the datasets
Keeping the test data aside for now, prediction models were applied to learn from the training data after applying a 70:30 split to i


