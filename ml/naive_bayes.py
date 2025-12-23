#libraries used
import pandas as pd
import numpy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#reading data from csv
matches_data = pd.read_csv("data/manuited_dataset.csv")


# Function to calculate actual probabilities from dataset
def calculate_actual_probabilities(dataframe, target_column='result'):
    result_counts = dataframe[target_column].value_counts()
    total_matches = len(dataframe)

    print(f"\nTotal matches: {total_matches}\n")
    
    probabilities = {}
    for result, count in result_counts.items():
        probability = count / total_matches
        probabilities[result] = probability
        print(f"{result.capitalize()}: {count} matches ({probability:.2%})")
    
    return probabilities

# Calculate and display actual probabilities from your dataset
actual_probs = calculate_actual_probabilities(matches_data)
print("="*50,"\n")



#Dropping opponents(names) as they add no value
matches_data = matches_data.drop('opponent',axis=1)

# ENCODE FIRST - before splitting into X and Y
label_encoders = {}
categorical_columns = ['home/away', 'match_time','opponent_strength_tier', 'recent_form', 'competition']

#Encoding object columns into encoding
for column in categorical_columns:
    label_encoder = LabelEncoder()
    matches_data[column] = label_encoder.fit_transform(matches_data[column])
    label_encoders[column] = label_encoder  # Fixed: was label_encoders[column] = label_encoders

#Encoding object column(target) into encoding
label_encoder_result = LabelEncoder()
matches_data['result'] = label_encoder_result.fit_transform(matches_data['result'])


#To see the mapping
for i, class_name in enumerate(label_encoder_result.classes_):
    print(f"{class_name} → class  {i}")
print("="*50,"\n")

# NOW split into X and Y (data is already encoded)
X = matches_data.drop('result',axis=1)
Y = matches_data['result']

#Splitting data into training and testing(80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#using gaussian naive bayes classifier
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, Y_train)

#make prediction on test set
Y_predict = naive_bayes_classifier.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(Y_test, Y_predict))
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_predict))
print("="*50,"\n")






# Match to predict
new_match = pd.DataFrame({
    'home/away': ['away'],
    'match_time': ['night'],
    'opponent_strength_tier': ['Top-tier'],
    'goals_scored': [1],
    'recent_form': ['Average'],
    'competition': ['Premier League']
})

print(new_match.to_string(index=False))


# Encode the new data using the same encoders
for col in categorical_columns:
    new_match[col] = label_encoders[col].transform(new_match[col])

prediction = naive_bayes_classifier.predict(new_match)
predicted_result = label_encoder_result.inverse_transform(prediction)
print(f"\nPredicted match result: {predicted_result[0]}")

# Get prediction probabilities
probabilities = naive_bayes_classifier.predict_proba(new_match)
print("\nNew Match Prediction probabilities:\n")
for result, prob in zip(label_encoder_result.classes_, probabilities[0]):
    print(f"{result}: {prob:.4f}")