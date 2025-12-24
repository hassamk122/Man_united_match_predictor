import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

class MatchPredictorModel:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.classifier = GaussianNB()
        self.label_encoders = {}
        self.label_encoder_result = None
        self.categorical_columns = ['home/away', 'match_time', 'opponent_strength_tier', 
                                    'recent_form', 'competition']
        self.feature_names = [] 
        self.accuracy = 0.0
        self.train_model()

    def train_model(self):
        matches_data = pd.read_csv(self.csv_path)
        matches_data = matches_data.drop('opponent', axis=1)
        
        # Encode categorical columns
        for column in self.categorical_columns:
            le = LabelEncoder()
            matches_data[column] = le.fit_transform(matches_data[column])
            self.label_encoders[column] = le
        
        self.label_encoder_result = LabelEncoder()
        matches_data['result'] = self.label_encoder_result.fit_transform(matches_data['result'])
        
        X = matches_data.drop('result', axis=1)
        self.feature_names = X.columns.tolist()
        
        Y = matches_data['result']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        self.classifier.fit(X_train, Y_train)
        
        Y_predict = self.classifier.predict(X_test)
        self.accuracy = accuracy_score(Y_test, Y_predict)

    def predict(self, input_dict):
        new_match = pd.DataFrame([input_dict])
        
        for col in self.categorical_columns:
            new_match[col] = self.label_encoders[col].transform(new_match[col])
        
        new_match = new_match[self.feature_names]
        
        # Get prediction and probabilities
        prediction = self.classifier.predict(new_match)
        predicted_result = self.label_encoder_result.inverse_transform(prediction)[0]
        probabilities = self.classifier.predict_proba(new_match)[0]
        

        prob_map = dict(zip(self.label_encoder_result.classes_, probabilities))
        
        return predicted_result, prob_map