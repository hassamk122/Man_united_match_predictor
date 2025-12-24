from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QLabel, 
                             QComboBox, QPushButton, QTextEdit, QSpinBox, 
                             QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class MatchPredictorApp(QMainWindow):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor 
        self.setWindowTitle("Man United Match Predictor")
        self.setGeometry(100, 100, 800, 600)
        self.init_ui()
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        
        title = QLabel("⚽ Manchester United Match Predictor")
        title.setFont(QFont('Arial', 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #DA291C; padding: 10px;")
        main_layout.addWidget(title)
        

        accuracy_pct = self.predictor.accuracy * 100
        input_group = QGroupBox(f"Model Accuracy: {accuracy_pct:.2f}%")
        input_layout = QGridLayout()
        
        self.combos = {}
        for i, col in enumerate(self.predictor.categorical_columns):
            input_layout.addWidget(QLabel(f"{col.replace('_', ' ').title()}:"), i, 0)
            combo = QComboBox()
            combo.addItems(self.predictor.label_encoders[col].classes_)
            input_layout.addWidget(combo, i, 1)
            self.combos[col] = combo

        input_layout.addWidget(QLabel("Goals Scored:"), 5, 0)
        self.goals_scored_spin = QSpinBox()
        self.goals_scored_spin.setRange(0, 10)
        self.goals_scored_spin.setValue(1)
        input_layout.addWidget(self.goals_scored_spin, 5, 1)
        
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        
        self.predict_button = QPushButton("Predict Match Result")
        self.predict_button.setFont(QFont('Arial', 16, QFont.Bold))
        self.predict_button.setStyleSheet("background-color: #DA291C; color: white; padding: 10px; border-radius: 5px;")
        self.predict_button.clicked.connect(self.handle_prediction)
        main_layout.addWidget(self.predict_button)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont('Courier', 10))
        main_layout.addWidget(self.results_text)

    def handle_prediction(self):
        input_data = {col: self.combos[col].currentText() for col in self.predictor.categorical_columns}
        input_data['goals_scored'] = self.goals_scored_spin.value()
        
        result, probs = self.predictor.predict(input_data)
        
        output = "=" * 40 + "\n PREDICTION: " + result.upper() + "\n" + "=" * 40 + "\n"
        for res, p in probs.items():
            output += f"{res.capitalize():10s}: {p:.2%} \n"
        
        self.results_text.setText(output)