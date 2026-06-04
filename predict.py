import joblib

# 1. Load the saved model
try:
    model = joblib.load('document_classifier.pkl')
except FileNotFoundError:
    print("Error: Model not found. Please run train.py first.")
    exit()

# Target categories corresponding to the training data
target_names = ['Computer Graphics', 'Space Science', 'Religion', 'Middle East Politics']

print("--- Document Classifier Active ---")
print("Type 'exit' to quit.\n")

# 2. Create an interactive loop for the user
while True:
    new_text = input("Enter a sentence or document snippet:\n> ")
    
    if new_text.lower() == 'exit':
        break
        
    if not new_text.strip():
        continue

    # 3. Predict the category
    # The pipeline automatically handles the TF-IDF vectorization for us
    prediction_index = model.predict([new_text])[0]
    predicted_category = target_names[prediction_index]

    print(f"--> Predicted Category: {predicted_category}\n")