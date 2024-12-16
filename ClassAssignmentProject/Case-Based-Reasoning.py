import numpy as np
from collections import namedtuple

# Define case structure
Case = namedtuple('Case', 'name age gender ethnicity medical_history symptoms test_results diagnosis treatment')

# Expanded case base
case_base = [
    Case(name="Case1", age=58, gender="Female", ethnicity="African", medical_history="Hypertension",
         symptoms=["frequent urination", "fatigue"], test_results={"HbA1c": 6.8, "BP": [135, 85]},
         diagnosis="Type 2 Diabetes", treatment="Metformin, lifestyle changes"),

    Case(name="Case2", age=70, gender="Male", ethnicity="Caucasian", medical_history="Pre-diabetes",
         symptoms=["blurred vision", "shortness of breath"], test_results={"HbA1c": 7.5, "BP": [150, 90]},
         diagnosis="Type 2 Diabetes", treatment="Insulin, low-carb diet"),

    Case(name="Case3", age=50, gender="Female", ethnicity="Asian", medical_history="Hypertension",
         symptoms=["headache", "fatigue"], test_results={"HbA1c": 5.5, "BP": [140, 90]},
         diagnosis="Hypertension", treatment="ACE Inhibitor, reduce sodium intake"),

    # Additional cases
    Case(name="Case4", age=61, gender="Male", ethnicity="African-American", medical_history="Diabetes, Hypertension",
         symptoms=["chest pain", "dizziness"], test_results={"HbA1c": 8.2, "BP": [160, 95]},
         diagnosis="Type 2 Diabetes, Hypertension", treatment="Insulin, beta-blockers, lifestyle changes"),

    Case(name="Case5", age=39, gender="Female", ethnicity="Hispanic", medical_history="None",
         symptoms=["extreme hunger", "frequent urination", "weight loss"], test_results={"HbA1c": 6.1, "BP": [120, 80]},
         diagnosis="Pre-diabetes", treatment="Dietary modifications, increased physical activity"),

    Case(name="Case6", age=55, gender="Male", ethnicity="Caucasian", medical_history="Hypertension",
         symptoms=["fatigue", "blurred vision"], test_results={"HbA1c": 5.4, "BP": [150, 92]},
         diagnosis="Hypertension", treatment="Calcium channel blockers, reduce sodium intake"),

    Case(name="Case7", age=65, gender="Female", ethnicity="Asian", medical_history="Diabetes",
         symptoms=["frequent infections", "dry skin"], test_results={"HbA1c": 7.8, "BP": [135, 80]},
         diagnosis="Type 2 Diabetes", treatment="Metformin, lifestyle and dietary changes"),

    Case(name="Case8", age=40, gender="Male", ethnicity="African-American", medical_history="Pre-diabetes",
         symptoms=["increased thirst", "tingling in feet"], test_results={"HbA1c": 6.4, "BP": [125, 85]},
         diagnosis="Pre-diabetes", treatment="Dietary changes, regular exercise"),

    Case(name="Case9", age=75, gender="Male", ethnicity="Caucasian", medical_history="Diabetes, Hypertension",
         symptoms=["shortness of breath", "blurred vision", "fatigue"], test_results={"HbA1c": 7.0, "BP": [140, 90]},
         diagnosis="Type 2 Diabetes, Hypertension", treatment="Insulin, ACE Inhibitors, lifestyle modifications"),

    Case(name="Case10", age=52, gender="Female", ethnicity="Hispanic", medical_history="Hypertension",
         symptoms=["headache", "shortness of breath"], test_results={"HbA1c": 5.9, "BP": [155, 92]},
         diagnosis="Hypertension", treatment="Diuretics, DASH diet"),

    Case(name="Case11", age=47, gender="Male", ethnicity="Caucasian", medical_history="Obesity",
         symptoms=["extreme fatigue", "blurred vision"], test_results={"HbA1c": 6.9, "BP": [140, 88]},
         diagnosis="Type 2 Diabetes", treatment="Metformin, weight loss program"),

    Case(name="Case12", age=64, gender="Female", ethnicity="African", medical_history="None",
         symptoms=["dizziness", "nausea"], test_results={"HbA1c": 5.6, "BP": [128, 82]},
         diagnosis="Hypertension", treatment="Thiazide diuretic, low-salt diet"),

    Case(name="Case13", age=73, gender="Male", ethnicity="Asian", medical_history="Hypertension",
         symptoms=["chest discomfort", "shortness of breath"], test_results={"HbA1c": 6.3, "BP": [160, 92]},
         diagnosis="Hypertension, possible heart disease", treatment="ACE inhibitors, beta-blockers"),

    Case(name="Case14", age=60, gender="Female", ethnicity="Hispanic", medical_history="Diabetes, Hyperlipidemia",
         symptoms=["frequent urination", "extreme thirst"], test_results={"HbA1c": 8.3, "BP": [145, 90]},
         diagnosis="Type 2 Diabetes, Hyperlipidemia", treatment="Insulin, statins, lifestyle changes"),

    Case(name="Case15", age=49, gender="Male", ethnicity="African-American", medical_history="Pre-diabetes",
         symptoms=["fatigue", "increased hunger"], test_results={"HbA1c": 6.5, "BP": [130, 85]},
         diagnosis="Pre-diabetes", treatment="Dietary changes, exercise regimen"),

    Case(name="Case16", age=68, gender="Female", ethnicity="Caucasian", medical_history="Hypertension",
         symptoms=["headache", "blurry vision"], test_results={"HbA1c": 5.7, "BP": [155, 95]},
         diagnosis="Hypertension", treatment="Calcium channel blockers, dietary modifications"),

    Case(name="Case17", age=57, gender="Male", ethnicity="African", medical_history="Type 2 Diabetes",
         symptoms=["numbness in feet", "fatigue"], test_results={"HbA1c": 7.9, "BP": [138, 86]},
         diagnosis="Type 2 Diabetes", treatment="Metformin, lifestyle changes"),

    Case(name="Case18", age=62, gender="Female", ethnicity="Asian", medical_history="None",
         symptoms=["frequent headaches", "shortness of breath"], test_results={"HbA1c": 5.4, "BP": [150, 85]},
         diagnosis="Hypertension", treatment="Diuretics, DASH diet"),

    Case(name="Case19", age=80, gender="Male", ethnicity="Caucasian", medical_history="Diabetes, Hypertension",
         symptoms=["fatigue", "muscle weakness"], test_results={"HbA1c": 7.3, "BP": [140, 92]},
         diagnosis="Type 2 Diabetes, Hypertension", treatment="Insulin, ACE inhibitors, exercise"),

    Case(name="Case20", age=55, gender="Female", ethnicity="Hispanic", medical_history="Hypertension, Obesity",
         symptoms=["chest pain", "dizziness"], test_results={"HbA1c": 6.6, "BP": [160, 100]},
         diagnosis="Hypertension, possible coronary artery disease", treatment="Beta-blockers, statins, lifestyle changes"),

]


# Function to calculate similarity between cases
def calculate_similarity(new_case, stored_case):
    # Separate BP values and HbA1c for each case
    new_hba1c = new_case.test_results.get("HbA1c", 0)
    stored_hba1c = stored_case.test_results.get("HbA1c", 0)
    new_bp = new_case.test_results.get("BP", [0, 0])
    stored_bp = stored_case.test_results.get("BP", [0, 0])

    # Calculate Euclidean distance between HbA1c and BP values separately and sum them
    hba1c_distance = abs(new_hba1c - stored_hba1c)
    bp_distance = np.linalg.norm(np.array(new_bp) - np.array(stored_bp))
    numeric_similarity = hba1c_distance + bp_distance

    # Categorical similarity (Jaccard for symptoms)
    symptom_similarity = len(set(new_case.symptoms).intersection(set(stored_case.symptoms))) / \
                         len(set(new_case.symptoms).union(set(stored_case.symptoms)))

    # Weighted similarity: 50% weight for numerical and 50% for categorical
    weighted_similarity = 0.5 * (1 - numeric_similarity / 100) + 0.5 * symptom_similarity
    return weighted_similarity


# Function to retrieve the most similar case
def retrieve_similar_case(new_case, case_base):
    similarities = []
    for case in case_base:
        similarity = calculate_similarity(new_case, case)
        similarities.append(similarity)

    # Retrieve the most similar case
    most_similar_index = np.argmax(similarities)
    return case_base[most_similar_index], similarities[most_similar_index]


# Function to adapt the treatment for the new case based on the retrieved similar case
def adapt_case(most_similar_case, new_case):
    adapted_treatment = most_similar_case.treatment
    # Adapt treatment based on specific test results or symptoms in the new case
    if "HbA1c" in new_case.test_results and new_case.test_results["HbA1c"] > 7.0:
        adapted_treatment += ", consider insulin therapy"

    return adapted_treatment


# Function to run the CBR system for a new patient case
def run_cbr_system(new_case, case_base):
    # Step 1: Retrieve the most similar case
    similar_case, similarity_score = retrieve_similar_case(new_case, case_base)
    print("\nMost similar case found:")
    print(
        f"  Name: {similar_case.name}, Age: {similar_case.age}, Gender: {similar_case.gender}, Ethnicity: {similar_case.ethnicity}")
    print(f"  Medical History: {similar_case.medical_history}")
    print(f"  Symptoms: {similar_case.symptoms}")
    print(f"  Test Results: {similar_case.test_results}")
    print(f"  Diagnosis: {similar_case.diagnosis}")
    print(f"  Treatment: {similar_case.treatment}")
    print(f"  Similarity Score: {similarity_score:.2f}")

    # Step 2: Adapt the treatment
    adapted_treatment = adapt_case(similar_case, new_case)
    print("\nAdapted Treatment Recommendation for New Case:")
    print(adapted_treatment)


# Function to get user input for a new case
def get_user_input():
    name = input("Enter patient name: ")
    age = int(input("Enter age: "))
    gender = input("Enter gender (Male/Female): ")
    ethnicity = input("Enter ethnicity: ")
    medical_history = input("Enter medical history (e.g., Diabetes, Hypertension): ")
    symptoms = input("Enter symptoms separated by commas (e.g., frequent urination, fatigue): ").split(", ")

    # Get test results
    hba1c = float(input("Enter HbA1c level: "))
    bp_systolic = int(input("Enter Systolic Blood Pressure (e.g., 120): "))
    bp_diastolic = int(input("Enter Diastolic Blood Pressure (e.g., 80): "))
    test_results = {"HbA1c": hba1c, "BP": [bp_systolic, bp_diastolic]}

    return Case(name=name, age=age, gender=gender, ethnicity=ethnicity, medical_history=medical_history,
                symptoms=symptoms, test_results=test_results, diagnosis=None, treatment=None)


# Get user input to define a new patient case
new_case = get_user_input()

# Run the CBR system
run_cbr_system(new_case, case_base)