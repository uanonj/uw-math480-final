import collections

DATA_FILE = "adult_data_set/adult.data"
TEST_FILE = "adult_data_set/adult.test"

UNKNOWN = "?"

DELIMETER = ", "

NUM_ATTRIBUTES = 14

TARGET_VALUES = [">50K", "<=50K"]

def age(value):
    """
    Records age.
    Values: 0 (20-29), 1 (30-39), 3 (40-49), 4 (50-59), 5 (60+)
    """
    age = int(value) / 10
    if age > 5:
        age = 5
    return age

def fnlwgt(value):
    """
    Records final weight (see http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names for more info)
    Values: 0 (0-99,999), 1 (100,000-199,999), 2 (200,000-299,999), 3 (300,000-399,999), 4 (400,000+)
    """
    final_weight = int(value) / 100000
    if final_weight > 4:
        final_weight = 4
    return final_weight

def education_num(value):
    """
    Records education number (years)
    Values: 0-16
    """
    return int(value)

def capital_gain(value):
    """
    Records capital gain.
    Values: 0 (0),  1 (1-2999), 2 (3000-9999), 3 (10000-79999),
            4 (8000+)
    """
    cap_gain = int(value) / 1000
    if cap_gain > 4:
        cap_gain = 4
    return cap_gain

def capital_loss(value):
    """
    Records capital loss.
    Values: 0 (0), 1 (1-999), 2 (1000-1999), 3 (2000-2999), 4 (3000-3999), 5 (4000+)
    """
    cap_loss = int(value) / 1000
    if cap_loss > 5:
        cap_loss = 5
    return cap_loss

def hours_per_week(value):
    """
    Records houres per week.
    Values: 0-10 (i.e., 0 (0-9), 1 (10-19), ..., 9 (90+)
    """
    hours = int(value) / 10
    if hours > 9:
        hours = 9
    return hours

ATTRIBUTES = range(NUM_ATTRIBUTES)

# Attributes with special handlers
ATTRIBUTE_HANDLERS = { 0 : age,
               2 : fnlwgt,
               4 : education_num,
               10: capital_gain,
               11: capital_loss,
               12: hours_per_week }

VALUES = { 0 : range(6),
           1 : ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"],
           2 : range(5),
           3 : ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
           4 : range(17),
           5 : ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
           6 : ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"],
           7 : ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
           8 : ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
           9 : ["Female", "Male"],
           10: range(5),
           11: range(6),
           12: range(10),
           13: ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"] }

def identity_func(value):
    """Identity function serving as a default for the
    ATTRIBUTES.get processing lookup"""
    return value

def classify(attribute, value):
    """Classifies the given attribute, value pair"""
    return ATTRIBUTE_HANDLERS.get(attribute, identity_func)(value)

def process_line(line):
    """processes the line into an array of categorized data"""
    line_data = line.split(DELIMETER)
    for i in range(len(line_data) - 1):
        line_data[i] = classify(i, line_data[i])
    return line_data
        
    
