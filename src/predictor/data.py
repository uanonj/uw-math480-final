import collections

DATA_FILE = "adult_data_set/adult.data"
TEST_FILE = "adult_data_set/adult.test"

UNKNOWN = "?"

DELIMETER

NUM_ATTRIBUTES = 14

TARGET_VALUES = [">50K", "<=50K"]

def age(value, target, attr_totals):
    """
    Records age.
    Values: 0 (20-29), 1 (30-39), 3 (40-49), 4 (50-59), 5 (60+)
    """
    age = int(value) / 10
    if age > 5:
        age = 5
    attr_totals[age][target] += 1

def workclass(value, target, attr_totals):
    """
    """
    attr_totals[value][target] += 1

def fnlwgt(value, target, attr_totals):
    """
    Records final weight (see http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names for more info)
    Values: 0 (0-99,999), 1 (100,000-199,999), 2 (200,000-299,999), 3 (300,000-399,999), 4 (400,000+)
    """
    final_weight = int(value) / 100000
    if final_weight > 4:
        final_weight = 4
    attr_totals[final_weight][target] += 1

def education(value, target, attr_totals):
    """
    """
    attr_totals[value][target] += 1

def education_num(value, target, attr_totals):
    """
    Records education number (years)
    Values: 0-16
    """
    attr_totals[int(value)][target] += 1

def marital_status(value, target, attr_totals):
    """
    """
    attr_totals[value][target] += 1

def occupation(value, target, attr_totals):
    """
    """
    attr_totals[value][target] += 1

def relationship(value, target, attr_totals):
    """
    """
    attr_totals[value][target] += 1

def race(value, target, attr_totals):
    """
    """
    attr_totals[value][target] += 1

def sex(value, target, attr_totals):
    """
    """
    attr_totals[value][target] += 1

def capital_gain(value, target, attr_totals):
    """
    Records capital gain.
    Values: 0 (0),  1 (1-2999), 2 (3000-9999), 3 (10000-79999),
            4 (8000+)
    """
    cap_gain = int(value) / 1000
    if cap_gain > 4:
        cap_gain = 4
    attr_totals[cap_gain][target] += 1

def capital_loss(value, target, attr_totals):
    """
    Records capital loss.
    Values: 0 (0), 1 (1-999), 2 (1000-1999), 3 (2000-2999), 4 (3000-3999), 5 (4000+)
    """
    cap_loss = int(value) / 1000
    if cap_loss > 5:
        cap_loss = 5
    attr_totals[cap_loss][target] += 1

def hours_per_week(value, target, attr_totals):
    """
    Records houres per week.
    Values: 0-10 (i.e., 0 (0-9), 1 (10-19), ..., 9 (90+)
    """
    hours = int(value) / 10
    if hours > 9:
        hours = 9
    attr_totals[hours][target] += 1

def native_country(value, target, attr_totals):
    """
    """
    attr_totals[value][target] += 1

ATTRIBUTES = { 0 : age,
               1 : workclass,
               2 : fnlwgt,
               3 : education,
               4 : education_num,
               5 : marital_status,
               6 : occupation,
               7 : relationship,
               8 : race,
               9 : sex,
               10: capital_gain,
               11: capital_loss,
               12: hours_per_week,
               13: native_country }

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

def initialize_totals():
    """
    Initializes a totals container suitable for the
    given data set
    """
    totals = {}
    for key in ATTRIBUTES.keys():
        totals[key] = {}
        for attr_value in VALUES[key]:
            totals[key][attr_value] = collections.Counter()
    return totals

def initialize_attribute_counts():
    return collections.Counter()

def classify(attr, value, target, totals, attribute_counts):
    if value != UNKNOWN:
        ATTRIBUTES[attr](value, target, totals[attr])
        attribute_counts[attr] += 1

def process_line(line):
    line_data = line.strip().split(DELIMETER)
    for i in range(len(line_data) - 1):
        line_data[i] = classify(i, line_data[i])
        
    
