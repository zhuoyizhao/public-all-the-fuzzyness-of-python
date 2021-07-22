import sqlite3
from sqlite3 import Error
from os import path
import string
import random
import datetime
import pandas as pd

FULL_PATH = path.realpath(__file__)
PATH_NAME = path.dirname(FULL_PATH)
DATA_PATH = PATH_NAME + f'/company_name_data/'

DATABASE = "dummy.db"
TABLE1 = "reference"
TABLE2 = "dataset"
NUM_DESIRED = 10000


# Format time duration
def format_timedelta(duration):
    seconds = duration.total_seconds()
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f'{h} hours, {m} minutes, {s} second'


# Connect to database
def create_connection(db):
    conn = None
    try:
        conn = sqlite3.connect(db)
        return conn
    except Error as e:
        print(e)
    return conn


# Extract data from database
def extract_data_from_database(sql, column_names):
    db_conn = create_connection(DATABASE)
    db_cur = db_conn.cursor()
    db_cur.execute(sql)
    sql_data = db_cur.fetchall()
    db_cur.close()
    df = pd.DataFrame(sql_data, columns=column_names)
    return df


# Generate ill-formatted company names using basic rules
def basic_rules(company_name):
    result = []

    # Rule 1
    # If fake-company-name has a punctuation (e.g., quote, dash, comma),
    # Remove the punctuation or replace it with a space.
    punctuations = string.punctuation
    for punctuation in punctuations:
        if punctuation in company_name:
            result.append(company_name.replace(punctuation, ""))
            result.append(company_name.replace(punctuation, " "))

    # Rule 2
    # If fake-company-name has a suffix like LLC, LTD, Inc,
    # Remove the suffix.
    suffix = ["llc", "ltd", "inc"]
    for suf in suffix:
        if suf in company_name.lower():
            result.append(company_name.replace(suf, ""))

    # Rule 3
    # If fake-company-name only has two words,
    # Remove the space between the two words and make them into one word.
    if len(company_name) == 2:
        result.append(company_name.replace(" ", ""))

    # Rule 4
    # Randomly change the case of the word,
    # Make them all lower case or all upper case.
    result.append(company_name.lower())
    result.append(company_name.upper())

    # Rule 5
    # If fake-company-name has a length larger than 15,
    # Keep only some consecutive words to make the name shorter.
    if len(company_name) >= 15:
        words = company_name.split()
        for i in range(len(words) - 1):
            if len(words[i] + words[i + 1]) < 15 and (words[i] != "and" and words[i + 1] != "and"):
                result.append(words[i] + ' ' + words[i + 1])

    # Rule 6
    # If fake-company-name has “and”,
    # Remove “and”.
    result.append(company_name.replace("and", ""))
    return result


# Generate random indices
def generate_random_indices(company_name):
    k = random.randrange(0, len(company_name) // 5)
    indices = random.sample(range(0, len(company_name)), k)
    return indices


# Generate ill-formatted company names using random rules
def random_rules(company_name):
    result = []

    # randomly insert a punctuation in the name
    punctuations = string.punctuation
    punctuations = punctuations[:23] + punctuations[24:]
    i = random.randrange(0, len(punctuations))
    j = random.randrange(0, len(company_name))
    punc_added = company_name[:j] + punctuations[i] + company_name[j:]
    result.append(punc_added)

    # misspellings
    if "ie" in company_name:
        result.append(company_name.replace("ie", "ei"))
    if "ant" in company_name:
        result.append(company_name.replace("ant", "ent"))
    if "ar" in company_name:
        result.append(company_name.replace("ar", "er"))
    if "f" in company_name:
        result.append(company_name.replace("f", "ph"))

    # randomly missing letter
    k = random.randrange(len(company_name) // 2, len(company_name))
    missing = ""
    for i in range(0, len(company_name)):
        if i == 0 or i % k != 0:
            missing += company_name[i]
    result.append(missing)

    # randomly replace with letter x
    indices = generate_random_indices(company_name)
    replaced = ""
    for i in range(len(company_name)):
        if i in indices:
            replaced += "x"
        else:
            replaced += company_name[i]
    result.append(replaced)

    # attach a random foreign character to each vowel
    foreign_name = company_name
    vowels = ['a', 'e', 'i', 'o', 'u']
    alternatives = ['àáâãäåæ', 'èéêë', 'ìíîï', 'ðòóôõöø', 'ùúûü']
    for i in range(len(vowels)):
        if vowels[i] in foreign_name:
            k = random.randrange(0, len(alternatives[i]))
            foreign_name = foreign_name.replace(vowels[i], alternatives[i][k])
    result.append(foreign_name)
    return result


# Populate dataset table
def populate_dataset_table():
    # Get reference table
    def get_reference_table():
        sql = f"""SELECT "fake_company_name", "ein" 
                    FROM reference """
        column_names = ["fake_company_name", "ein"]
        data = extract_data_from_database(sql, column_names)
        print("-- reference table loaded from database. \n")
        return data

    # Create irregular company name list
    def create_irregular_company_name_list(company_name):
        common_rule_list = basic_rules(company_name)
        misspellings_list = random_rules(company_name)
        result = common_rule_list + misspellings_list
        return list(set(result))

    # Randomly select company names
    def randomly_select_company_names(row):
        result = []
        num_names = len(row['irregular_name_list'])
        num_selected = random.randrange(0, num_names)
        name_indices = random.sample(range(num_names), num_selected)
        for idx in name_indices:
            num_repeat = random.randrange(0, num_names)
            for j in range(num_repeat):
                result.append((row['irregular_name_list'][idx], row['ein']))
        return result

    random.seed(0)

    data = get_reference_table()

    data['irregular_name_list'] = data['fake_company_name'].apply(lambda x: create_irregular_company_name_list(x))
    print("-- Irregular company names generated. \n")

    records = []
    rows = data.to_dict('records')
    for row in rows:
        record = randomly_select_company_names(row)
        records += record
    print("-- Irregular company names randomly selected. \n")

    # under sample the dataset to a smaller one
    indices = random.sample(range(0, len(records)), NUM_DESIRED)
    indices.sort()

    final = []
    for i in range(len(records)):
        if i in indices:
            final.append(records[i])
    print("-- Under sampling to 10,000 rows. \n")

    df = pd.DataFrame(final, columns=['irregular_company_name', 'ein'])
    df['no'] = range(1, 1 + len(df))
    df = df[['no', 'irregular_company_name', 'ein']]

    conn = create_connection(DATABASE)
    df.to_sql(f'{TABLE2}', con=conn, if_exists='replace', index=False)
    conn.close()
    print("-- Dataset table saved to database. \n")


def main():
    time_start = datetime.datetime.now()
    print(f"*** Script started at: {time_start.ctime()} \n")

    populate_dataset_table()

    time_end = datetime.datetime.now()
    print(f"*** Script finished at: {time_end.ctime()} \n")

    time_used = time_end - time_start
    print(f"*** Time elapsed: {format_timedelta(time_used)} \n")


if __name__ == "__main__":
    main()
