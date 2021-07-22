from pathlib import Path
import sqlite3
from sqlite3 import Error
from os import path
import datetime
import pandas as pd
import re
import string
import operator
from fuzzywuzzy import fuzz as fw_fuzz
from fuzzywuzzy import process as fw_process
from rapidfuzz import fuzz as rf_fuzz
from fuzzywuzzy import process as rf_process
from operator import itemgetter

FULL_PATH = path.realpath(__file__)
PATH_NAME = path.dirname(FULL_PATH)

DATABASE = "dummy.db"
TABLE1 = "reference"
TABLE2 = "dataset"

FUZZY_PACKAGE = "rapidfuzz"  # fuzzywuzzy
FUZZY_APPROACH = "fuzz"  # process

RESULT_PATH = PATH_NAME + f'/result/{FUZZY_PACKAGE}/{FUZZY_APPROACH}/'
Path(RESULT_PATH).mkdir(parents=True, exist_ok=True)


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


# Clean data
def clean_data(data, col):
    # remove punctuations in a column
    punctuations_except_single_quote = string.punctuation.replace("'", "")
    data[col] = data[col].str.replace("[{}]".format(punctuations_except_single_quote), " ", regex=True)
    data[col] = data[col].str.replace("'", "")
    # create a new column that removes all space in a col
    data[col + '_no_space'] = data[col].str.replace(" ", "")
    return data


# Replace foreign characters
def replace_foreign_characters(name):
    vowels = ['a', 'e', 'i', 'o', 'u']
    alternatives = ['àáâãäåæ', 'èéêë', 'ìíîï', 'ðòóôõöø', 'ùúûü']
    clean_name = ""
    for c in name:
        clean_name += c
        for i in range(len(vowels)):
            if c in alternatives[i]:
                clean_name = clean_name[:-1]
                clean_name += vowels[i]
    return clean_name


# Get reference table
def get_reference_table():
    sql = f"""SELECT "fake_company_name", "ein" 
                FROM {TABLE1} """
    column_names = ["fake_company_name", "ein"]
    data = extract_data_from_database(sql, column_names)
    data = clean_data(data=data, col="fake_company_name")
    print(f"-- {TABLE1} table extracted and cleaned. \n")
    return data


# Get dataset table
def get_dataset_table():
    sql = f"""SELECT "irregular_company_name", "ein" 
                FROM {TABLE2}  """
    column_names = ["irregular_company_name", "ein"]
    data = extract_data_from_database(sql, column_names)
    data = clean_data(data=data, col="irregular_company_name")
    data['irregular_company_name'] = data['irregular_company_name'].apply(lambda x: replace_foreign_characters(x))
    print(f"-- {TABLE2} table extracted and cleaned. \n")
    return data


# Get potential company name list
def get_potential_company_name_list(name, reference_table, reference_col, by):
    potential_names = []

    if name != '':
        if by == 'exact':
            selected_rows = reference_table[
                reference_table[reference_col].str.contains(name, flags=re.IGNORECASE, regex=True, na=False)]
            potential_names = selected_rows[reference_col].tolist()

        elif by == 'word':
            words = [word for word in name.split() if
                     len(word) > 1 and (word.lower() not in ('llc', 'ltd', 'co', 'inc'))]
            words_format = '|'.join(words)
            selected_rows = reference_table[
                reference_table[reference_col].str.contains(words_format, flags=re.IGNORECASE, regex=True, na=False)]
            potential_names = selected_rows[reference_col].tolist()

        elif by == 'nospace':
            selected_rows = reference_table[
                reference_table[reference_col].str.contains(name, flags=re.IGNORECASE, regex=True, na=False)]
            potential_names = selected_rows[reference_col].tolist()

        else:
            print("Keyword by is wrong, should be either name, word or nospace.")

    return list(set(potential_names))


# Return best match
def return_best_match(fuzzy_results):
    fuzzy_results.sort(key=itemgetter(1), reverse=True)
    matched_name, ratio = fuzzy_results[0]
    return matched_name, ratio


# Match by function in fuzzy wuzzy package
def match_by_function_in_fuzzywuzzy_package(name, potential_names, fuzzy_function):
    matched_name = ratio = None

    if FUZZY_APPROACH == "fuzz":

        if fuzzy_function == "ratio":
            fuzzy_results = [(s, fw_fuzz.ratio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        elif fuzzy_function == "partial_ratio":
            fuzzy_results = [(s, fw_fuzz.partial_ratio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        elif fuzzy_function == "token_sort_ratio":
            fuzzy_results = [(s, fw_fuzz.token_sort_ratio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        elif fuzzy_function == "token_set_ratio":
            fuzzy_results = [(s, fw_fuzz.token_set_ratio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        elif fuzzy_function == "partial_token_sort_ratio":
            fuzzy_results = [(s, fw_fuzz.partial_token_sort_ratio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        elif fuzzy_function == "partial_token_set_ratio":
            fuzzy_results = [(s, fw_fuzz.partial_token_set_ratio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        elif fuzzy_function == "QRatio":
            fuzzy_results = [(s, fw_fuzz.QRatio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        elif fuzzy_function == "WRatio":
            fuzzy_results = [(s, fw_fuzz.WRatio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        else:
            print("fuzzy_function parameter is out of range! \n")

    elif FUZZY_APPROACH == "process":

        if fuzzy_function == "ratio":
            matched_name, ratio = list(fw_process.extractOne(name, potential_names, scorer=fw_fuzz.ratio))

        elif fuzzy_function == "partial_ratio":
            matched_name, ratio = list(fw_process.extractOne(name, potential_names, scorer=fw_fuzz.partial_ratio))

        elif fuzzy_function == "token_sort_ratio":
            matched_name, ratio = list(fw_process.extractOne(name, potential_names, scorer=fw_fuzz.token_sort_ratio))

        elif fuzzy_function == "token_set_ratio":
            matched_name, ratio = list(fw_process.extractOne(name, potential_names, scorer=fw_fuzz.token_set_ratio))

        elif fuzzy_function == "partial_token_sort_ratio":
            matched_name, ratio = list(
                fw_process.extractOne(name, potential_names, scorer=fw_fuzz.partial_token_sort_ratio))

        elif fuzzy_function == "partial_token_set_ratio":
            matched_name, ratio = list(
                fw_process.extractOne(name, potential_names, scorer=fw_fuzz.partial_token_set_ratio))

        elif fuzzy_function == "QRatio":
            matched_name, ratio = list(fw_process.extractOne(name, potential_names, scorer=fw_fuzz.QRatio))

        elif fuzzy_function == "WRatio":
            matched_name, ratio = list(fw_process.extractOne(name, potential_names, scorer=fw_fuzz.WRatio))
        else:
            print("fuzzy_function parameter is out of range! \n")

    else:
        print("fuzzy_approach parameter is out of range! \n")

    return matched_name, ratio


# Match by function in rapid fuzz package
def match_by_function_in_rapidfuzz_package(name, potential_names, fuzzy_function):
    matched_name = ratio = None

    if FUZZY_APPROACH == "fuzz":

        if fuzzy_function == "ratio":
            fuzzy_results = [(s, rf_fuzz.ratio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        elif fuzzy_function == "partial_ratio":
            fuzzy_results = [(s, rf_fuzz.partial_ratio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        elif fuzzy_function == "token_sort_ratio":
            fuzzy_results = [(s, rf_fuzz.token_sort_ratio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        elif fuzzy_function == "token_set_ratio":
            fuzzy_results = [(s, rf_fuzz.token_set_ratio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        elif fuzzy_function == "partial_token_sort_ratio":
            fuzzy_results = [(s, rf_fuzz.partial_token_sort_ratio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        elif fuzzy_function == "partial_token_set_ratio":
            fuzzy_results = [(s, rf_fuzz.partial_token_set_ratio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        elif fuzzy_function == "QRatio":
            fuzzy_results = [(s, rf_fuzz.QRatio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        elif fuzzy_function == "WRatio":
            fuzzy_results = [(s, rf_fuzz.WRatio(name, s)) for s in potential_names]
            matched_name, ratio = return_best_match(fuzzy_results)

        else:
            print("fuzzy_function parameter is out of range! \n")

    elif FUZZY_APPROACH == "process":

        if fuzzy_function == "ratio":
            matched_name, ratio = list(rf_process.extractOne(name, potential_names, scorer=rf_fuzz.ratio))

        elif fuzzy_function == "partial_ratio":
            matched_name, ratio = list(rf_process.extractOne(name, potential_names, scorer=rf_fuzz.partial_ratio))

        elif fuzzy_function == "token_sort_ratio":
            matched_name, ratio = list(rf_process.extractOne(name, potential_names, scorer=rf_fuzz.token_sort_ratio))

        elif fuzzy_function == "token_set_ratio":
            matched_name, ratio = list(rf_process.extractOne(name, potential_names, scorer=rf_fuzz.token_set_ratio))

        elif fuzzy_function == "partial_token_sort_ratio":
            matched_name, ratio = list(
                rf_process.extractOne(name, potential_names, scorer=rf_fuzz.partial_token_sort_ratio))

        elif fuzzy_function == "partial_token_set_ratio":
            matched_name, ratio = list(
                rf_process.extractOne(name, potential_names, scorer=rf_fuzz.partial_token_set_ratio))

        elif fuzzy_function == "QRatio":
            matched_name, ratio = list(rf_process.extractOne(name, potential_names, scorer=rf_fuzz.QRatio))

        elif fuzzy_function == "WRatio":
            matched_name, ratio = list(rf_process.extractOne(name, potential_names, scorer=rf_fuzz.WRatio))
        else:
            print("fuzzy_function parameter is out of range! \n")

    else:
        print("fuzzy_approach parameter is out of range! \n")

    return matched_name, ratio


# Match by package and function
def fuzzy_match_by_package_and_function(name, potential_names, fuzzy_function):
    matched_name = ratio = None
    if FUZZY_PACKAGE == "fuzzywuzzy":
        matched_name, ratio = match_by_function_in_fuzzywuzzy_package(name, potential_names, fuzzy_function)
    elif FUZZY_PACKAGE == "rapidfuzz":
        matched_name, ratio = match_by_function_in_rapidfuzz_package(name, potential_names, fuzzy_function)
    else:
        print("fuzzy_function parameter is out of range! \n")
    return matched_name, ratio


# Get fuzzy match name ratio ein
def get_fuzzy_match_name_ratio_ein(row, reference_table, fuzzy_function):
    name = row['irregular_company_name']
    name_nospace = row['irregular_company_name_no_space']
    reference_col = "fake_company_name"

    result = []
    for by in ['exact', 'word', 'nospace']:
        sub_result = []

        if by == 'nospace':
            name = name_nospace
            reference_col += '_no_space'

        if name != '':
            potential_names = get_potential_company_name_list(name=name,
                                                              reference_table=reference_table,
                                                              reference_col=reference_col,
                                                              by=by)
            if potential_names:
                matched_name, ratio = fuzzy_match_by_package_and_function(name, potential_names, fuzzy_function)
                matched_salesforce_id = \
                    reference_table.loc[reference_table[reference_col] == matched_name, "ein"].values[
                        0]
                sub_result = [matched_name, ratio, matched_salesforce_id]

        if sub_result:
            result.append(sub_result)
    return result


# Evaluate performance
def evaluate_performance(data, fuzzy_function):
    matrix = {}

    num_total = data.shape[0]
    num_correct = data[data['Correct'] == True].shape[0]
    num_confident = data[data['Confident'] == True].shape[0]

    matrix['Package'] = FUZZY_PACKAGE
    matrix['Approach'] = FUZZY_APPROACH
    matrix['Function'] = fuzzy_function
    matrix['Total'] = num_total
    matrix['CorrectCount'] = num_correct
    matrix['ConfidentCount'] = num_confident

    df = pd.DataFrame([matrix])

    for col in ['Confident', 'Correct']:
        df[col + 'Percentage'] = round(df[col + 'Count'] / df['Total'], 4)

    df['CorrectOverConfidentPercentage'] = round(df['CorrectCount'] / df['ConfidentCount'],
                                                 4) if num_confident != 0 else None

    print("* SUMMARY *")
    print(f"Total: {num_total}")
    print(f"Correct: {num_correct} - {format(num_correct / num_total * 100, '.2f')}%")
    print(f"Confident: {num_confident} - {format(num_confident / num_total * 100, '.2f')}%")
    print(
        f"Correct over confident: {format(num_correct / num_confident * 100, '.2f') if num_confident != 0 else None}% \n")

    print("-- Report summary generated and saved. \n")

    return df


# Test single function
def test_single_function(fuzzy_function):
    time_start = datetime.datetime.now()

    reference_table = get_reference_table()
    data = get_dataset_table()

    # Fuzzy string match
    data['AllCandidates'] = data.apply(
        lambda row: get_fuzzy_match_name_ratio_ein(row, reference_table, fuzzy_function), axis=1)
    data['BestCandidate'] = data['AllCandidates'].apply(
        lambda x: sorted(x, key=operator.itemgetter(1), reverse=True)[0] if x != [] else x)

    data['BestMatchName'] = data['BestCandidate'].apply(lambda x: x[0] if x != [] else None)
    data['Ratio'] = data['BestCandidate'].apply(lambda x: x[1] if x != [] else None)
    data['SuggestedEIN'] = data['BestCandidate'].apply(lambda x: x[2] if x != [] else None)
    print("-- Fuzzy string matching to identify EIN. \n")

    # Check confidence and correctness
    data['Confident'] = data['Ratio'].apply(lambda x: True if x >= 90 else False)
    data['Correct'] = data.apply(
        lambda row: True if row['ein'] == row['SuggestedEIN'] else False, axis=1)

    data.drop(columns=['irregular_company_name_no_space', 'AllCandidates', 'BestCandidate'], inplace=True)

    data.sort_values(by=['Confident', 'Correct', 'Ratio'],
                     ascending=[True, True, True],
                     inplace=True)

    data.to_excel(
        RESULT_PATH + f"Identification_Result_{FUZZY_PACKAGE}_{FUZZY_APPROACH}_{fuzzy_function}.xlsx",
        index=False)

    # incorrect identification
    incorrect = data[data['Correct'] == False]
    if incorrect is not None:
        incorrect.to_excel(
            RESULT_PATH + f'Incorrect_Identification_{FUZZY_PACKAGE}_{FUZZY_APPROACH}_{fuzzy_function}.xlsx',
            index=False)

    num_shipments = data.shape[0]
    performance = evaluate_performance(data, fuzzy_function)

    time_end = datetime.datetime.now()

    time_used = time_end - time_start
    print(f"*** Time elapsed: {format_timedelta(time_used)} \n")
    print(f"*** Average time per shipment: {round((time_used / num_shipments).total_seconds(), 2)} seconds \n")

    performance['RunTimeInSecond'] = round(time_used.total_seconds())
    performance['AverageTimePerShipmentInSecond'] = round((time_used / num_shipments).total_seconds(), 2)
    return performance


# Test all functions
def test_all_functions():
    print(f"FUZZY PACKAGE: {FUZZY_PACKAGE}")
    print(f"FUZZY APPROACH: {FUZZY_APPROACH}")
    print()

    performance_dfs = []
    functions = ["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio",
                 "partial_token_sort_ratio", "partial_token_set_ratio", "QRatio", "WRatio"]

    for i in range(len(functions)):

        fuzzy_function = functions[i]
        print(f"Current function: {fuzzy_function} ({i + 1}/{len(functions)}) \n")

        performance = test_single_function(fuzzy_function)

        if performance is not None:
            if performance is not None:
                performance_dfs.append(performance)
                if performance_dfs:
                    result = pd.concat(performance_dfs, ignore_index=True)
                    result.to_excel(
                        RESULT_PATH + f'Performance_Summary_{FUZZY_PACKAGE}_{FUZZY_APPROACH}.xlsx',
                        index=False)
                else:
                    print("Performance summary dataframe is empty!")


def main():
    time_start = datetime.datetime.now()
    print(f"*** Script started at: {time_start.ctime()} \n")

    test_all_functions()

    time_end = datetime.datetime.now()
    print(f"*** Script finished at: {time_end.ctime()} \n")

    time_used = time_end - time_start
    print(f"*** Time elapsed: {format_timedelta(time_used)} \n")


if __name__ == "__main__":
    main()
