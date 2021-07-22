import sqlite3
from sqlite3 import Error
from os import path, listdir
import random
import datetime

import pandas as pd

FULL_PATH = path.realpath(__file__)
PATH_NAME = path.dirname(FULL_PATH)
DATA_PATH = PATH_NAME + f'/company_name_data/'

DATABASE = "dummy.db"
TABLE1 = "reference"


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


# Load dummy data into sql database
def load_dummy_data_into_database():
    conn = create_connection(DATABASE)

    random.seed(0)
    sheets = sorted(listdir(DATA_PATH))
    dfs = []

    for sheet in sheets:
        if sheet.startswith('faux_id_fake_companies') and sheet.endswith('.csv'):
            no = sheet.split('.')[0][-2:]
            df = pd.read_csv(DATA_PATH + sheet)
            df['new_id'] = no + '-' + df['id'].astype(str)
            df = df[['new_id', 'fake-company-name', 'ein']]
            dfs.append(df)
            print(f"{sheet} loaded. \n")

    result = pd.concat(dfs, ignore_index=True)

    result.rename(columns={"fake-company-name": "fake_company_name"}, inplace=True)
    result.drop_duplicates(subset=['ein'], keep='first')
    result.drop_duplicates(subset=['fake_company_name'], keep='first', inplace=True)

    print(f"Shape: {result.shape} \n")
    print(result.head())
    result.to_sql(f'{TABLE1}', con=conn, if_exists='replace', index=False)

    conn.close()


def main():
    time_start = datetime.datetime.now()
    print(f"*** Script started at: {time_start.ctime()} \n")

    load_dummy_data_into_database()

    time_end = datetime.datetime.now()
    print(f"*** Script finished at: {time_end.ctime()} \n")

    time_used = time_end - time_start
    print(f"*** Time elapsed: {format_timedelta(time_used)} \n")


if __name__ == "__main__":
    main()
