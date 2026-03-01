import csv
import os
import pandas as pd

class CSVLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_csv(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} does not exist.")
        
        try:
            data = pd.read_csv(self.file_path, encoding="utf-8-sig")
            return data
        except Exception as e:
            raise Exception(f"An error occurred while loading the CSV file: {e}")

if __name__ == "__main__":
    # Example usage
    file_path = r'data\csv_users\user_debt_spiral\Chequing_Account.csv'
    loader = CSVLoader(file_path)
    try:
        data = loader.load_csv()
        print(data.head())
    except Exception as e:
        print(e)