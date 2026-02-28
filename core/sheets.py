import gspread
from oauth2client.service_account import ServiceAccountCredentials
import logging

def get_sheets_client(credentials_path):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        logging.error(f"Failed to connect to Google Sheets: {e}")
        return None

def log_prediction_to_sheets(client, sheet_name, data):
    try:
        sheet = client.open(sheet_name).sheet1
        sheet.append_row(data)
    except Exception as e:
        logging.error(f"Failed to log data to Google Sheets: {e}")
