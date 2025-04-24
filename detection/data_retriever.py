from difflib import SequenceMatcher
import pymysql
import os
from dotenv import load_dotenv

class RDSDataFetcher:
    def __init__(self):
        self._load_env_variables()
        self.connection = self._connect_to_database()

    def _load_env_variables(self):
        load_dotenv()
        self.rds_host = os.getenv("rds_host")
        self.rds_user = os.getenv("rds_user")
        self.rds_password = os.getenv("rds_password")
        self.rds_port = int(os.getenv("rds_port", 3306))
        self.rds_dbname = os.getenv("rds_dbname")

    def _connect_to_database(self):
        try:
            return pymysql.connect(
                host=self.rds_host,
                user=self.rds_user,
                password=self.rds_password,
                port=self.rds_port,
                database=self.rds_dbname
            )
        except pymysql.MySQLError as e:
            print(f"Connection error: {e}")
            raise

    def _get_all_unique_ids(self, table_name):
        """Fetch all unique_ids from a table."""
        try:
            # Use backticks only around table names with special characters
            query = f"SELECT DISTINCT uniqueId FROM `{table_name}`"
            with self.connection.cursor() as cursor:
                cursor.execute(query)
                return [row[0] for row in cursor.fetchall()]
        except pymysql.MySQLError as e:
            print(f"Query error (unique_id list): {e}")
            return []

    def _find_closest_unique_id(self, input_id, table_name, threshold=0.8):
        """Find the closest unique_id in the specified table."""
        candidates = self._get_all_unique_ids(table_name)
        best_match = None
        best_ratio = 0.0

        for candidate in candidates:
            ratio = SequenceMatcher(None, input_id, candidate).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_match = candidate
                best_ratio = ratio

        return best_match

    def _get_latest_row(self, uniqueId, userId):
        """Fetch the latest matching row by unique_id and user_id."""
        table_name = 'row-data'
        try:
            with self.connection.cursor(pymysql.cursors.DictCursor) as cursor:
                query = f"""
                    SELECT uniqueId,box_number,invoice_number,box_quantity,part_number FROM `{table_name}`
                    WHERE uniqueId = %s AND userId = %s
                    ORDER BY updatedAt DESC LIMIT 1
                    
                """
                cursor.execute(query,(uniqueId, userId))
                return cursor.fetchone()
        except pymysql.MySQLError as e:
            print(f"Query error: {e}")
            return None
        
    def _get_box_dispatch_info(self, uniqueId, userId):
        """Fetch the latest matching row by unique_id and user_id."""
        table_name = 'dispatches'
        try:
            with self.connection.cursor(pymysql.cursors.DictCursor) as cursor:
                query = f"""
                    SELECT SUM(box_number) box_number,SUM(box_quantity) box_quantity FROM `{table_name}`
                    WHERE uniqueId = %s AND userId = %s
                    GROUP BY uniqueId
                    
                """
                cursor.execute(query,(uniqueId, userId))
                return cursor.fetchone()
        except pymysql.MySQLError as e:
            print(f"Query error: {e}")
            return None

    def fetch_closest_match(self, input_unique_id, user_id):
        """Main logic for matching with closeness check first in row-data."""

        closest_id = self._find_closest_unique_id(input_unique_id, "row-data")  # Fixed query for table name
        #print("closest id:", closest_id)
        if not closest_id:
            print("No close uniqueId match found in row-data.")
            return None

        #print(f"Closest uniqueId match found in row-data: {closest_id}")

        # Step 2: Try dispatches first
        dispatched_info = self._get_box_dispatch_info(closest_id, user_id)
        row_data_info = self._get_latest_row(closest_id, user_id)

        if dispatched_info:
            #print("Data found in dispatches.")

            #print("\n\nrow-data info:", row_data_info)
            dispatched_box_quantity = int(dispatched_info['box_quantity'])
            dispatched_box_number = int(dispatched_info['box_number'])


            #print("\n\nDispatched info:", dispatched_info)
            row_data_info['box_quantity'] = int(row_data_info['box_quantity']) - dispatched_box_quantity
            row_data_info['box_number'] = int(row_data_info['box_number']) - dispatched_box_number

            return row_data_info
        elif row_data_info:
            #print("Data didn't find in dispatches.")
            return row_data_info

        print("No matching row found in dispatches or row-data.")
        return None

    def close_connection(self):
        if self.connection:
            self.connection.close()
            #print("Connection closed.")

# Example usage
if __name__ == "__main__":
    fetcher = RDSDataFetcher()
    uniqueId = "@A1145"  # or whatever slightly incorrect/typo id
    userId = 1

    result = fetcher.fetch_closest_match(uniqueId, userId)
    print(result)

    fetcher.close_connection()
