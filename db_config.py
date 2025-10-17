import mysql.connector
import pandas as pd
import time
from typing import List, Dict, Tuple, Any


class DatabaseManager:
    def __init__(self, host="localhost", user="root", password="", database="companies", port=3306):
        self.connection_config = {
            'host': host, 'user': user, 'password': password, 
            'database': database, 'port': port, 'autocommit': False
        }
        print("DatabaseManager initialized")

    def _get_connection(self):
        try:
            conn = mysql.connector.connect(**self.connection_config)
            return conn
        except Exception as e:
            print(f"Connection failed: {e}")
            raise e

    def _ensure_initialized(self):
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS companies (
                    cin VARCHAR(25) PRIMARY KEY,
                    company_name TEXT,
                    company_roc_code VARCHAR(10),
                    company_category VARCHAR(50),
                    company_sub_category VARCHAR(50),
                    company_class VARCHAR(50),
                    authorized_capital VARCHAR(100),
                    paidup_capital VARCHAR(100),
                    registration_date DATE,
                    registered_office_address TEXT,
                    listing_status VARCHAR(50),
                    company_status VARCHAR(50),
                    company_state_code VARCHAR(10),
                    company_type VARCHAR(50),
                    nic_code VARCHAR(10),
                    company_industrial_classification VARCHAR(100),
                    row_hash VARCHAR(32),
                    sync_flag VARCHAR(20),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS companies_archive (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    cin VARCHAR(25),
                    company_name TEXT,
                    company_roc_code VARCHAR(10),
                    company_category VARCHAR(50),
                    company_sub_category VARCHAR(50),
                    company_class VARCHAR(50),
                    authorized_capital VARCHAR(100),
                    paidup_capital VARCHAR(100),
                    registration_date DATE,
                    registered_office_address TEXT,
                    listing_status VARCHAR(50),
                    company_status VARCHAR(50),
                    company_state_code VARCHAR(10),
                    company_type VARCHAR(50),
                    nic_code VARCHAR(10),
                    company_industrial_classification VARCHAR(100),
                    row_hash VARCHAR(32),
                    sync_flag VARCHAR(20),
                    archived_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    archive_reason VARCHAR(100)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS change_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    cin VARCHAR(25),
                    company_name TEXT,
                    company_roc_code VARCHAR(10),
                    company_category VARCHAR(50),
                    company_sub_category VARCHAR(50),
                    company_class VARCHAR(50),
                    authorized_capital VARCHAR(100),
                    paidup_capital VARCHAR(100),
                    registration_date DATE,
                    registered_office_address TEXT,
                    listing_status VARCHAR(50),
                    company_status VARCHAR(50),
                    company_state_code VARCHAR(10),
                    company_type VARCHAR(50),
                    nic_code VARCHAR(10),
                    company_industrial_classification VARCHAR(100),
                    row_hash VARCHAR(32),
                    sync_flag VARCHAR(20),
                    change_type VARCHAR(50),  -- 'NEW', 'REMOVED', 'UPDATED'
                    change_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_cin (cin),
                    INDEX idx_change_type (change_type),
                    INDEX idx_timestamp (change_timestamp)
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            print("Complete tables created with full change log")
        except Exception as e:
            print(f"Table creation failed: {e}")

    def replace_companies_from_dataframe(self, df):
        """BULK INSERT with proper error handling"""
        print(f"DATABASE METHOD CALLED - Input shape: {df.shape}")
        print(f"Columns received: {list(df.columns)}")
        
        conn = None
        cursor = None
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            print("Sample data check:")
            for i in range(min(3, len(df))):
                row = df.iloc[i]
                print(f"  Row {i}: CIN={row.get('cin', 'MISSING')}, Name={row.get('company_name', 'MISSING')}")
           
            cursor.execute("START TRANSACTION")
            
            cursor.execute("TRUNCATE TABLE companies")
            print("Cleared existing data")
         
            data_tuples = []
            for _, row in df.iterrows():
                data_tuples.append(tuple(
                    row[col] if pd.notna(row[col]) else "" 
                    for col in [
                        'cin', 'company_name', 'company_roc_code', 'company_category', 'company_sub_category',
                        'company_class', 'authorized_capital', 'paidup_capital', 'registration_date',
                        'registered_office_address', 'listing_status', 'company_status', 'company_state_code',
                        'company_type', 'nic_code', 'company_industrial_classification', 'row_hash', 'sync_flag'
                    ]
                ))
            
            total_records = len(data_tuples)
            print(f"Prepared {total_records:,} records for insertion")
        
            sql = """
                INSERT INTO companies (
                    cin, company_name, company_roc_code, company_category, company_sub_category,
                    company_class, authorized_capital, paidup_capital, registration_date,
                    registered_office_address, listing_status, company_status, company_state_code,
                    company_type, nic_code, company_industrial_classification, row_hash, sync_flag
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            batch_size = 1000
            inserted = 0
            
            for i in range(0, total_records, batch_size):
                batch = data_tuples[i:i + batch_size]
                cursor.executemany(sql, batch)
                inserted += len(batch)
                
                if (i // batch_size) % 10 == 0:
                    progress = (inserted / total_records) * 100
                    print(f"⏳ {progress:.1f}% - {inserted:,}/{total_records:,} records")
            
            # Commit
            conn.commit()
            print(f"BULK INSERT COMPLETE: {inserted:,} records")
            
            return inserted
            
        except Exception as e:
            print(f"DATABASE ERROR: {e}")
            if conn:
                conn.rollback()
                print("🔄 Transaction rolled back")
            raise e
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_all_companies(self):
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM companies")
            result = cursor.fetchall()
            return result
        except Exception as e:
            print(f"get_all_companies failed: {e}")
            return []
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_statistics(self):
        """Get basic stats"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM companies")
            total = cursor.fetchone()[0]
            return {
                'total_companies': total,
                'total_changes': 0, 
                'new_incorporations': 0
            }
        except:
            return {'total_companies': 0, 'total_changes': 0, 'new_incorporations': 0}
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def bulk_log_changes(self, changes_data: List[Tuple[str, str, str, str, str]]) -> int:
        if not changes_data:
            return 0
            
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("START TRANSACTION")
            
            logged_count = 0
            for cin, change_type, field_name, old_value, new_value in changes_data:
                try:
                    # For complete schema, we need to provide all required fields
                    cursor.execute("""
                        INSERT INTO change_log (
                            cin, change_type, company_name, company_roc_code, company_category, 
                            company_sub_category, company_class, authorized_capital, paidup_capital,
                            registration_date, registered_office_address, listing_status, company_status,
                            company_state_code, company_type, nic_code, company_industrial_classification,
                            row_hash, sync_flag
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        cin, change_type, 
                        "", "", "", "", "", "", "",  # empty company details
                        "", "", "", "", "", "", "", "", ""  # more empty fields
                    ))
                    logged_count += 1
                except Exception as e:
                    print(f"Failed to log change for {cin}: {e}")
                    continue
            
            conn.commit()
            print(f"Logged {logged_count} basic changes to change_log")
            return logged_count
            
        except Exception as e:
            conn.rollback()
            print(f"Error in bulk log changes: {e}")
            return 0
        finally:
            cursor.close()
            conn.close()

    def get_recent_changes_complete(self,limit=None):
        """Get recent changes with COMPLETE company details"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Check if table exists
            cursor.execute("SHOW TABLES LIKE 'change_log'")
            if not cursor.fetchone():
                print("change_log table doesn't exist yet")
                return pd.DataFrame()
            
            sql = f"""
                SELECT 
                    cin, company_name, company_roc_code, company_category, company_sub_category,
                    company_class, authorized_capital, paidup_capital, registration_date,
                    registered_office_address, listing_status, company_status, company_state_code,
                    company_type, nic_code, company_industrial_classification, row_hash, sync_flag,
                    change_type as Change_Type,
                    DATE_FORMAT(change_timestamp, '%Y-%m-%d %H:%i:%s') as Date
                FROM change_log 
                ORDER BY change_timestamp DESC 
            """
            
            cursor.execute(sql)
            result = cursor.fetchall()
            
            print(f"Retrieved {len(result)} complete changes from change_log")
            return pd.DataFrame(result) if result else pd.DataFrame()
            
        except Exception as e:
            print(f"get_recent_changes_complete failed: {e}")
            return pd.DataFrame()
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_archived_companies_full(self, cin_query: str = ""):
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            if cin_query:
                cursor.execute("""
                    SELECT 
                        cin, company_name, company_roc_code, company_category, company_sub_category,
                        company_class, authorized_capital, paidup_capital, registration_date,
                        registered_office_address, listing_status, company_status, company_state_code,
                        company_type, nic_code, company_industrial_classification, row_hash, sync_flag,
                        archived_at, archive_reason
                    FROM companies_archive 
                    WHERE cin LIKE %s
                    ORDER BY archived_at DESC
                """, (f"%{cin_query}%",))
            else:
                cursor.execute("""
                    SELECT 
                        cin, company_name, company_roc_code, company_category, company_sub_category,
                        company_class, authorized_capital, paidup_capital, registration_date,
                        registered_office_address, listing_status, company_status, company_state_code,
                        company_type, nic_code, company_industrial_classification, row_hash, sync_flag,
                        archived_at, archive_reason
                    FROM companies_archive 
                    ORDER BY archived_at DESC
                """)
                
            result = cursor.fetchall()
            return pd.DataFrame(result) if result else pd.DataFrame()
        except Exception as e:
            print(f"get_archived_companies_full failed: {e}")
            return pd.DataFrame()
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_companies_full(self, cin_query: str = ""):
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            if cin_query:
                cursor.execute("""
                    SELECT 
                        cin, company_name, company_roc_code, company_category, company_sub_category,
                        company_class, authorized_capital, paidup_capital, registration_date,
                        registered_office_address, listing_status, company_status, company_state_code,
                        company_type, nic_code, company_industrial_classification, row_hash, sync_flag
                    FROM companies 
                    WHERE cin LIKE %s
                    ORDER BY cin
                """, (f"%{cin_query}%",))
            else:
                cursor.execute("""
                    SELECT 
                        cin, company_name, company_roc_code, company_category, company_sub_category,
                        company_class, authorized_capital, paidup_capital, registration_date,
                        registered_office_address, listing_status, company_status, company_state_code,
                        company_type, nic_code, company_industrial_classification, row_hash, sync_flag
                    FROM companies 
                    ORDER BY cin
                """)
                
            result = cursor.fetchall()
            return pd.DataFrame(result) if result else pd.DataFrame()
        except Exception as e:
            print(f"get_companies_full failed: {e}")
            return pd.DataFrame()
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def get_companies_with_history(self, cin_query: str = ""):
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            if cin_query:
                cursor.execute("""
                    SELECT 
                        cin, company_name, company_status, company_state_code,
                        registration_date, listing_status, sync_flag
                    FROM companies 
                    WHERE cin LIKE %s
                    ORDER BY cin
                """, (f"%{cin_query}%",))
            else:
                cursor.execute("""
                    SELECT 
                        cin, company_name, company_status, company_state_code,
                        registration_date, listing_status, sync_flag
                    FROM companies 
                    ORDER BY cin
                """)
                
            result = cursor.fetchall()
            return pd.DataFrame(result) if result else pd.DataFrame()
        except Exception as e:
            print(f"get_companies_with_history failed: {e}")
            return pd.DataFrame()
        finally:
            if cursor: cursor.close()
            if conn: conn.close()

    def move_to_archive(self, companies_data: List[Dict], reason: str):
        """Move companies to archive table"""
        if not companies_data:
            return 0
            
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("START TRANSACTION")
            
            archived_count = 0
            for company in companies_data:
                try:
                    cursor.execute("""
                        INSERT INTO companies_archive 
                        (cin, company_name, company_roc_code, company_category, company_sub_category,
                        company_class, authorized_capital, paidup_capital, registration_date,
                        registered_office_address, listing_status, company_status, company_state_code,
                        company_type, nic_code, company_industrial_classification, row_hash, sync_flag, archive_reason)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        company.get('cin'), company.get('company_name'), company.get('company_roc_code'),
                        company.get('company_category'), company.get('company_sub_category'), company.get('company_class'),
                        company.get('authorized_capital'), company.get('paidup_capital'), company.get('registration_date'),
                        company.get('registered_office_address'), company.get('listing_status'), company.get('company_status'),
                        company.get('company_state_code'), company.get('company_type'), company.get('nic_code'),
                        company.get('company_industrial_classification'), company.get('row_hash'), company.get('sync_flag'), reason
                    ))
                    archived_count += 1
                except Exception as e:
                    print(f"Failed to archive {company.get('cin')}: {e}")
                    continue
            
            conn.commit()
            print(f"Successfully archived {archived_count} companies")
            return archived_count
            
        except Exception as e:
            conn.rollback()
            print(f"Archive process failed: {e}")
            return 0
        finally:
            cursor.close()
            conn.close()

    def delete_companies(self, cins: List[str]):
        """Delete companies from main table"""
        if not cins:
            return
            
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            placeholders = ','.join(['%s'] * len(cins))
            sql = f"DELETE FROM companies WHERE cin IN ({placeholders})"
            cursor.execute(sql, cins)
            conn.commit()
            print(f"Deleted {len(cins)} companies from main table")
        except Exception as e:
            print(f"Error deleting companies: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()

    def get_archived_companies(self, cin_query: str = ""):
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            
            if cin_query:
                cursor.execute("""
                    SELECT cin, company_name, company_status, archived_at, archive_reason
                    FROM companies_archive 
                    WHERE cin LIKE %s
                    ORDER BY archived_at DESC
                """, (f"%{cin_query}%",))
            else:
                cursor.execute("""
                    SELECT cin, company_name, company_status, archived_at, archive_reason
                    FROM companies_archive 
                    ORDER BY archived_at DESC
                """)
                
            result = cursor.fetchall()
            return pd.DataFrame(result) if result else pd.DataFrame()
        except Exception as e:
            print(f"get_archived_companies failed: {e}")
            return pd.DataFrame()
        finally:
            if cursor: cursor.close()
            if conn: conn.close()
            
    def log_company_change_complete(self, company_data: Dict, change_type: str):
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO change_log (
                    cin, company_name, company_roc_code, company_category, company_sub_category,
                    company_class, authorized_capital, paidup_capital, registration_date,
                    registered_office_address, listing_status, company_status, company_state_code,
                    company_type, nic_code, company_industrial_classification, row_hash, sync_flag, change_type
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                company_data.get('cin'), company_data.get('company_name'), company_data.get('company_roc_code'),
                company_data.get('company_category'), company_data.get('company_sub_category'), company_data.get('company_class'),
                company_data.get('authorized_capital'), company_data.get('paidup_capital'), company_data.get('registration_date'),
                company_data.get('registered_office_address'), company_data.get('listing_status'), company_data.get('company_status'),
                company_data.get('company_state_code'), company_data.get('company_type'), company_data.get('nic_code'),
                company_data.get('company_industrial_classification'), company_data.get('row_hash'), company_data.get('sync_flag'), change_type
            ))
            
            conn.commit()
            print(f"Logged {change_type} for {company_data.get('cin')}")
            
        except Exception as e:
            print(f"Error logging complete change: {e}")
            conn.rollback()
        finally:
            if cursor: cursor.close()
            if conn: conn.close()