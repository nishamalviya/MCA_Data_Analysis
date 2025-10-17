# main_streamlit.py — Streamlit version of Company Data Tracker

import os
import hashlib
import time
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Any
from pathlib import Path

import pandas as pd
import numpy as np
from dateutil import parser
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_chat import message as st_message

import matplotlib
matplotlib.use('Agg')

from db_config import DatabaseManager
from zauba_scraper import enrich_added

MASTER_COLUMNS = [
    'CIN','CompanyName','CompanyROCcode','CompanyCategory','CompanySubCategory','CompanyClass',
    'AuthorizedCapital','PaidupCapital','CompanyRegistrationdate_date','Registered_Office_Address',
    'Listingstatus','CompanyStatus','CompanyStateCode','CompanyIndian_Foreign_Company',
    'nic_code','CompanyIndustrialClassification','row_hash','sync_flag'
]

import mysql.connector
import subprocess

class CompanyChatbot:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        
    def _get_db_connection(self):
        return mysql.connector.connect(**self.db_config)
    
    def get_database_stats(self) -> Dict[str, Any]:
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            stats = {}
            
            cursor.execute("SELECT COUNT(*) FROM companies")
            stats['total_companies'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM change_log")
            stats['total_changes'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM companies_archive")
            stats['total_archived'] = cursor.fetchone()[0]
           
            cursor.execute("""
                SELECT change_type, COUNT(*) 
                FROM change_log 
                WHERE change_timestamp >= DATE_SUB(NOW(), INTERVAL 7 DAY)
                GROUP BY change_type
            """)
            recent_changes = cursor.fetchall()
            stats['recent_changes'] = dict(recent_changes) if recent_changes else {}
            
            cursor.execute("""
                SELECT company_state_code, COUNT(*) 
                FROM companies 
                GROUP BY company_state_code 
                ORDER BY COUNT(*) DESC 
                LIMIT 10
            """)
            top_states = cursor.fetchall()
            stats['top_states'] = dict(top_states) if top_states else {}
            
            cursor.execute("""
                SELECT company_status, COUNT(*) 
                FROM companies 
                GROUP BY company_status 
                ORDER BY COUNT(*) DESC
            """)
            status_dist = cursor.fetchall()
            stats['status_distribution'] = dict(status_dist) if status_dist else {}
            
            cursor.execute("""
                SELECT company_category, COUNT(*) 
                FROM companies 
                GROUP BY company_category 
                ORDER BY COUNT(*) DESC 
                LIMIT 10
            """)
            top_categories = cursor.fetchall()
            stats['top_categories'] = dict(top_categories) if top_categories else {}
            
            cursor.close()
            conn.close()
            
            return stats
            
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}
    
    def execute_sql_query(self, query: str) -> Tuple[List[Dict], str]:
        try:
            # Basic security check - prevent destructive operations
            destructive_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE']
            if any(keyword in query.upper() for keyword in destructive_keywords):
                return [], "Sorry, I can only execute SELECT queries for data analysis."
            
            conn = self._get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return results, "Success"
            
        except Exception as e:
            return [], f"Query execution error: {str(e)}"
    
    def get_table_schema(self) -> str:
        schema_info = []
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            tables = ['companies', 'change_log', 'companies_archive']
            for table in tables:
                cursor.execute(f"DESCRIBE {table}")
                columns = cursor.fetchall()
                schema_info.append(f"Table: {table}")
                for col in columns:
                    schema_info.append(f"  - {col[0]} ({col[1]})")
                schema_info.append("")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            schema_info.append(f"Error getting schema: {e}")
        
        return "\n".join(schema_info)
    
    def query_ollama(self, prompt: str, context: str = "") -> str:
        try:
            full_prompt = f"""
            You are a helpful assistant for a company database tracking system. 
            The database contains information about Indian companies including their CIN, name, status, capital, registration details, etc.
            
            CONTEXT INFORMATION:
            {context}
            
            DATABASE SCHEMA:
            {self.get_table_schema()}
            
            USER QUESTION: {prompt}
            
            Please provide a helpful, accurate response based on the available database structure and context. 
            If you need specific data, you can suggest SQL queries that could answer the question.
            Be concise but informative.
            """
           
            result = subprocess.run(
                ['ollama', 'run', 'llama3.2', full_prompt],
                capture_output=True,
                text=True,
                timeout=120  
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Error querying Ollama: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "Sorry, the query took too long to process. Please try a simpler question."
        except Exception as e:
            return f"Error communicating with Ollama: {str(e)}"
    
    def generate_sql_from_question(self, question: str) -> str:
        sql_suggestions = {
            "how many companies": "SELECT COUNT(*) as total_companies FROM companies",
            "list new companies": "SELECT cin, company_name, registration_date FROM companies WHERE sync_flag = 'NEW' ORDER BY registration_date DESC LIMIT 10",
            "recent changes": "SELECT cin, company_name, change_type, change_timestamp FROM change_log ORDER BY change_timestamp DESC LIMIT 10",
            "companies by state": "SELECT company_state_code, COUNT(*) as count FROM companies GROUP BY company_state_code ORDER BY count DESC",
            "active companies": "SELECT COUNT(*) as active_count FROM companies WHERE company_status = 'Active'",
            "listed companies": "SELECT COUNT(*) as listed_count FROM companies WHERE listing_status = 'Listed'",
            "top categories": "SELECT company_category, COUNT(*) as count FROM companies GROUP BY company_category ORDER BY count DESC LIMIT 10",
            "capital analysis": "SELECT AVG(CAST(REPLACE(REPLACE(paidup_capital, ',', ''), '₹', '') AS UNSIGNED)) as avg_capital FROM companies WHERE paidup_capital != ''",
            "registration trend": "SELECT YEAR(registration_date) as year, COUNT(*) as count FROM companies WHERE registration_date IS NOT NULL GROUP BY YEAR(registration_date) ORDER BY year"
        }
        
        question_lower = question.lower()
        for key, query in sql_suggestions.items():
            if key in question_lower:
                return query
                
        return "SELECT * FROM companies LIMIT 5"  
    
    def process_question(self, question: str) -> str:
        db_stats = self.get_database_stats()
        context = f"""
        DATABASE STATISTICS:
        - Total Companies: {db_stats.get('total_companies', 0):,}
        - Total Changes Logged: {db_stats.get('total_changes', 0):,}
        - Archived Companies: {db_stats.get('total_archived', 0):,}
        - Recent Changes (7 days): {db_stats.get('recent_changes', {})}
        - Top States: {db_stats.get('top_states', {})}
        - Status Distribution: {db_stats.get('status_distribution', {})}
        - Top Categories: {db_stats.get('top_categories', {})}
        """
        
        data_related_keywords = ['how many', 'list', 'show', 'count', 'find', 'search', 'what is', 'which', 'when', 'where']
        if any(keyword in question.lower() for keyword in data_related_keywords):
            sql_query = self.generate_sql_from_question(question)
            results, error = self.execute_sql_query(sql_query)
            
            if error == "Success" and results:
                data_context = f"Query Results (showing first 5 rows): {json.dumps(results[:5], default=str)}"
                context += f"\n{data_context}"
        
        response = self.query_ollama(question, context)
        
        return response

def robust_read_csv(file_input, nrows=None):
    strategies = [
        lambda f: pd.read_csv(f, dtype=str, nrows=nrows, 
                          encoding='utf-8', low_memory=False),
        
        lambda f: pd.read_csv(f, dtype=str, nrows=nrows, 
                          engine='python', encoding='utf-8'),
        
        lambda f: pd.read_csv(f, dtype=str, nrows=nrows,
                          on_bad_lines='skip', encoding='utf-8'),
        
        lambda f: pd.read_csv(f, dtype=str, nrows=nrows,
                          encoding='latin-1', low_memory=False),
        
        lambda f: manual_csv_parse(f, nrows)
    ]
    
    for i, strategy in enumerate(strategies):
        try:
            if hasattr(file_input, 'seek'):
                file_input.seek(0) 
                result = strategy(file_input)
            else:
                with open(file_input, 'r', encoding='utf-8') as f:
                    result = strategy(f)
            
            print(f"CSV reading successful with strategy {i+1}")
            return result
        except Exception as e:
            print(f"Strategy {i+1} failed: {e}")
            if i == len(strategies) - 1: 
                raise e
            continue

def manual_csv_parse(file_input, nrows=None):
    if hasattr(file_input, 'read'):
        file_input.seek(0)
        try:
            content = file_input.read().decode('utf-8', errors='ignore')
        except:
            file_input.seek(0)
            content = file_input.read().decode('latin-1', errors='ignore')
    else:
        try:
            with open(file_input, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            with open(file_input, 'r', encoding='latin-1') as f:
                content = f.read()
    
    lines = content.split('\n')
    if not lines:
        return pd.DataFrame()
    
    header = lines[0].split(',')
    header = [col.strip().replace('"', '') for col in header]
    
    data = []
    for line_num, line in enumerate(lines[1:], 1):
        if nrows and len(data) >= nrows:
            break
            
        if line.strip():
            try:
                row = []
                field = ""
                in_quotes = False
                
                for char in line:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        row.append(field.strip().replace('"', ''))
                        field = ""
                    else:
                        field += char
               
                row.append(field.strip().replace('"', ''))
                
                if len(row) < len(header):
                    row.extend([''] * (len(header) - len(row)))
                elif len(row) > len(header):
                    row = row[:len(header)]
                
                data.append(row)
                
            except Exception as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
    
    return pd.DataFrame(data, columns=header)

def safe_read_preview(uploaded_file, nrows=3):
    try:
        return robust_read_csv(uploaded_file, nrows=nrows)
    except Exception as e:
        st.error(f"Could not read {uploaded_file.name}: {str(e)}")
        return pd.DataFrame({"Error": [f"Could not read file: {str(e)}"]})

class CompanyTracker:
    def __init__(self):
        self.db_manager = DatabaseManager(
            host="localhost",
            user="root", 
            password="", 
            database="companies",
            port=3306
        )
        self.db_manager._ensure_initialized()
        self.current_changes: List[Dict[str, str]] = []  
        self.added_pairs: List[Tuple[str, str]] = []     
        print("CompanyTracker initialized successfully with MySQL!")

    def _get_permanent_counts(self):
        try:
            
            changes_df = self.db_manager.get_recent_changes_complete()
            
            if changes_df.empty:
                return {'added': 0, 'removed': 0, 'total': 0}
            
            added = len(changes_df[changes_df['Change_Type'] == 'NEW'])
            removed = len(changes_df[changes_df['Change_Type'] == 'REMOVED'])
            total = len(changes_df)
            
            return {
                'added': added,
                'removed': removed, 
                'total': total
            }
        except Exception as e:
            print(f"Error in _get_permanent_counts: {e}")
            return {'added': 0, 'removed': 0, 'total': 0}

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=lambda x: str(x).strip().replace('/', '_').replace(' ', '_'))
        if 'CIN' not in df.columns:
            raise ValueError("CIN column not found in the uploaded file")
        df['CIN'] = df['CIN'].astype(str).str.strip().str.upper()

        text_columns = [
            'CompanyName', 'CompanyROCcode', 'CompanyCategory',
            'CompanySubCategory', 'CompanyClass', 'CompanyStatus',
            'CompanyStateCode', 'CompanyIndian_Foreign_Company',
            'nic_code', 'CompanyIndustrialClassification',
            'Registered_Office_Address', 'Listingstatus'
        ]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        for col in ['AuthorizedCapital', 'PaidupCapital']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False).str.strip()

        def parse_date_safe(x):
            try:
                return parser.parse(str(x)).date().isoformat()
            except Exception:
                return ""
        if 'CompanyRegistrationdate_date' in df.columns:
            df['CompanyRegistrationdate_date'] = df['CompanyRegistrationdate_date'].apply(parse_date_safe)

        def _row_sig(r):
            return f"{r.get('CIN','')}-{r.get('CompanyName','')}-{r.get('PaidupCapital','')}-{r.get('CompanyStatus','')}-{r.get('AuthorizedCapital','')}"
        df['row_hash'] = df.apply(lambda r: hashlib.md5(_row_sig(r).encode()).hexdigest(), axis=1)

        return df.drop_duplicates(subset='CIN', keep='first')

    def combine_multiple_files(self, uploaded_files: List) -> str:
        if not uploaded_files:
            raise ValueError("No files uploaded")
        
        print(f"Combining {len(uploaded_files)} files...")
        
        combined_df = pd.DataFrame()
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                df = robust_read_csv(uploaded_file)
                print(f"File {i+1}: {uploaded_file.name} - {len(df)} rows")
                df_clean = self.clean_dataframe(df)
                
                if combined_df.empty:
                    combined_df = df_clean
                else:
                    combined_df = pd.concat([combined_df, df_clean], ignore_index=True)
                        
            except Exception as e:
                print(f"Error processing {uploaded_file.name}: {e}")
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue  
        
        if combined_df.empty:
            raise ValueError("No valid data could be read from any uploaded files")
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset='CIN', keep='first')
        after_dedup = len(combined_df)
        
        print(f"Deduplication: {before_dedup} -> {after_dedup} rows")
        
        output_path = "combined_master.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"💾 Combined master file saved: {output_path} with {len(combined_df)} unique companies")
        
        return output_path

    def get_all_companies_df(self, cin_query: str = "") -> pd.DataFrame:
        all_rows = self.db_manager.get_all_companies()
        if not all_rows:
            return pd.DataFrame(columns=[
                "cin","company_name","company_status","company_state_code",
                "registration_date","listing_status","sync_flag"
            ])
        df = pd.DataFrame(all_rows)
        wanted = [
            "cin","company_name","company_status","company_state_code",
            "registration_date","listing_status","sync_flag"
        ]
        for col in wanted:
            if col not in df.columns:
                df[col] = ""
        df = df[wanted].fillna("")
        if cin_query:
            df = df[df["cin"].str.contains(str(cin_query), case=False, na=False)]
        return df.sort_values("cin").reset_index(drop=True)

    def get_companies_full_df(self) -> pd.DataFrame:
        return self.db_manager.get_companies_full()

    def process_multiple_files(self, uploaded_files: List):
        print(f"Processing {len(uploaded_files)} files...")
        
        if not uploaded_files:
            return "Please upload at least one CSV file first", pd.DataFrame(), pd.DataFrame(), False, pd.DataFrame()

        try:
            start_time = time.time()
            
            combined_file_path = self.combine_multiple_files(uploaded_files)
            
            new_df = robust_read_csv(combined_file_path).fillna("")
            print(f"Combined CSV read: {len(new_df):,} unique rows")
            
            current_companies = self.db_manager.get_all_companies()
            current_df = pd.DataFrame(current_companies) if current_companies else pd.DataFrame()
            print(f"DB data fetched: {len(current_df):,} rows")

            current_cins = set(current_df['cin']) if not current_df.empty else set()
            new_cins = set(new_df['CIN'])
            
            added = new_cins - current_cins
            removed = current_cins - new_cins

            print(f"Added: {len(added):,}, Removed: {len(removed):,}")

            if removed:
                print(f"Moving {len(removed):,} companies to archive...")
                
                removed_companies_data = []
                for cin in list(removed):
                    company_row = current_df[current_df['cin'] == cin]
                    if not company_row.empty:
                        company_data = company_row.iloc[0].to_dict()
                        removed_companies_data.append(company_data)
                     
                        self.db_manager.log_company_change_complete(company_data, 'REMOVED')
                
                if removed_companies_data:
                    archived_count = self.db_manager.move_to_archive(removed_companies_data, "Removed from latest dataset")
                    print(f"Archived {archived_count} companies")
                    self.db_manager.delete_companies(list(removed))

            self.added_pairs = []
            for cin in list(added):
                company_row = new_df[new_df['CIN'] == cin]
                if not company_row.empty:
                    company_data = {
                        'cin': cin,
                        'company_name': company_row['CompanyName'].iloc[0] if 'CompanyName' in company_row.columns else '',
                        'company_roc_code': company_row['CompanyROCcode'].iloc[0] if 'CompanyROCcode' in company_row.columns else '',
                        'company_category': company_row['CompanyCategory'].iloc[0] if 'CompanyCategory' in company_row.columns else '',
                        'company_sub_category': company_row['CompanySubCategory'].iloc[0] if 'CompanySubCategory' in company_row.columns else '',
                        'company_class': company_row['CompanyClass'].iloc[0] if 'CompanyClass' in company_row.columns else '',
                        'authorized_capital': company_row['AuthorizedCapital'].iloc[0] if 'AuthorizedCapital' in company_row.columns else '',
                        'paidup_capital': company_row['PaidupCapital'].iloc[0] if 'PaidupCapital' in company_row.columns else '',
                        'registration_date': company_row['CompanyRegistrationdate_date'].iloc[0] if 'CompanyRegistrationdate_date' in company_row.columns else '',
                        'registered_office_address': company_row['Registered_Office_Address'].iloc[0] if 'Registered_Office_Address' in company_row.columns else '',
                        'listing_status': company_row['Listingstatus'].iloc[0] if 'Listingstatus' in company_row.columns else '',
                        'company_status': company_row['CompanyStatus'].iloc[0] if 'CompanyStatus' in company_row.columns else '',
                        'company_state_code': company_row['CompanyStateCode'].iloc[0] if 'CompanyStateCode' in company_row.columns else '',
                        'company_type': company_row['CompanyIndian_Foreign_Company'].iloc[0] if 'CompanyIndian_Foreign_Company' in company_row.columns else '',
                        'nic_code': company_row['nic_code'].iloc[0] if 'nic_code' in company_row.columns else '',
                        'company_industrial_classification': company_row['CompanyIndustrialClassification'].iloc[0] if 'CompanyIndustrialClassification' in company_row.columns else '',
                        'row_hash': company_row['row_hash'].iloc[0] if 'row_hash' in company_row.columns else '',
                        'sync_flag': 'NEW'
                    }
                    
                    self.added_pairs.append((cin, company_data['company_name']))
                    
                    self.db_manager.log_company_change_complete(company_data, 'NEW')

            master_df = new_df.copy()

            for col in MASTER_COLUMNS:
                if col not in master_df.columns:
                    master_df[col] = "" if col != "sync_flag" else "UNCHANGED"

            master_df['sync_flag'] = 'UNCHANGED'
            if added:
                master_df.loc[master_df['CIN'].isin(added), 'sync_flag'] = 'NEW'

            master_for_db = master_df.rename(columns={
                'CIN': 'cin',
                'CompanyName': 'company_name',
                'CompanyROCcode': 'company_roc_code',
                'CompanyCategory': 'company_category',
                'CompanySubCategory': 'company_sub_category', 
                'CompanyClass': 'company_class',
                'AuthorizedCapital': 'authorized_capital',
                'PaidupCapital': 'paidup_capital',
                'CompanyRegistrationdate_date': 'registration_date',
                'Registered_Office_Address': 'registered_office_address',
                'Listingstatus': 'listing_status',
                'CompanyStatus': 'company_status',
                'CompanyStateCode': 'company_state_code',
                'CompanyIndian_Foreign_Company': 'company_type',
                'nic_code': 'nic_code',
                'CompanyIndustrialClassification': 'company_industrial_classification',
                'row_hash': 'row_hash',
                'sync_flag': 'sync_flag'
            })
         
            required_cols = [
                'cin', 'company_name', 'company_roc_code', 'company_category', 'company_sub_category',
                'company_class', 'authorized_capital', 'paidup_capital', 'registration_date',
                'registered_office_address', 'listing_status', 'company_status', 'company_state_code',
                'company_type', 'nic_code', 'company_industrial_classification', 'row_hash', 'sync_flag'
            ]
            
            for col in required_cols:
                if col not in master_for_db.columns:
                    master_for_db[col] = ""
      
            affected = self.db_manager.replace_companies_from_dataframe(master_for_db[required_cols])
            print(f"Main table updated: {affected:,} records")

            complete_changes_df = self.db_manager.get_recent_changes_complete()
            print(f"Retrieved {len(complete_changes_df)} complete changes from change_log")

            end_time = time.time()
            processing_time = round(end_time - start_time, 2)

            permanent_counts = self._get_permanent_counts()
    
            summary = (
                f"**Sync Complete in {processing_time}s**\n\n"
                f"**Added:** {permanent_counts['added']:,}  \n"
                f"**Removed:** {permanent_counts['removed']:,}  \n"
                f"**Current Companies:** {affected:,}  \n"
                f"**Total Changes:** {permanent_counts['total']:,}  \n"
                f"**Files Processed:** {len(uploaded_files)}  \n"
                f"*All changes permanently stored in database*"
            )

            return summary, complete_changes_df, self.get_summary_dataframe(), len(added) > 0, self.get_companies_full_df()

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"Processing error: {e}")
            import traceback
            traceback.print_exc()
            empty_df = pd.DataFrame()
            return error_msg, empty_df, empty_df, False, empty_df

    def get_summary_dataframe(self) -> pd.DataFrame:
        try:
            # Use the NEW method that works with complete schema
            changes_df = self.db_manager.get_recent_changes_complete()
            
            if changes_df.empty:
                return pd.DataFrame({
                    'Metric': ['New Companies', 'Removed Companies', 'Total Changes'],
                    'Count': [0, 0, 0]
                })

            new_added_count = len(changes_df[changes_df['Change_Type'] == 'NEW'])
            removed_count = len(changes_df[changes_df['Change_Type'] == 'REMOVED'])
            total_count = len(changes_df)
            
            print(f"Summary - New: {new_added_count}, Removed: {removed_count}, Total: {total_count}")
            
            return pd.DataFrame({
                'Metric': ['New Companies', 'Removed Companies', 'Total Changes'],
                'Count': [new_added_count, removed_count, total_count]
            })
        except Exception as e:
            print(f"Error in get_summary_dataframe: {e}")
            return pd.DataFrame({
                'Metric': ['New Companies', 'Removed Companies', 'Total Changes'],
                'Count': [0, 0, 0]
            })

    def filter_changes_or_all(self, filter_type: str, cin_query: str) -> pd.DataFrame:
        try:
            if filter_type == "All":
                # Current active companies with ALL columns
                return self.db_manager.get_companies_full(cin_query)
                
            elif filter_type == "New added":
                # NEW companies from change_log with COMPLETE details
                changes_df = self.db_manager.get_recent_changes_complete()
                if not changes_df.empty:
                    changes_df = changes_df[changes_df['Change_Type'] == 'NEW']
                    if cin_query and not changes_df.empty:
                        mask = changes_df["cin"].astype(str).str.contains(str(cin_query), case=False, na=False)
                        changes_df = changes_df[mask]
                return changes_df
                
            elif filter_type == "Removed/Deregistered":
                # REMOVED companies from change_log with COMPLETE details
                changes_df = self.db_manager.get_recent_changes_complete()
                if not changes_df.empty:
                    changes_df = changes_df[changes_df['Change_Type'] == 'REMOVED']
                    if cin_query and not changes_df.empty:
                        mask = changes_df["cin"].astype(str).str.contains(str(cin_query), case=False, na=False)
                        changes_df = changes_df[mask]
                return changes_df
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error in filter_changes_or_all: {e}")
            return pd.DataFrame()

    def get_added_count(self) -> int:
        return len([1 for r in self.current_changes if r.get("Change_Type") == "New Incorporation"])

    def get_added_pairs(self) -> List[Tuple[str, str]]:
        return self.added_pairs[:] if self.added_pairs else [
            (r.get("CIN",""), r.get("New_Value",""))
            for r in self.current_changes
            if r.get("Change_Type") == "New Incorporation"
        ]

def plot_registration_trend(df: pd.DataFrame):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6))
    if df.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Registration Trend", fontsize=14, fontweight='bold', pad=20)
        return fig
        
    years = pd.to_datetime(df['registration_date'], errors='coerce').dt.year.dropna().astype(int)
    if years.empty:
        ax.text(0.5, 0.5, 'No registration dates available', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("Registration Trend", fontsize=14, fontweight='bold', pad=20)
        return fig
        
    counts = years.value_counts().sort_index()
    

    ax.plot(counts.index, counts.values, marker='o', linewidth=3, markersize=8, 
            color='#3b82f6', markerfacecolor='white', markeredgewidth=2)
    ax.fill_between(counts.index, counts.values, alpha=0.2, color='#3b82f6')
    
    ax.set_title("Company Registrations Trend", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Year", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Companies", fontsize=12, labelpad=10)

    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
 
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#e5e7eb')
        spine.set_linewidth(1)
    
    plt.tight_layout()
    return fig

def plot_top_states(df: pd.DataFrame, top_n=15):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 7))
    if df.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Top States", fontsize=14, fontweight='bold', pad=20)
        return fig
        
    s = df['company_state_code'].fillna('Unknown').value_counts().head(top_n)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(s)))
    
    bars = ax.bar(range(len(s)), s.values, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax.set_title(f"Top {top_n} States by Company Count", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("State Code", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Companies", fontsize=12, labelpad=10)

    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s.index, rotation=45, ha='right', fontsize=10)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(s.values)*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
  
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

def plot_status_split(df: pd.DataFrame):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 8))
    if df.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Status Distribution", fontsize=14, fontweight='bold', pad=20)
        return fig
        
    s = df['company_status'].fillna('Unknown').value_counts()
 
    colors = plt.cm.Set3(np.linspace(0, 1, len(s)))
 
    wedges, texts, autotexts = ax.pie(
        s.values, 
        labels=s.index, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors,
        textprops={'fontsize': 10, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        labeldistance=1.1
    )
    

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    ax.set_title("Company Status Distribution", fontsize=14, fontweight='bold', pad=30)
   
    ax.legend(wedges, s.index, title="Status", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    return fig

def plot_category_class(df: pd.DataFrame):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(14, 8))
    if df.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Category vs Class", fontsize=14, fontweight='bold', pad=20)
        return fig
        
    pivot = (df.assign(company_category=df['company_category'].fillna('Unknown'),
                       company_class=df['company_class'].fillna('Unknown'))
               .groupby(['company_category','company_class'])
               .size().unstack(fill_value=0))
  
    colors = plt.cm.tab20(np.linspace(0, 1, len(pivot.columns)))
    bars = pivot.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_title("Company Category vs Class Distribution", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Company Category", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Companies", fontsize=12, labelpad=10)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
   
    plt.legend(title='Company Class', bbox_to_anchor=(1.05, 1), loc='upper left', 
               frameon=True, fancybox=True, shadow=True)
    
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

def plot_listing_status(df: pd.DataFrame):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6))
    if df.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Listing Status", fontsize=14, fontweight='bold', pad=20)
        return fig
        
    listing_data = df['listing_status'].fillna('Unknown').value_counts()

    color_map = {
        'Listed': '#10b981', 
        'Unlisted': '#ef4444', 
        'Unknown': '#6b7280',
        'Yes': '#10b981',
        'No': '#ef4444'
    }
    
    colors = [color_map.get(x, '#6b7280') for x in listing_data.index]
    
    bars = ax.bar(listing_data.index, listing_data.values, color=colors, alpha=0.8, 
                  edgecolor='white', linewidth=2)
    
    ax.set_title("Listing Status Distribution", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Listing Status", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Companies", fontsize=12, labelpad=10)
   
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(listing_data.values)*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=11)
  
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

def plot_company_type_distribution(df: pd.DataFrame):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6))
    if df.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Company Types", fontsize=14, fontweight='bold', pad=20)
        return fig
        
    type_data = df['company_type'].fillna('Unknown').value_counts().head(10)
  
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(type_data)))
    
    bars = ax.barh(range(len(type_data)), type_data.values, color=colors, alpha=0.8, 
                   edgecolor='white', linewidth=1.5)
    
    ax.set_title("Top 10 Company Types", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Number of Companies", fontsize=12, labelpad=10)
    ax.set_ylabel("Company Type", fontsize=12, labelpad=10)
   
    ax.set_yticks(range(len(type_data)))
    ax.set_yticklabels(type_data.index, fontsize=10)
  
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + max(type_data.values)*0.01, bar.get_y() + bar.get_height()/2.,
                f'{int(width):,}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.grid(True, alpha=0.2, axis='x', linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

def plot_capital_distribution(df: pd.DataFrame):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 6))
    if df.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Capital Distribution", fontsize=14, fontweight='bold', pad=20)
        return fig
   
    df_clean = df.copy()
    df_clean['paidup_capital_num'] = pd.to_numeric(df_clean['paidup_capital'].str.replace(',', '').str.replace('₹', ''), errors='coerce')
    df_clean = df_clean.dropna(subset=['paidup_capital_num'])
    
    if df_clean.empty:
        ax.text(0.5, 0.5, 'No valid capital data available', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("Capital Distribution", fontsize=14, fontweight='bold', pad=20)
        return fig
   
    bins = [0, 100000, 1000000, 10000000, 100000000, float('inf')]
    labels = ['< ₹1L', '₹1L-10L', '₹10L-1Cr', '₹1Cr-10Cr', '> ₹10Cr']
    df_clean['capital_range'] = pd.cut(df_clean['paidup_capital_num'], bins=bins, labels=labels)
    
    capital_data = df_clean['capital_range'].value_counts().sort_index()
   
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(capital_data)))
    
    bars = ax.bar(capital_data.index, capital_data.values, color=colors, alpha=0.8, 
                  edgecolor='white', linewidth=2)
    
    ax.set_title("Paid-up Capital Distribution", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Capital Range", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Companies", fontsize=12, labelpad=10)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(capital_data.values)*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    return fig

def plot_monthly_registrations(df: pd.DataFrame):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 6))
    if df.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Monthly Registrations", fontsize=14, fontweight='bold', pad=20)
        return fig
        
    df_clean = df.copy()
    df_clean['registration_month'] = pd.to_datetime(df_clean['registration_date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['registration_month'])
    df_clean['year_month'] = df_clean['registration_month'].dt.to_period('M')
    
    monthly_data = df_clean['year_month'].value_counts().sort_index().tail(24)  
    
    if monthly_data.empty:
        ax.text(0.5, 0.5, 'No valid registration date data', ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("Monthly Registrations", fontsize=14, fontweight='bold', pad=20)
        return fig
 
    ax.fill_between(range(len(monthly_data)), monthly_data.values, alpha=0.3, color='#f59e0b')
    ax.plot(range(len(monthly_data)), monthly_data.values, marker='o', linewidth=3, 
            color='#f59e0b', markersize=6, markerfacecolor='white', markeredgewidth=2)
    
    ax.set_title("Monthly Company Registrations (Last 24 Months)", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Month", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Companies", fontsize=12, labelpad=10)

    tick_positions = range(0, len(monthly_data), max(1, len(monthly_data)//8))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(monthly_data.index[i]) for i in tick_positions], rotation=45, ha='right', fontsize=9)
  
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

def plot_roc_distribution(df: pd.DataFrame):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 6))
    if df.empty:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("ROC Distribution", fontsize=14, fontweight='bold', pad=20)
        return fig
        
    roc_data = df['company_roc_code'].fillna('Unknown').value_counts().head(15)
  
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(roc_data)))
    
    bars = ax.bar(roc_data.index, roc_data.values, color=colors, alpha=0.8, 
                  edgecolor='white', linewidth=1.5)
    
    ax.set_title("Top 15 ROC (Registrar of Companies) Distribution", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("ROC Code", fontsize=12, labelpad=10)
    ax.set_ylabel("Number of Companies", fontsize=12, labelpad=10)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(roc_data.values)*0.01,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
   
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

def _coerce_year_range(val: Any) -> Tuple[int, int]:
    default_rng = (1990, datetime.now().year)
    try:
        if isinstance(val, (list, tuple)) and len(val) == 2:
            a = int(val[0]); b = int(val[1]); return (a, b) if a <= b else (b, a)
        if isinstance(val, str) and '-' in val:
            a, b = val.split('-', 1); a = int(a.strip()); b = int(b.strip()); return (a, b) if a <= b else (b, a)
        if isinstance(val, (int, float)):
            y = int(val); return (y, y)
    except Exception:
        pass
    return default_rng

def apply_filters_for_analysis(df: pd.DataFrame,
                               status: List[str],
                               listing: List[str],
                               category: List[str],
                               clazz: List[str],
                               states: List[str],
                               year_range: Any) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if status:   out = out[out['company_status'].fillna('').isin(status)]
    if listing:  out = out[out['listing_status'].fillna('').isin(listing)]
    if category: out = out[out['company_category'].fillna('').isin(category)]
    if clazz:    out = out[out['company_class'].fillna('').isin(clazz)]
    if states:   out = out[out['company_state_code'].fillna('').isin(states)]

    y0, y1 = _coerce_year_range(year_range)
    years = pd.to_datetime(out['registration_date'], errors='coerce').dt.year
    mask = (years >= y0) & (years <= y1)
    out = out[mask.fillna(False)]
    return out

def compute_score_cards(df: pd.DataFrame) -> Dict[str, any]:
    total = len(df)
    active = (df['company_status'].fillna('').str.lower() == 'active').sum() if not df.empty else 0
    listed = (df['listing_status'].fillna('').str.lower().isin(['listed','yes','true','active'])).sum() if not df.empty else 0
    states = df['company_state_code'].nunique() if not df.empty else 0
    categories = df['company_category'].nunique() if not df.empty else 0
    
    try:
        capital_series = pd.to_numeric(df['paidup_capital'].str.replace(',', '').str.replace('₹', ''), errors='coerce')
        avg_capital = capital_series.mean()
        if pd.isna(avg_capital):
            avg_capital = 0
    except:
        avg_capital = 0
    
    return {
        "Total Companies": total, 
        "Active": int(active), 
        "Listed": int(listed), 
        "States": int(states),
        "Categories": int(categories),
        "Avg Capital": f"₹{int(avg_capital):,}"
    }

def main():
    st.set_page_config(
        page_title="Company Data Tracker",
        page_icon="🏢",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    if 'tracker' not in st.session_state:
        st.session_state.tracker = CompanyTracker()
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = CompanyChatbot({
            'host': "localhost",
            'user': "root", 
            'password': "",
            'database': "companies",
            'port': 3306
        })
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #ddd;
    }
    .score-title {
        font-size: 0.9rem;
        color: #666;
        margin: 0;
    }
    .score-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">Companies Data</h1>', unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a page", 
                                   ["Data Management", "Advanced Analysis", "AI Chat Assistant"])

    if app_mode == "Data Management":
        render_data_management()

    elif app_mode == "Advanced Analysis":
        render_advanced_analysis()
 
    elif app_mode == "AI Chat Assistant":
        render_chat_assistant()

def render_data_management():
 
    st.subheader("Upload Multiple CSV Files")
    uploaded_files = st.file_uploader(
        "Choose CSV files (minimum 3)", 
        type="csv", 
        accept_multiple_files=True,
        help="Upload multiple CSV files. They will be combined and processed as a single batch."
    )
    
    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected: {[f.name for f in uploaded_files]}")
        
        with st.expander("File Previews"):
            for i, uploaded_file in enumerate(uploaded_files):
                st.write(f"**File {i+1}: {uploaded_file.name}**")
                try:
                    preview_df = safe_read_preview(uploaded_file, nrows=3)
                    if not preview_df.empty and 'Error' not in preview_df.columns:
                        st.dataframe(preview_df, use_container_width=True)
                        st.write(f"Preview loaded successfully - {len(preview_df)} rows")
                    else:
                        st.error(f"Could not preview {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error previewing {uploaded_file.name}: {str(e)}")
    
    if uploaded_files and len(uploaded_files) >= 3:
        if st.button("🚀 Process All Files & Update Database", type="primary"):
            with st.spinner(f"Processing {len(uploaded_files)} CSV files together..."):
                status, changes, stats, show_enrich, data_df = st.session_state.tracker.process_multiple_files(uploaded_files)
                
                st.success(status)
               
                if not stats.empty and any(stats['Count'] > 0):
                    st.subheader("Summary")
                    st.dataframe(stats, use_container_width=True)
                
                if not changes.empty:
                    st.subheader("Changes History")
                    changes_display = changes.copy()
                    date_columns = ['registration_date', 'change_timestamp']
                    for col in date_columns:
                        if col in changes_display.columns:
                            changes_display[col] = changes_display[col].astype(str)
                    st.dataframe(changes_display, use_container_width=True)
                    
                    if show_enrich:
                        st.info("🎯 New companies detected! You can now get more information using the button below.")
    elif uploaded_files and len(uploaded_files) < 3:
        st.warning("Please upload at least 3 CSV files to proceed.")
  
    st.subheader("Newly Added Companies")
  
    added_df = st.session_state.tracker.filter_changes_or_all("New added", "")
    
    if not added_df.empty:
        added_display = added_df.copy()
       
        for col in added_display.columns:
            added_display[col] = added_display[col].astype(str)
        
        # Display the added companies table
        st.dataframe(added_display, use_container_width=True)
       
        if st.button("🔍 Get More Information", type="primary"):
            with st.spinner("Scraping additional company details from ZaubaCorp..."):
                pairs = []
                for _, row in added_df.iterrows():
                    cin = str(row.get('cin', '')) if pd.notna(row.get('cin')) else ''
                    company_name = str(row.get('company_name', '')) if pd.notna(row.get('company_name')) else ''
                    if cin and company_name and cin.strip() and company_name.strip():
                        pairs.append((cin.strip(), company_name.strip()))
                
                st.info(f"Found {len(pairs)} companies for scraping...")
                
                if pairs:
                    try:
                        from zauba_scraper import enrich_added
                        
                        progress_text = st.empty()
                        progress_bar = st.progress(0)
                        
                        for i in range(len(pairs)):
                            progress_bar.progress((i + 1) / len(pairs))
                            progress_text.text(f"Scraping company {i + 1} of {len(pairs)}...")
                            time.sleep(0.1) 
                    
                        enriched_df = enrich_added(
                            pairs, 
                            output_excel_path="added_enriched.xlsx", 
                            headless=True,
                            min_delay=2.0,
                            max_delay=4.0
                        )
                        
                        progress_text.empty()
                        progress_bar.empty()
                        
                        st.success(f"Successfully enriched {len(enriched_df)} companies!")
                        
                        st.subheader("Detailed Company Information from ZaubaCorp")
                        
                        enriched_display = enriched_df.copy()
                        for col in enriched_display.columns:
                            enriched_display[col] = enriched_display[col].astype(str)
                        
                        st.dataframe(enriched_display, use_container_width=True)
                        
                        success_count = len(enriched_df[enriched_df['status'] == 'success'])
                        error_count = len(enriched_df[enriched_df['status'] == 'error'])
                        no_results_count = len(enriched_df[enriched_df['status'] == 'no_results'])
                        
                        st.metric("Scraping Results", 
                                 f"{success_count} Success / {error_count} Errors / {no_results_count} No Results")
            
                        try:
                            with open("added_enriched.xlsx", "rb") as file:
                                st.download_button(
                                    label="📥 Download Enriched Data (Excel)",
                                    data=file,
                                    file_name="added_enriched.xlsx",
                                    mime="application/vnd.ms-excel"
                                )
                        except Exception as e:
                            st.error(f"Download error: {e}")
                            
                    except Exception as e:
                        st.error(f"Scraping error: {e}")
                        st.info("Please check your internet connection and try again.")
                else:
                    st.warning("No valid company data available for enrichment. Make sure CIN and Company Name are present.")
    else:
        st.info("No newly added companies found.")
    
    st.subheader("Current Companies")
    cin_query_all = st.text_input("Search companies by CIN", placeholder="Type CIN...", key="cin_search_all")
    all_companies_df = st.session_state.tracker.get_all_companies_df(cin_query_all)
    
    if not all_companies_df.empty:
        all_companies_display = all_companies_df.copy()
        for col in all_companies_display.columns:
            all_companies_display[col] = all_companies_display[col].astype(str)
        
        st.dataframe(all_companies_display, use_container_width=True)
    else:
        st.info("No companies found.")

def render_advanced_analysis():
    
    if 'analysis_df' not in st.session_state:
        st.session_state.analysis_df = st.session_state.tracker.get_companies_full_df()
    
    st.sidebar.subheader("Analysis Filters")
    
    # Get unique values for filters
    status_options = sorted([x for x in st.session_state.analysis_df['company_status'].dropna().unique().tolist() if x != ""])
    listing_options = sorted([x for x in st.session_state.analysis_df['listing_status'].dropna().unique().tolist() if x != ""])
    category_options = sorted([x for x in st.session_state.analysis_df['company_category'].dropna().unique().tolist() if x != ""])
    class_options = sorted([x for x in st.session_state.analysis_df['company_class'].dropna().unique().tolist() if x != ""])
    state_options = sorted([x for x in st.session_state.analysis_df['company_state_code'].dropna().unique().tolist() if x != ""])
    
    years = pd.to_datetime(st.session_state.analysis_df['registration_date'], errors='coerce').dt.year.dropna().astype(int)
    y_min = int(years.min()) if not years.empty else 1990
    y_max = int(years.max()) if not years.empty else datetime.now().year
    
    selected_status = st.sidebar.multiselect("Company Status", status_options)
    selected_listing = st.sidebar.multiselect("Listing Status", listing_options)
    selected_category = st.sidebar.multiselect("Company Category", category_options)
    selected_class = st.sidebar.multiselect("Company Class", class_options)
    selected_states = st.sidebar.multiselect("State Code", state_options)
    year_range = st.sidebar.slider("Registration Year Range", y_min, y_max, (y_min, y_max))
    
    filtered_df = apply_filters_for_analysis(
        st.session_state.analysis_df,
        selected_status, selected_listing, selected_category, 
        selected_class, selected_states, year_range
    )
  
    st.subheader("Overview Metrics")
    cards = compute_score_cards(filtered_df)
    
    cols = st.columns(6)
    metric_keys = list(cards.keys())
    for i, col in enumerate(cols):
        if i < len(metric_keys):
            with col:
                st.markdown(f"""
                <div class="score-card">
                    <p class="score-title">{metric_keys[i]}</p>
                    <p class="score-value">{cards[metric_keys[i]]}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Charts
    st.subheader("Company Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(plot_registration_trend(filtered_df))
    
    with col2:
        st.pyplot(plot_top_states(filtered_df))
    
    st.subheader("Status & Classification")
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(plot_status_split(filtered_df))
    
    with col2:
        st.pyplot(plot_category_class(filtered_df))
    
    st.subheader("Financial Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(plot_listing_status(filtered_df))
    
    with col2:
        st.pyplot(plot_capital_distribution(filtered_df))
    
    st.subheader("Company Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(plot_company_type_distribution(filtered_df))
    
    with col2:
        st.pyplot(plot_monthly_registrations(filtered_df))
    
    st.subheader("Regulatory Analysis")
    st.pyplot(plot_roc_distribution(filtered_df))
    
    # Refresh button
    if st.button("Refresh Analysis Data"):
        st.session_state.analysis_df = st.session_state.tracker.get_companies_full_df()
        st.rerun()

def render_chat_assistant():
    st.header("AI Chat Assistant")
    st.markdown("Ask questions about your company data using AI")
    
    st.sidebar.subheader("Example Questions")
    example_questions = [
        "How many companies are in the database?",
        "Show me recent company changes",
        "What are the top states for company registrations?",
        "How many active vs inactive companies?",
        "List new companies added recently",
        "Show company registration trends by year"
    ]

    for question in example_questions:
        if st.sidebar.button(question, key=question):
            st.session_state.selected_question = question
    
    chat_container = st.container()
    
    with chat_container:
        for i, chat in enumerate(st.session_state.chat_history):
            if chat['role'] == 'user':
                st_message(chat['content'], is_user=True, key=f"user_{i}")
            else:
                st_message(chat['content'], is_user=False, key=f"assistant_{i}")
    
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            default_value = st.session_state.get('selected_question', '')
            user_input = st.text_input(
                "Type your question about the company data...",
                value=default_value,
                key="chat_input_field",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("Send", type="primary")

    if submit_button and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.process_question(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        if 'selected_question' in st.session_state:
            del st.session_state.selected_question
        
        st.rerun()
 
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        if 'selected_question' in st.session_state:
            del st.session_state.selected_question
        st.rerun()

if __name__ == "__main__":
    main()