import mysql.connector
import subprocess
import json
import re
import threading
from time import time
from typing import List, Dict, Any, Tuple

class CompanyChatbot:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conversation_history = []
        self.cache_stats = {}
        self.cache_time = 0
        self.cache_duration = 300 
        
    def _get_db_connection(self):
        return mysql.connector.connect(**self.db_config)
    
    def get_cached_stats(self) -> Dict[str, Any]:
        current_time = time.time()
        if current_time - self.cache_time < self.cache_duration and self.cache_stats:
            return self.cache_stats
         
        self.cache_stats = self._get_database_stats()
        self.cache_time = current_time
        return self.cache_stats
    
    def _get_database_stats(self) -> Dict[str, Any]:
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            stats = {}
            
            cursor.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM companies) as total_companies,
                    (SELECT COUNT(*) FROM change_log) as total_changes,
                    (SELECT COUNT(*) FROM companies_archive) as total_archived
            """)
            counts = cursor.fetchone()
            stats['total_companies'] = counts[0]
            stats['total_changes'] = counts[1]
            stats['total_archived'] = counts[2]
            
            cursor.execute("""
                SELECT change_type, COUNT(*) 
                FROM change_log 
                WHERE change_timestamp >= DATE_SUB(NOW(), INTERVAL 30 DAY)
                GROUP BY change_type
                LIMIT 10
            """)
            recent_changes = cursor.fetchall()
            stats['recent_changes'] = dict(recent_changes) if recent_changes else {}
            
            # Get top states (limited for performance)
            cursor.execute("""
                SELECT company_state_code, COUNT(*) 
                FROM companies 
                WHERE company_state_code IS NOT NULL AND company_state_code != ''
                GROUP BY company_state_code 
                ORDER BY COUNT(*) DESC 
                LIMIT 5
            """)
            top_states = cursor.fetchall()
            stats['top_states'] = dict(top_states) if top_states else {}
            
            # Get status distribution (limited for performance)
            cursor.execute("""
                SELECT company_status, COUNT(*) 
                FROM companies 
                WHERE company_status IS NOT NULL AND company_status != ''
                GROUP BY company_status 
                ORDER BY COUNT(*) DESC
                LIMIT 5
            """)
            status_dist = cursor.fetchall()
            stats['status_distribution'] = dict(status_dist) if status_dist else {}
            
            cursor.close()
            conn.close()
            
            return stats
            
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {
                'total_companies': 0,
                'total_changes': 0,
                'total_archived': 0,
                'recent_changes': {},
                'top_states': {},
                'status_distribution': {}
            }
    
    def execute_sql_query(self, query: str) -> Tuple[List[Dict], str]:
        try:
            # Enhanced security check
            destructive_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'TRUNCATE', 'GRANT', 'REVOKE']
            if any(keyword in query.upper().split() for keyword in destructive_keywords):
                return [], "Sorry, I can only execute SELECT queries for data analysis."
          
            if not query.upper().strip().endswith('LIMIT') and 'LIMIT' not in query.upper():
                query = query.rstrip(';') + " LIMIT 10;"
            
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
        if hasattr(self, '_cached_schema'):
            return self._cached_schema
            
        schema_info = []
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            tables = ['companies', 'change_log', 'companies_archive']
            for table in tables:
                cursor.execute(f"DESCRIBE {table}")
                columns = cursor.fetchall()
                schema_info.append(f"Table: {table}")
                for col in columns[:8]: 
                    schema_info.append(f"  - {col[0]} ({col[1]})")
                if len(columns) > 8:
                    schema_info.append(f"  - ... and {len(columns) - 8} more columns")
                schema_info.append("")
            
            cursor.close()
            conn.close()
            
            self._cached_schema = "\n".join(schema_info)
            return self._cached_schema
            
        except Exception as e:
            return f"Error getting schema: {e}"
    
    def query_ollama_optimized(self, prompt: str, context: str = "") -> str:
        try:
            # Truncate context if too long for performance
            if len(context) > 2000:
                context = context[:2000] + "... [truncated]"
            
            full_prompt = f"""
            You are a helpful assistant for a company database tracking system. 
            The database contains information about Indian companies.

            CONTEXT INFORMATION:
            {context}

            USER QUESTION: {prompt}

            Please provide a concise, helpful response. If suggesting SQL queries, keep them simple.
            Be direct and factual.
            """
          
            result = subprocess.run(
                ['ollama', 'run', 'llama3.2', full_prompt],
                capture_output=True,
                text=True,
                timeout=30, 
                encoding='utf-8',
                errors='ignore' 
            )
            
            if result.returncode == 0 and result.stdout.strip():
                response = result.stdout.strip()
                if len(response) > 1000:
                    response = response[:1000] + "..."
                return response
            else:
                return "I apologize, but I'm having trouble processing your request right now. Please try again with a simpler question."
                
        except subprocess.TimeoutExpired:
            return "The request took too long to process. Please try a simpler question or check if Ollama is running properly."
        except UnicodeDecodeError as e:
            return "There was an encoding issue with the response. Please try again."
        except Exception as e:
            return f"Unable to connect to AI service. Please ensure Ollama is installed and running. Error: {str(e)}"
    
    def generate_sql_from_question(self, question: str) -> str:
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['how many', 'count', 'total']):
            if 'compan' in question_lower:
                return "SELECT COUNT(*) as total FROM companies"
            elif 'change' in question_lower or 'log' in question_lower:
                return "SELECT COUNT(*) as total FROM change_log"
            elif 'archive' in question_lower:
                return "SELECT COUNT(*) as total FROM companies_archive"
                
        elif any(word in question_lower for word in ['list', 'show', 'display']):
            if 'new' in question_lower:
                return "SELECT cin, company_name FROM companies WHERE sync_flag = 'NEW' LIMIT 10"
            elif 'recent' in question_lower or 'change' in question_lower:
                return "SELECT cin, change_type, change_timestamp FROM change_log ORDER BY change_timestamp DESC LIMIT 10"
            else:
                return "SELECT cin, company_name, company_status FROM companies LIMIT 10"
                
        elif 'state' in question_lower:
            return "SELECT company_state_code, COUNT(*) as count FROM companies GROUP BY company_state_code ORDER BY count DESC LIMIT 5"
            
        elif 'status' in question_lower:
            return "SELECT company_status, COUNT(*) as count FROM companies GROUP BY company_status ORDER BY count DESC LIMIT 5"
            
        elif 'active' in question_lower:
            return "SELECT COUNT(*) as active_count FROM companies WHERE company_status LIKE '%Active%'"
            
        elif 'category' in question_lower:
            return "SELECT company_category, COUNT(*) as count FROM companies GROUP BY company_category ORDER BY count DESC LIMIT 5"
        
        return "SELECT COUNT(*) as total_companies FROM companies"
    
    def process_question_fast(self, question: str) -> str:
        self.conversation_history.append({"role": "user", "content": question})
        if len(self.conversation_history) > 10:  # Keep only last 10 messages
            self.conversation_history = self.conversation_history[-10:]
        
        db_stats = self.get_cached_stats()
        
        context = f"""
        Database: {db_stats.get('total_companies', 0):,} companies, {db_stats.get('total_changes', 0):,} changes.
        Top states: {list(db_stats.get('top_states', {}).keys())[:3]}
        Status: {list(db_stats.get('status_distribution', {}).keys())[:3]}
        """
        
        simple_answers = self._get_simple_answer(question, db_stats)
        if simple_answers:
            response = simple_answers
        else:
            response = self.query_ollama_optimized(question, context)
        
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def _get_simple_answer(self, question: str, stats: Dict) -> str:
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['how many', 'count', 'total']):
            if 'compan' in question_lower:
                return f"There are {stats.get('total_companies', 0):,} companies in the database."
            elif 'change' in question_lower or 'log' in question_lower:
                return f"There are {stats.get('total_changes', 0):,} changes logged in the system."
            elif 'archive' in question_lower:
                return f"There are {stats.get('total_archived', 0):,} archived companies."
                
        elif 'hello' in question_lower or 'hi ' in question_lower or 'hey' in question_lower:
            return "Hello! I'm your company database assistant. I can help you query information about companies, changes, and statistics."
            
        elif 'help' in question_lower:
            return """I can help you with:
• Company counts and statistics
• Recent changes and updates  
• State-wise distributions
• Status information
• Basic data queries

Try asking about total companies, recent changes, or state distributions!"""
        
        return ""
    
    def clear_history(self):
        self.conversation_history = []