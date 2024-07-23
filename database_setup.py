import sqlite3

#DATABASE SETUP

def init_db():
    conn = sqlite3.connect('lung_disease_detection.db')
    c = conn.cursor()

    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            role TEXT
        )
    ''')

    # Create patients table
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            dob TEXT,
            gender TEXT,
            id_number TEXT,
            contact_details TEXT,
            medical_history TEXT,
            medication_history TEXT,
            clinical_notes TEXT,
            vital_signs TEXT,
            immunization_history TEXT,
            insurance_details TEXT,
            consent_forms TEXT,
            emergency_contacts TEXT,
            provider_details TEXT,
            lifestyle_factors TEXT
        )
    ''')

    # Create reports table
    c.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            report TEXT,
            created_by TEXT,
            created_at TEXT,
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        )
    ''')

    conn.commit()
    conn.close()

init_db()