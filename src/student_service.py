from typing import Optional, Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor
from src.config import DB_CONFIG

# Mock SDMIS relational records covering Phase 2 Modules 1 to 5
STUDENT_DATABASE: Dict[str, Dict[str, Any]] = {
    "270041123456": {
        "enrollment_no": "270041123456",
        "student_name": "Taran Mahajan",
        "course": "Senior Secondary (12th)",
        "course_level": "Senior Secondary (12th)",
        "admission_stream": "Stream-1 (Block-1)",
        "admission_status": "Confirmed",
        "study_centre": "AI-270041 (Sarvodaya Vidyalaya, Delhi)",
        "study_centre_code": "AI-270041",
        "exam_centre": "Centre-8102 (Kendriya Vidyalaya, Sector 8, Rohini)",
        "hall_ticket_status": "Generated & Available for Download",
        "pending_documents": [],
        "fee_status": "Fully Paid (Ref: NIOS-FEE-99124)",
        "registered_subjects": [
            {"code": "301", "name": "Hindi", "type": "Theory", "exam_date": "12-Oct-2026", "time": "2:30 PM - 5:30 PM"},
            {"code": "316", "name": "Geography", "type": "Theory + Practical", "exam_date": "21-Oct-2026", "time": "2:30 PM - 5:30 PM"},
            {"code": "311", "name": "Mathematics", "type": "Theory", "exam_date": "28-Oct-2026", "time": "2:30 PM - 5:30 PM"}
        ],
        # Module 3: Results & Marks Data
        "results": {
            "exam_session": "April-May 2026",
            "overall_status": "PASSED",
            "total_marks": "382/500",
            "digilocker_status": "Available / Pushed to DigiLocker",
            "subject_marks": [
                {"code": "301", "name": "Hindi", "theory": 68, "practical": "-", "tma": 18, "total": 86, "grade": "A2"},
                {"code": "316", "name": "Geography", "theory": 48, "practical": 26, "tma": 16, "total": 90, "grade": "A1"},
                {"code": "311", "name": "Mathematics", "theory": 55, "practical": "-", "tma": 17, "total": 72, "grade": "B1"},
                {"code": "312", "name": "Physics", "theory": 42, "practical": 22, "tma": 14, "total": 78, "grade": "B1"},
                {"code": "313", "name": "Chemistry", "theory": 38, "practical": 24, "tma": 15, "total": 77, "grade": "B1"}
            ]
        },
        # Module 4: Fee & Payment History
        "payments": {
            "total_paid": "₹ 3,450.00",
            "pending_dues": "₹ 0.00",
            "transactions": [
                {"txn_id": "NIOS-PAY-99812", "date": "15-Jan-2026", "purpose": "Stream-1 Admission & Registration Fee", "amount": "₹ 2,200.00", "status": "Success"},
                {"txn_id": "NIOS-PAY-10492", "date": "10-Jun-2026", "purpose": "Public Exam Fee (3 Theory + 2 Practical)", "amount": "₹ 1,250.00", "status": "Success"}
            ]
        },
        # Module 5: Grievances & Service Requests
        "grievances": [
            {"ticket_id": "GRV-2026-4412", "date": "04-Sep-2026", "category": "Correction in Father Name", "status": "Resolved / Updated", "last_update": "18-Sep-2026"},
            {"ticket_id": "GRV-2026-8921", "date": "20-Oct-2026", "category": "Study Material Delivery Inquiry", "status": "In Progress (Dispatched via SpeedPost)", "last_update": "24-Oct-2026"}
        ]
    }
}

def get_student_context_by_enrollment(enrollment_no: str) -> Optional[Dict[str, Any]]:
    """
    Fetches strictly scoped student facts from relational records.
    Never uses vector similarity for personal records to guarantee zero data leakage.
    """
    sql = """
    SELECT 
        s.enrollment_no,
        s.student_name,
        s.course_level,
        s.admission_stream,
        s.admission_status,
        s.study_centre_code,
        s.exam_centre,
        s.fee_status,
        s.registered_subjects,
        s.pending_documents,
        s.hall_ticket_status,
        s.results_json,
        s.payments_json,
        s.grievances_json
    FROM student_profiles s
    WHERE s.enrollment_no = %s;
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
        cur = conn.cursor()
        cur.execute(sql, (enrollment_no,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return dict(row)
    except Exception:
        pass

    # Direct fallback to relational mock adapter
    return STUDENT_DATABASE.get(enrollment_no)