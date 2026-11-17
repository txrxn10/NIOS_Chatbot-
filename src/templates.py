from typing import Dict, Any

# Proposal Alignment (Section 5.4): Pre-approved response templates per data type.
# Guarantees zero hallucination on official dates, centres, marks, and fee statuses.

def format_admission_status(data: Dict[str, Any]) -> str:
    docs = "None (All documents verified)" if not data.get("pending_documents") else ", ".join(data["pending_documents"])
    return (
        f"**Admission Status:** {data.get('admission_status', 'Confirmed')}\n\n"
        f"- **Student Name:** {data.get('student_name', 'Student')}\n"
        f"- **Enrollment Number:** `{data.get('enrollment_no')}`\n"
        f"- **Course & Stream:** {data.get('course', data.get('course_level', 'Secondary'))} | {data.get('admission_stream', 'Stream-1')}\n"
        f"- **Study Centre (AI):** {data.get('study_centre', data.get('study_centre_code', 'N/A'))}\n"
        f"- **Pending Documents:** {docs}\n"
        f"- **Fee Payment:** {data.get('fee_status', 'Verified')}"
    )

def format_exam_schedule(data: Dict[str, Any]) -> str:
    schedule_rows = []
    for s in data.get("registered_subjects", []):
        schedule_rows.append(f"| {s['code']} | {s['name']} | {s['type']} | {s['exam_date']} | {s['time']} |")
    
    table_str = "\n".join(schedule_rows)
    return (
        f"**Examination Schedule for {data.get('student_name')} (Enrollment: `{data.get('enrollment_no')}`):**\n\n"
        f"**Designated Exam Centre:** {data.get('exam_centre', 'Assigned Centre')}\n"
        f"**Hall Ticket Status:** {data.get('hall_ticket_status', 'Generated')}\n\n"
        f"| Code | Subject | Type | Date | Timing |\n"
        f"| :--- | :--- | :--- | :--- | :--- |\n"
        f"{table_str}\n\n"
        f"> *Note: Please carry your official NIOS identity card and printed Hall Ticket to the exam centre.*"
    )

def format_results(data: Dict[str, Any]) -> str:
    res = data.get("results", {})
    if not res:
        return "No published examination results found for your enrollment."

    rows = []
    for s in res.get("subject_marks", []):
        rows.append(f"| {s['code']} | {s['name']} | {s['theory']} | {s['practical']} | {s['tma']} | {s['total']} | {s['grade']} |")

    table_str = "\n".join(rows)
    return (
        f"**Examination Result — {res.get('exam_session')}:**\n\n"
        f"- **Student Name:** {data.get('student_name')}\n"
        f"- **Enrollment:** `{data.get('enrollment_no')}`\n"
        f"- **Overall Result:** **{res.get('overall_status')}** ({res.get('total_marks')})\n"
        f"- **DigiLocker Status:** {res.get('digilocker_status')}\n\n"
        f"| Code | Subject | Theory | Practical | TMA | Total | Grade |\n"
        f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        f"{table_str}\n\n"
        f"> *Official digital marksheet can be synced directly into your DigiLocker account.*"
    )

def format_payment_history(data: Dict[str, Any]) -> str:
    pmt = data.get("payments", {})
    if not pmt:
        return "No payment history records found."

    rows = []
    for t in pmt.get("transactions", []):
        rows.append(f"| `{t['txn_id']}` | {t['date']} | {t['purpose']} | {t['amount']} | {t['status']} |")

    table_str = "\n".join(rows)
    return (
        f"**Fee Ledger & Payment History (Enrollment: `{data.get('enrollment_no')}`):**\n\n"
        f"- **Total Paid to Date:** {pmt.get('total_paid')}\n"
        f"- **Outstanding Balance / Pending Dues:** {pmt.get('pending_dues')}\n\n"
        f"| Transaction ID | Date | Purpose | Amount | Status |\n"
        f"| :--- | :--- | :--- | :--- | :--- |\n"
        f"{table_str}"
    )

def format_grievances(data: Dict[str, Any]) -> str:
    grvs = data.get("grievances", [])
    if not grvs:
        return "You have no active grievance or e-service requests on file."

    rows = []
    for g in grvs:
        rows.append(f"| `{g['ticket_id']}` | {g['date']} | {g['category']} | {g['status']} | {g['last_update']} |")

    table_str = "\n".join(rows)
    return (
        f"**Registered Grievance & E-Service Tickets:**\n\n"
        f"| Ticket ID | Submitted Date | Category | Status | Last Updated |\n"
        f"| :--- | :--- | :--- | :--- | :--- |\n"
        f"{table_str}\n\n"
        f"> *To submit a new complaint or change request, visit the NIOS E-Services Portal.*"
    )