import json
import re
import os
import pypdf

PDF_PATH = "data/Academic_Prospectus_2021_22.pdf"
OUTPUT_JSON = "data/nios_prospectus_chunks.json"

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_structured_knowledge_base():
    reader = pypdf.PdfReader(PDF_PATH)
    total_pages = len(reader.pages)
    print(f"Loaded PDF with {total_pages} pages.")

    chunks = []
    chunk_counter = 1

    def add_chunk(text: str, archetype: str, page: int, title: str, section: str = ""):
        nonlocal chunk_counter
        # Format chunk with contextual prefix for high-accuracy embedding & BM25 recall
        context_header = f"Document: NIOS Academic Prospectus 2021-22 | Section: {section} | Page: {page}\nTitle: {title}\n"
        full_content = context_header + text.strip()
        
        chunks.append({
            "chunk_id": f"chunk_{chunk_counter:04d}",
            "text": full_content,
            "metadata": {
                "source": PDF_PATH,
                "page": page,
                "type": archetype,
                "title": title,
                "section": section
            }
        })
        chunk_counter += 1

    # ==========================================
    # 1. CORE TABULAR CHUNKS (Explicit High-Fidelity Tables)
    # ==========================================
    
    # Table 1: Scheme of Studies (Class X & XII)
    add_chunk(
        """Table-1: Scheme of Studies for Academic Courses

### Secondary Level (Class X)
- Group 'A' (Languages): (201) Hindi, (202) English, (203) Bengali, (204) Marathi, (205) Telugu, (206) Urdu, (207) Gujarati, (208) Kannada, (209) Sanskrit, (210) Punjabi, (228) Assamese, (231) Nepali, (232) Malayalam, (233) Odia, (235) Arabic, (236) Persian, (237) Tamil, (238) Sindhi.
- Group 'B' (Academic Subjects): (211) Mathematics*, (212) Science & Technology*, (213) Social Science, (214) Economics, (215) Business Studies, (216) Home Science*, (222) Psychology, (223) Indian Culture & Heritage, (224) Accountancy, (225) Painting*, (229) Data Entry Operations*, (242) Hindustani Sangeet*, (243) Carnatic Sangeet*, (245) Veda Adhyayan#, (246) Sanskrit Vyakarana#, (247) Bharatiya Darshan#, (248) Sanskrit Sahitya#, (249) Entrepreneurship.
* Asterisk (*) indicates subjects with practical and theory.
# Hash (#) indicates Indian Knowledge Tradition subjects available in Sanskrit medium only.
Certification Rule: Pass in 5 subjects with minimum 1 or maximum 2 languages from Group A and remaining from Group B.

### Senior Secondary Level (Class XII)
- Group 'A' (Languages): (301) Hindi, (302) English, (303) Bengali, (304) Tamil, (305) Odia, (306) Urdu, (307) Gujarati, (309) Sanskrit, (310) Punjabi, (341) Arabic, (342) Persian, (343) Malayalam.
- Group 'B': (311) Mathematics, (316) Geography*, (318) Economics, (319) Business Studies, (320) Accountancy, (321) Home Science*, (328) Psychology, (332) Painting*, (336) Data Entry Operations*, (337) Tourism, (338) Introduction to Law, (339) Library & Information Science*, (345) Veda Adhyayan#, (346) Sanskrit Vyakarana#, (347) Bharatiya Darshan#, (348) Sanskrit Sahitya#, (376) Early Childhood Care & Education.
- Group 'C': (312) Physics*, (315) History, (333) Environmental Science*, (374) Military Studies.
- Group 'D': (313) Chemistry*, (317) Political Science, (335) Mass Communication*, (375) Military History.
- Group 'E': (314) Biology*, (373) Physical Education and Yog*.
- Group 'F': (330) Computer Science*, (331) Sociology.
Rule: Only one subject can be chosen from each of Group C, D, E, and F.""",
        archetype="table_data",
        page=5,
        title="Table-1 : Scheme of Studies for Academic Courses",
        section="1.3 Scheme of Studies"
    )

    # Table 1-A: Medium Availability Matrix
    # Table 1-A: Medium Availability Matrix
    add_chunk(
        """Table 1.A: Availability of Study Material in Different Mediums

### Secondary Level (Class X) Subject Availability:
- (211) Mathematics: Available in English, Hindi, Urdu, Odia, Gujarati, Marathi, Telugu, Malayalam, Tamil, Assamese, Kannada. NOT available in Sanskrit, Punjabi.
- (212) Science & Technology: Available in English, Hindi, Urdu, Odia, Gujarati, Marathi, Telugu, Malayalam, Tamil, Punjabi, Kannada. NOT available in Sanskrit, Assamese.
- (213) Social Science: Available in English, Hindi, Urdu, Odia, Gujarati, Marathi, Telugu, Malayalam, Tamil, Punjabi, Kannada. NOT available in Sanskrit, Assamese.
- (214) Economics: Available in English, Hindi, Urdu, Odia, Gujarati, Marathi, Telugu, Malayalam, Tamil, Kannada. NOT available in Punjabi. NOT available in Assamese. NOT available in Sanskrit.
- (215) Business Studies: Available in English, Hindi, Urdu, Odia, Gujarati, Marathi, Telugu, Malayalam, Tamil, Punjabi, Assamese. NOT available in Kannada. NOT available in Sanskrit.
- (216) Home Science: Available in English, Hindi, Urdu, Odia, Gujarati, Marathi, Telugu, Malayalam, Tamil, Punjabi, Assamese, Kannada.
- (222) Psychology: Available in English, Hindi, Urdu, Odia, Gujarati, Marathi, Telugu, Malayalam, Tamil, Assamese. NOT available in Punjabi, Kannada.
- (223) Indian Culture & Heritage (ICH): Available in English, Hindi, Urdu, Odia, Gujarati, Marathi, Telugu, Malayalam, Tamil. NOT available in Punjabi, Assamese, Kannada.
- (224) Accountancy: Available in English, Hindi, Urdu, Gujarati, Marathi, Telugu, Malayalam, Tamil, Punjabi. NOT available in Odia, Assamese, Kannada.
- (225) Painting: Available in English, Hindi, Urdu, Odia, Gujarati, Marathi, Telugu, Malayalam, Tamil. NOT available in Punjabi, Assamese, Kannada.
- (229) Data Entry Operations: Available in English, Hindi, Urdu, Odia, Gujarati, Marathi, Telugu, Malayalam, Tamil, Assamese. NOT available in Punjabi, Kannada.
- (242) Carnatic Music: Available in English only.
- (243) Hindustani Music: Available in English and Hindi only.
- (245) Veda Adhyayan, (246) Sanskrit Vyakarana, (247) Bharatiya Darshan, (248) Sanskrit Sahitya: Available in Hindi and Sanskrit only.
- (249) Entrepreneurship: Available in English only.

### Senior Secondary Level (Class XII) Subject Availability:
- (311) Maths, (312) Physics, (313) Chemistry, (314) Biology, (315) History, (316) Geography, (317) Political Science, (318) Economics, (319) Business Studies, (320) Accountancy, (321) Home Science, (328) Psychology, (330) Computer Science, (331) Sociology, (332) Painting, (333) Environmental Science, (335) Mass Communication, (336) Data Entry: Available in English, Hindi, Urdu, Odia, Gujarati, Bengali.
- (337) Tourism, (338) Introduction to Law: Available in English, Hindi.
- (339) Library and Information Science: Available in English, Hindi, Urdu.
- (345-348) Indian Knowledge Tradition: Available in Hindi, Sanskrit.
- (373) Physical Education, (374) Military Studies, (375) Military History, (376) ECCE: Available in English only.""",
        archetype="table_data",
        page=6,
        title="Table 1.A : Availability of Study Material in Mediums",
        section="1.3 Scheme of Studies"
    )
# ==========================================
    # 1. CORE TABULAR CHUNKS (Explicit High-Fidelity Tables)
    # ==========================================

    # --- ADD FIX 1: OBE Pathways Hierarchy (Page 3) ---
    add_chunk(
        """Section 1.2 & Diagram: Programmes, Courses and Pathways (OBE Structure)

- Open Basic Education (OBE) Standard Equivalencies:
  * OBE-A level: Equivalent to 3rd standard (Primary level)
  * OBE-B level: Equivalent to 5th standard (Primary level)
  * OBE-C level: Equivalent to 8th standard (Middle level)
- Academic Progression Flow:
  * OBE-A and OBE-B lead into OBE-C level.
  * OBE-C level feeds directly into Secondary Course (Class X) OR Vocational Courses.
  * Secondary Course leads into Senior Secondary Course (Class XII) OR Vocational Courses.
  * Senior Secondary leads into higher education or Vocational Courses.""",
        archetype="workflow_procedure",
        page=3,
        title="Programmes, Courses and Pathways (OBE Structure)",
        section="1.2 Programmes, Courses and Pathways"
    )

    # --- ADD FIX 2: Explicit Negative Language Constraints for Table 1-A (Page 6) ---
    # Update your existing Table 1.A chunk or replace the Economics / Business Studies rows with:
    # (214) Economics: Available in English, Hindi, Urdu, Odia, Gujarati, Marathi, Telugu, Malayalam, Tamil, Kannada. NOT available in Punjabi. NOT available in Assamese. NOT available in Sanskrit.
    # (215) Business Studies: Available in English, Hindi, Urdu, Odia, Gujarati, Marathi, Telugu, Malayalam, Tamil, Punjabi, Assamese. NOT available in Kannada. NOT available in Sanskrit.
    # Section 1.2.2: Schooling of ITI Students & DGT MoU
    add_chunk(
        """Section 1.2.2: Schooling of ITI Students & DGT MoU

Under this scheme, ITI students get an opportunity to earn the Secondary and Senior Secondary Certificate from NIOS who are pursuing education through ITIs or have cleared ITI courses.
- Memorandum of Understanding (MoU): Signed with Directorate General of Training (DGT), Ministry of Skill Development and Entrepreneurship (MSDE), Government of India.
- Credit Transfer: The learner receives three (3) credits from ITI under the TOC scheme.
- Additional Subjects Required: The candidate needs to study and pass two additional subjects from NIOS:
  1. One language course of NIOS.
  2. One academic subject offered by NIOS.
- Certification: Upon clearing the two subjects from NIOS, candidate receives a combined marksheet under the TOC scheme.""",
        archetype="workflow_procedure",
        page=4,
        title="Schooling of ITI Students (DGT MoU & Credit Transfer)",
        section="1.2.2 Schooling of ITI Students"
    )

    # Section 2.1.1: Stream-1 Admission Dates & Late Fee Slabs
    add_chunk(
        """Section 2.1.1 & Important Dates: Online Admission Schedule and Late Fee Slabs for Stream-1

### Block-1 (For April/May Public Examination next year):
- 16th March to 31st July: Without late fee
- 1st August to 15th August: With late fee of Rs. 200/-
- 16th August to 31st August: With late fee of Rs. 400/-
- 1st September to 15th September: With late fee of Rs. 700/-
First examination opportunity: April/May examination of the next year.

### Block-2 (For October/November Public Examination next year):
- 16th September to 31st January: Without late fee
- 1st February to 15th February: With late fee of Rs. 200/-
- 16th February to 28th February: With late fee of Rs. 400/-
- 1st March to 15th March: With late fee of Rs. 700/-
First examination opportunity: October/November examination of the next year.

Note: Online admission is open 24x7. Admission fee does NOT include examination fees.""",
        archetype="table_data",
        page=8,
        title="Stream-1 Online Admission Dates and Late Fee Slabs (Block-1 & Block-2)",
        section="2.1.1 Online Admission for all learners (Stream-1)"
    )
    # Table 2: Standalone Vocational Subjects
    add_chunk(
        """Table-2: List of Vocational Education Courses offered in combination with Academic Courses

### Secondary Level (Class X)
- (217) Typewriting (Hindi)
- (218) Typewriting (English)
- (221) Typewriting (Urdu)
- (252) Carpentry
- (253) Solar Energy Technician
- (254) Bio Gas Technician
- (255) Laundry Services
- (256) Bakery & Confectionery
- (257) Welding Technology

### Senior Secondary Level (Class XII)
- (322) Typewriting (Hindi)
- (323) Typewriting (English)
- (364) Typewriting (Urdu)
- (324) Stenography (Hindi)
- (325) Stenography (English)
- (329) Stenography (Urdu)
- (326) Secretarial Practice
- (351) Plant Protection
- (354) Furniture and Cabinet Making
- (355) Electroplating
- (356) House Keeping
- (357) Catering Management
- (358) Food Processing
- (360) Hotel Front Office Operation
- (361) Poultry Farming
- (362) Soil and Fertilizer Management
- (363) Preservation of Fruits and Vegetables
Note: Must take separate admission at an Accredited Vocational Institute (AVI). Passed credit transferred to Academic Certificate upon request.""",
        archetype="table_data",
        page=7,
        title="Table-2 : Vocational Courses with Academic Courses",
        section="1.4 Combination of Vocational and Academic Subjects"
    )

    # Table 3: Entry Requirements
    add_chunk(
        """Table-3: Entry Requirements (Age and Eligibility)

### Secondary Course (Class X)
- Minimum Age: 14 years completed (e.g., as on 31st July for Block-1 or 31st January for Block-2). No upper age limit.
- Minimum Educational Qualification: Certificate of Class VIII pass, OR Self-Certificate stating "I have studied and able to pursue the secondary course".
- Supporting Documents for Date of Birth: Birth Certificate from Municipal Authority/Registrar of Births & Deaths; Passport; Original Transfer Certificate/School Leaving Certificate; or Aadhaar Card with complete dd/mm/yyyy.

### Senior Secondary Course (Class XII)
- Minimum Age: 15 years completed (e.g., as on 31st July for Block-1 or 31st January for Block-2).
- Minimum Educational Qualification: Certificate of the Secondary Examination passed from a recognised Board of Secondary Education.
- Supporting Document for Date of Birth: Matriculation/Secondary Certificate having date of birth record. No other proof accepted.""",
        archetype="table_data",
        page=16,
        title="Table-3 : Entry Requirements",
        section="2.4 Entry Requirements"
    )

    # Table 4: Fee Structure Stream-1
    add_chunk(
        """Table-4: Fee Structure for Online Admission of Stream-1

### Secondary Course (Class X)
- General Category Male: Rs. 1800/- (for 5 subjects)
- General Category Female: Rs. 1450/- (for 5 subjects)
- Exempted Category (SC/ST, Ex-Servicemen, Persons with Disabilities - PWD): Rs. 1200/- (for 5 subjects)
- Additional Subject Fee: Rs. 720/- per subject (for all categories)

### Senior Secondary Course (Class XII)
- General Category Male: Rs. 2000/- (for 5 subjects)
- General Category Female: Rs. 1650/- (for 5 subjects)
- Exempted Category (SC/ST, Ex-Servicemen, PWD): Rs. 1300/- (for 5 subjects)
- Additional Subject Fee: Rs. 720/- per subject (for all categories)

Online Transaction Fee: In addition to above, Rs. 50/- charged per online transaction.
Note: Admission fee includes registration, cost of study materials, PCPs, ID card, and assignment evaluation. It does NOT include examination fees.""",
        archetype="table_data",
        page=19,
        title="Table-4 : Fee Structure for Stream-1 Admission",
        section="2.8 Fee Structure for Admission"
    )

    # Table 6A & 6B: Dual Enrolment, Part Admission, Streams 2, 3, 4
    add_chunk(
        """Table-6A & Table-6B: Fee Structure for Dual Enrolment, Part Admission, Streams 2, 3 and 4

### Table-6A: Stream-1 Dual Enrolment / Part Admission / TOC
- Secondary: Registration Fee Rs. 480/- per subject + Cost of Material Rs. 360/- per subject + TOC Fee Rs. 180/- per subject.
- Senior Secondary: Registration Fee Rs. 480/- per subject + Cost of Material Rs. 400/- per subject + TOC Fee Rs. 180/- per subject.

### Table-6B: Online Admission for Streams 2, 3 and 4
- 1. Stream-2 (Secondary/Sr. Secondary Public Exam):
  * Registration Fee: Rs. 480/- per subject
  * Exam Fee: Rs. 250/- per subject for Theory; Additional Rs. 120/- per subject for Practical.
  * TOC Fee: Rs. 180/- per subject.
  * Change/Additional Subject Fee: Rs. 720/- per subject.
- 2. Stream-3 (Secondary On-Demand Examination - ODES):
  * Registration Fee: Rs. 480/- per subject
  * Exam Fee: Rs. 500/- per subject for Theory; Additional Rs. 200/- per subject for Practical.
  * TOC Fee: Rs. 180/- per subject.
  * Change/Additional Subject Fee: Rs. 720/- per subject.
- 3. Stream-4 (Senior Secondary On-Demand Examination - ODES):
  * Registration Fee: Rs. 480/- per subject
  * Exam Fee: Rs. 500/- per subject for Theory; Additional Rs. 200/- per subject for Practical.
  * TOC Fee: Rs. 180/- per subject.
  * Change/Additional Subject Fee: Rs. 720/- per subject.
Online transaction fee: Rs. 50/- per transaction.""",
        archetype="table_data",
        page=20,
        title="Table-6A & 6B : Fee Structure for Streams 2, 3, 4 and Dual Enrolment",
        section="2.8 Fee Structure for Admission"
    )

    # Section 2.12: Correction Fees
    add_chunk(
        """Fee for Correction in Admission Record & Administrative Services (Section 2.12)

1. Correction in Learner's / Father's / Mother's / Guardian's name: Rs. 100/-
2. Correction in Date of Birth: Rs. 100/-
3. Change in Learner's / Parent's / Guardian's name: Rs. 200/-
4. Change in Date of Birth (as per rule): Rs. 200/-
5. Correction of Photograph: Rs. 100/-
6. Change in learner's address: Rs. 100/-
7. Change of Study Centre (AI): Rs. 400/- (Allowed till 30th April for Oct-Nov exam, and 30th October for April-May exam)
8. Transfer of Credit (TOC) fee: Rs. 150/- per subject at time of admission
9. TOC fee after registration: Rs. 500/- per subject
10. Late Fee for submission of Assignments (TMA): Rs. 1500/- per subject
11. Issue of Duplicate Identity Card (first time): Rs. 100/- (with copy of police FIR)
12. Issue of Duplicate Identity Card for the second time: Rs. 500/-
13. Change of Examination Centre for Public Exam or ODES: Rs. 1500/-""",
        archetype="table_data",
        page=23,
        title="Fee for Correction in Admission Record and Other Administrative Services",
        section="2.12 Procedure for Correction in Admission Records"
    )

    # Table 7: Examination Fee & Miscellaneous
    add_chunk(
        """Table-7: Examination Fee and Other Miscellaneous Fees

### (i) Public Examination Fees (Secondary and Senior Secondary)
- Examination Fee: Rs. 250/- per subject (Theory)
- Additional fee for Practicals: Rs. 120/- per subject
- Late fee for examination registration: Rs. 100/- per subject
- Consolidated late fee at Regional Centre: Rs. 1500/- per subject

### (ii) On-Demand Examination (ODE) Fees
- Theory exam fee: Rs. 500/- per subject
- Additional practical fee: Rs. 200/- per subject

### (iii) Miscellaneous Document & Processing Fees
- Duplicate Pass Certificate: Rs. 200/-
- Duplicate Provisional Certificate: Rs. 200/-
- Duplicate Migration Certificate: Rs. 200/-
- Duplicate Marksheet: Rs. 200/-
- Urgent Issue (Provisional, Migration, Marksheet within 48 hrs): Rs. 400/- each
- Postal Speed Post delivery charge: Rs. 50/-
- Re-checking of Answer Script: Rs. 300/- per subject (Theory only, apply within 15 days)
- Re-evaluation of Answer Script (Senior Secondary Public Exam): Rs. 800/- per subject (Theory only, apply within 15 days)
- Re-evaluation of Answer Script (Senior Secondary ODE): Rs. 1000/- per subject (Theory only, apply within 15 days)
- Photocopy of Answer Script: Rs. 700/- per answer script (within 15 days of result)
- Document verification for private companies: Rs. 500/- (Urgent within 48 hrs: Rs. 1000/-)
- Verification/Transcription for foreign institutions: Rs. 5000/-""",
        archetype="table_data",
        page=32,
        title="Table-7 : Examination Fee and Other Miscellaneous Fees",
        section="4.5 Issue of Marksheet and Certificate"
    )

    # Table 8: Certification Criteria
    add_chunk(
        """Table-8: Certification Criteria and Passing Standards

### Secondary Course (Class X)
- Passing Standard in a Subject: Minimum of 33% marks in aggregate (Theory plus Practical combined). Internal Assessment (TMA) marks shown separately.
- Certification Criteria: Pass in 5 subjects including at least one but not more than two languages from Group A.

### Senior Secondary Course (Class XII)
- Passing Standard in a Subject: Minimum of 33% marks in aggregate AS WELL AS separately in theory and practicals in public exam.
- Certification Criteria: Pass in 5 subjects including at least one but not more than two languages from Group A.""",
        archetype="table_data",
        page=33,
        title="Table-8 : Certification Criteria",
        section="4.6 Certification Criteria"
    )

    # Table 9: Scheme of Examination (Marks & Duration)
    add_chunk(
        """Table-9: Scheme of Examination (Number of Papers, Maximum Marks, and Duration)

### Secondary Level (Class X)
- (201) Hindi, (202) English, (206) Urdu, (209) Sanskrit, (203) Bengali, (204) Marathi, (205) Telugu, (207) Gujarati, (208) Kannada, (210) Punjabi, (228) Assamese, (231) Nepali, (232) Malayalam, (233) Odia, (235) Arabic, (236) Persian, (237) Tamil, (238) Sindhi: 1 Paper, 100 Marks, Duration 3 Hours.
- (211) Mathematics: Theory 1 Paper, 85 Marks, 2.5 Hours | Practical 1 Paper, 15 Marks, 3 Hours.
- (212) Science & Technology: Theory 1 Paper, 85 Marks, 2.5 Hours | Practical 1 Paper, 15 Marks, 3 Hours.
- (213) Social Science: 1 Paper, 100 Marks, 3 Hours.
- (214) Economics: 1 Paper, 100 Marks, 3 Hours.
- (215) Business Studies: 1 Paper, 100 Marks, 3 Hours.
- (216) Home Science: Theory 1 Paper, 85 Marks, 2.5 Hours | Practical 1 Paper, 15 Marks, 3 Hours.
- (222) Psychology: 1 Paper, 100 Marks, 3 Hours.
- (223) Indian Culture & Heritage: 1 Paper, 100 Marks, 3 Hours.
- (224) Accountancy: 1 Paper, 100 Marks, 3 Hours.
- (225) Painting: Theory 1 Paper, 30 Marks, 1.5 Hours | Practical 1 Paper, 70 Marks, 3 Hours.
- (229) Data Entry Operations: Theory 1 Paper, 40 Marks, 2 Hours | Practical 1 Paper, 60 Marks, 2 Hours.
- (242) Hindustani Sangeet: Theory 1 Paper, 40 Marks, 2 Hours | Practical 1 Paper, 60 Marks, 2 Hours.
- (243) Carnatic Sangeet: Theory 1 Paper, 40 Marks, 2 Hours | Practical 1 Paper, 60 Marks, 2 Hours.
- (245) Veda Adhyayan, (246) Sanskrit Vyakarana, (247) Bharatiya Darshan, (248) Sanskrit Sahitya: 1 Paper, 100 Marks, 3 Hours.
- (249) Entrepreneurship: 1 Paper, 100 Marks, 3 Hours.

### Senior Secondary Level (Class XII)
- All Languages (301, 302, 306, 309, 307, 303, 304, 305, 310, 341, 342, 343): 1 Paper, 100 Marks, 3 Hours.
- (311) Mathematics: 1 Paper, 100 Marks, 3 Hours.
- (321) Home Science: Theory 80 Marks (3 Hours), Practical 20 Marks (3 Hours).
- (316) Geography: Theory 80 Marks (3 Hours), Practical 20 Marks (3 Hours).
- (318) Economics, (319) Business Studies, (320) Accountancy, (328) Psychology, (331) Sociology, (337) Tourism, (338) Intro to Law: 1 Paper, 100 Marks, 3 Hours.
- (332) Painting: Theory 30 Marks (1.5 Hours), Practical 70 Marks (3 Hours).
- (336) Data Entry Operations: Theory 40 Marks (2 Hours), Practical 60 Marks (2 Hours).
- (376) Early Childhood Care & Education: 1 Paper, 100 Marks, 3 Hours.
- (312) Physics, (313) Chemistry, (314) Biology, (333) Environmental Science: Theory 80 Marks (3 Hours), Practical 20 Marks (3 Hours).
- (315) History, (317) Political Science, (374) Military Studies, (375) Military History: 1 Paper, 100 Marks, 3 Hours.
- (335) Mass Communication: Theory 80 Marks (3 Hours), Practical 20 Marks (3 Hours).
- (330) Computer Science: Theory 60 Marks (3 Hours), Practical 40 Marks (2 Hours).
- (373) Physical Education and Yog: Theory 70 Marks (2 Hours), Practical 30 Marks (2 Hours).""",
        archetype="table_data",
        page=34,
        title="Table-9 : Scheme of Examination (Marks & Durations)",
        section="4.7 Scheme of Examinations"
    )

    # Table 10 & 11: TMA Schedule and Subjects
    add_chunk(
        """Table 10 & 11: Tutor Marked Assignments (TMA) Schedule and List of Subjects

### Table 10: Schedule of Submission of Assignments
- April/May Public Examination:
  * Submission of Assignments by learners to AI (Study Centre): Till 31st January
  * Feedback by Teacher/Facilitator to learners: Till 15th February
- October/November Public Examination:
  * Submission of Assignments by learners to AI (Study Centre): Till 31st July
  * Feedback by Teacher/Facilitator to learners: Till 16th August
- Late Fee Clause: After due date, TMA can be submitted with late fee of Rs. 1500/- per subject online till 15th July (for April exam) and till 15th January (for Oct-Nov exam) at Regional Centre.
- TMA Weightage: Internal assessment carries 20% weightage of theory. External exam carries remaining 80%.
- Stream Eligibility: TMA is applicable ONLY for Stream-1 learners. Not applicable for Streams 2, 3, and 4.""",
        archetype="table_data",
        page=37,
        title="Table 10 & 11 : TMA Schedules and Guidelines",
        section="5. Tutor Marked Assignments (TMAs)"
    )

    # Table 12: Disability Codes
    add_chunk(
        """Table 12: Codes for Marking Different Disabilities

1. Blindness: Code B
2. Low Vision: Code LV
3. Locomotor Disability: Code L
4. Hearing and Speech Impairment: Code HI
5. Leprosy cured: Code LC
6. Mental Illness: Code MI
7. Mental Retardation: Code MR
8. Autism: Code Aut
9. Cerebral Palsy: Code CP
10. Multiple Disabilities: Code MD
11. Specific Learning Disabilities: Code SLD""",
        archetype="table_data",
        page=43,
        title="Table-12 : Disability Codes",
        section="6. Provisions for Persons with Disabilities"
    )

    # ==========================================
    # 2. DIAGRAMMATIC & FLOWCHART CHUNKS
    # ==========================================

    # IVRS Support Tree (Page 2)
    add_chunk(
        """Learner Support Centre (LSC) Toll Free IVRS Navigation Structure (Dial 1800 180 9393)

Language Selection: Press 1 for Hindi | Press 2 for English
Main Menu Options:
- Press 1 for Admission:
  * If NOT yet registered, Press for:
    1. Date of Admission
    2. Equivalency of NIOS Certificate
    3. Admission Fees
    4. Subject selection
    5. Choice of Study Centre
  * If ALREADY REGISTERED, Press for:
    1. Admission status
    2. Dispatch of Course Material
    3. Personal Contact Programme (PCP)
    4. Change of Subject, Additional Subject, and any type of correction
    5. Transfer of Credit (TOC) and Re-admission
- Press 2 for Examination:
  * Press for: 1. Date Sheet, 2. Payment of Examination Fee, 3. Examination Centres, 4. On-Demand Examination, 5. Requirements about Exam / ODE, 6. Tutor Marked Assignments (TMA)
- Press 3 for Result:
  * Press for: 1. Secondary Exam Result, 2. Senior Secondary Exam Result, 3. On-Demand Examination Result, 4. Re-checking of Marks, 5. Mark Sheet / Migration or Provisional Certificate, 6. Duplicate documents
- Press 4 for Academic Courses (Courses offered, Medium of Instruction, Sample Question Papers)
- Press 5 for Vocational Courses and fees
- Press 6 for Life Skill Information
- Press 7 for Accreditation
- Press 9 to Speak to LSC Executive (Customer Support)
- Press * to go back to previous menu""",
        archetype="workflow_procedure",
        page=2,
        title="Learner Support Centre (LSC) IVRS Telephone Tree",
        section="Learner Support Services"
    )

    # Online Admission Flowchart (Page 13)
    add_chunk(
        """Online Admission Process Flowchart (sdmis.nios.ac.in)

1. Access Portal: Visit sdmis.nios.ac.in -> Click 'Register' -> Read instructions -> Click 'Proceed'.
2. Identity Selection: Select State/UT -> Select Identity Type (Aadhaar Number, Aadhaar Enrolment ID, Passport, Ration Card, Bank Account Number, etc.) -> Enter Identity Number. (Aadhaar name and gender are authenticated against UIDAI).
3. Course Selection: Select Secondary (Class X) or Senior Secondary (Class XII) -> Click 'Submit'.
4. Admission Form Sections: Form is divided into 4 primary sections:
   - Section 1: Basic Details (divided into 3 sub-sections: (a) Basic Details - Applicant Name, Father's Name, Mother's Name, DOB, Mobile, Email; (b) Address; (c) Other Information - Category, Medium, Upload Photo & Identity Document).
   - Section 2: Optional Details (Family income, parents' qualifications).
   - Section 3: Subject Selection (Select subjects, board details for Class 10th if Sr. Secondary, specify TOC subjects).
   - Section 4: Study Centre Selection (Choose maximum 3 preferences of AI Study Centres based on state and district).
5. Review & Submission: Review application -> Make edits if needed.
6. Payment: Make online payment via gateway (IDBI, UBI, ICICI, HDFC) or CSC Wallet.
7. Post-Payment: Reference Number generated -> Wait for document verification -> Enrolment Number generated.""",
        archetype="workflow_procedure",
        page=13,
        title="Online Admission Process Flowchart",
        section="2.2 Procedure for Online Admission"
    )

    # On-Demand Examination (ODE) Flowchart (Page 14)
    add_chunk(
        """Online Registration Process for On-Demand Examination (ODE)

Step-by-Step Flow:
- Step 1: Register through OTP Mobile and Email.
- Step 2: Fill in basic details (SKIPPED for existing NIOS learners).
- Step 3: Select Subjects studied/appeared in Board exam (SKIPPED for existing NIOS learners).
- Step 4: Select / Add Subjects in which you want to appear in ODE.
- Step 5: Select Exam Centre where you want to appear for ODE (HQ Noida, Regional Centres, or designated Kendriya Vidyalayas).
- Step 6: View date-wise availability of seats for the chosen Exam Centre.
- Step 7: Select Date of Exam from the available calendar.
- Step 8 & 9: If the desired date of exam is NOT available -> Step 9: Choose dates from the Calendar for next month.
- Step 10: Make Fee payment for selected subjects (Rs. 500/- per theory subject + Rs. 200/- per practical subject).
- Step 11: Upload relevant and requisite documents.
- Step 12 & 13: Seats allotted as per availability. If dates not available -> Step 13: Try again for another date without paying fee again.
- Post-Booking: Login to dashboard -> Wait for approval -> Download Identity Card and Hall Ticket from dashboard -> Contact Exam Centre for practical exams.
Rules: Existing learners can re-book dates from dashboard after appearing. Cannot register for ODE twice in the same month for the same subject.""",
        archetype="workflow_procedure",
        page=14,
        title="Online Registration Process for On-Demand Examination (ODE)",
        section="4.2 On Demand Examination System"
    )

    # ==========================================
    # 3. CRITICAL POLICY CALLOUTS & RULES
    # ==========================================

    # Two-Year Gap Rule (Page 29)
    add_chunk(
        """Important Note: Mandatory Two-Year Gap Requirement for Senior Secondary Certification

- Eligibility Condition: A learner admitted to the Senior Secondary Course must ensure an essential gap of two years from the year of passing the Secondary (Class X) examination for obtaining a Pass Certificate at the Senior Secondary level.
- Maximum Subject Appearance: If the learner does not have the requisite gap of two years, he/she can appear in a MAXIMUM OF FOUR SUBJECTS ONLY in the forthcoming April-May examination and subsequent October-November examination.
- Fee Forfeiture Rule: In case a learner registers for more than 4 subjects without having completed the 2-year gap, only the first 4 subjects will be considered for appearing. The fee paid for the first four opted subjects will be adjusted, and the remaining fee paid for other subjects will be FORFEITED. The learner becomes eligible to appear in the remaining subjects only after completing the two years gap.""",
        archetype="policy_callout",
        page=29,
        title="Important Note: Essential Two-Year Gap Requirement for Senior Secondary",
        section="4.1 Public Examinations"
    )

    # Provisions for PWD (Chapter 6)
    add_chunk(
        """Provisions for Learners with Disabilities (PWD) in NIOS Examinations (Chapter 6)

- Compensatory Additional Time: An additional 20 minutes per hour of examination for each subject is provided to compensate for reading, fatigue, dictation, etc. If medically certified, a rest break of not more than 10 minutes per hour is permitted.
- Amanuensis (Scribe) Rules: Eligible for blindness, low vision, autism, cerebral palsy, intellectual disability, multiple disabilities, hearing/speech impairment, or sudden hand injury.
- Amanuensis Qualification: Scribe should not be related to examinee and cannot be from the same subject. If from same stream, scribe's educational qualification must be at least one class lower than the learner.
- Amanuensis Fee: Remuneration of Rs. 100/- per session is paid to the amanuensis by the Centre Superintendent. The candidate is NOT required to pay any fee. (Learners with temporary disability pay the fee themselves).
- Alternative Questions: Blind and low vision candidates receive alternate theory questions in lieu of map pointing, geometry diagrams, graphs, and drawings.
- Use of Computers: Allowed with assistive hardware/text reading software with prior permission from Regional Director.""",
        archetype="policy_callout",
        page=39,
        title="Examination Accommodations and Scribe Rules for Persons with Disabilities",
        section="6. General and Specific Provisions (Disability wise)"
    )

    # Transfer of Credit (TOC) Rules (Section 2.6)
    add_chunk(
        """Transfer of Credit (TOC) Policy and Requirements (Section 2.6)

- Eligibility: Ex-Secondary/Senior Secondary learners who passed at least one subject but failed/did not qualify the overall course from a recognised Board.
- Maximum Subjects: TOC is allowed for a MAXIMUM OF TWO SUBJECTS.
- Validity Window: Subjects must have been passed with at least 33% marks in the last five years (i.e., passed during 2017 or later).
- Practicals: For Senior Secondary, TOC in subjects with practicals is allowed only when passed in theory and practical separately.
- Submission Requirements: Original failed marksheet must be sent to the Regional Centre along with the Admission Form within 10 days of online admission.
- Late Fee: Submission after 10 days incurs a late fee of Rs. 500/- per subject charged through e-service.
- TOC Fee: Rs. 180/- per subject (Rs. 150/- under general correction if applied during admission).""",
        archetype="policy_callout",
        page=17,
        title="Transfer of Credit (TOC) Eligibility and Conditions",
        section="2.6 Transfer of Credit (TOC)"
    )

    # ==========================================
    # 4. PROCESS REMAINING TEXT FROM PDF (FAQs, Chapters)
    # ==========================================
    print("Processing prospectus body text and FAQs...")
    
    # Extract FAQs explicitly (Pages 4-5 in PDF index)
    faq_text = reader.pages[4].extract_text() + "\n" + reader.pages[5].extract_text()
    faq_items = re.findall(r'(\d+\.\s+What[^\n\?]+\?.*?)(?=\n\d+\.|\Z)', faq_text, re.DOTALL)
    for item in faq_items:
        first_line = item.strip().split('\n')[0]
        q_title = clean_text(first_line[:80])
        add_chunk(item, archetype="faq_item", page=4, title=f"FAQ: {q_title}", section="Frequently Asked Questions")

    # Ingest chapter prose cleanly by section
    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        # Skip pages already handled as full high-fidelity structured tables/charts
        if page_num in [2, 5, 6, 7, 13, 14, 16, 19, 20, 23, 29, 32, 33, 34, 37, 39, 43]:
            continue
            
        raw_text = page.extract_text()
        if not raw_text or len(raw_text.strip()) < 80:
            continue

        paragraphs = raw_text.split('\n\n')
        curr_buffer = ""
        
        for p in paragraphs:
            cleaned = clean_text(p)
            if not cleaned:
                continue
            if len(curr_buffer) + len(cleaned) < 1400:
                curr_buffer += "\n" + cleaned
            else:
                if len(curr_buffer) > 150:
                    add_chunk(
                        curr_buffer,
                        archetype="prose_section",
                        page=page_num,
                        title=f"Prospectus Content Page {page_num}",
                        section="General Prospectus Policy"
                    )
                curr_buffer = cleaned
                
        if len(curr_buffer) > 150:
            add_chunk(
                curr_buffer,
                archetype="prose_section",
                page=page_num,
                title=f"Prospectus Content Page {page_num}",
                section="General Prospectus Policy"
            )

    print(f"Generated {len(chunks)} cleanly structured chunks.")

    # Save to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Wrote output to {OUTPUT_JSON}")

if __name__ == "__main__":
    build_structured_knowledge_base()