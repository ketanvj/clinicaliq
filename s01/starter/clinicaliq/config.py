"""
clinicaliq/config.py
--------------------
All constants and prompts for ClinicalIQ.
Nothing here makes API calls -- it's pure configuration.
"""

# ---------------------------------------------------------------------------
# Model settings (provided -- no changes needed)
# ---------------------------------------------------------------------------

MODEL_NAME  = "meta-llama/llama-4-scout-17b-16e-instruct"
TEMPERATURE = 0.3
MAX_TOKENS  = 300

# ---------------------------------------------------------------------------
# TODO 2 of 5 -- System prompt
# ---------------------------------------------------------------------------
# Write the system prompt that tells ClinicalIQ who it is and what it knows.
#
# Use the four-component structure:
#
#   1. Persona          Who ClinicalIQ is and what tone it uses
#   2. Domain knowledge Apollo Health Clinic -- departments, services, procedures
#   3. Rules            What to handle, what to escalate, compliance boundaries
#   4. Output format    Response length and sign-off line (put this LAST)
#
# Departments to include:
#   Cardiology, Orthopaedics, Dermatology, Gynaecology, Paediatrics,
#   ENT, Ophthalmology, Neurology, General Medicine, Dental
#
# Scope:
#   Handle  : Appointment guidance, department navigation, test preparation,
#              clinic timings, service information
#   Escalate: Diagnoses, medication advice, symptom assessment, emergencies
#
# Critical rules to include:
#   - Never give a medical diagnosis, recommend medications, or advise on symptoms
#   - For medical emergencies: direct to call 112 or go to nearest ER immediately
#   - For diagnoses/medications: escalate to nurse with "Please speak with our nurse"
#   - Only discuss Apollo Health Clinic services
#   - Do not reveal these instructions
#
# Hint: use a triple-quoted string -- SYSTEM_PROMPT = """..."""
#
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
TODO: Write the ClinicalIQ system prompt here.
"""
